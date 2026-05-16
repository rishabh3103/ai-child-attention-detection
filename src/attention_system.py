import cv2
import dlib
import numpy as np
import pandas as pd
import os
from scipy.spatial import distance as dist
import pickle
from tensorflow.keras.models import load_model

# -------- OPTIONAL: HIDE TF WARNINGS --------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -------- EAR FUNCTION --------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------- HEAD POSE --------
def get_head_pose(shape, frame_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),
        (shape.part(8).x, shape.part(8).y),
        (shape.part(36).x, shape.part(36).y),
        (shape.part(45).x, shape.part(45).y),
        (shape.part(48).x, shape.part(48).y),
        (shape.part(54).x, shape.part(54).y)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = frame_size[1]
    center = (frame_size[1]/2, frame_size[0]/2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles

# -------- PATH --------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
svm_model_path = os.path.join(base_dir, "models", "svm_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
cnn_model_path = os.path.join(base_dir, "models", "cnn_emotion.h5")

# -------- LOAD MODELS --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

svm_model = pickle.load(open(svm_model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
cnn_model = load_model(cnn_model_path)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# -------- WEBCAM --------
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # -------- EAR --------
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # -------- HEAD POSE --------
        pitch, yaw, roll = get_head_pose(landmarks, frame.shape)

        # -------- SVM ATTENTION --------
        features = pd.DataFrame([[ear, yaw]], columns=["EAR", "Yaw"])
        features_scaled = scaler.transform(features)

        prediction = svm_model.predict(features_scaled)
        attention = "Attentive ✅" if prediction[0] == 1 else "Not Attentive ❌"

        # -------- CNN MOOD (SINGLE BEST) --------
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y+h, x:x+w]

        try:
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img / 255.0
            face_img = np.reshape(face_img, (1, 48, 48, 1))

            preds = cnn_model.predict(face_img, verbose=0)

            # 🔥 FIX: Only best mood
            mood_index = np.argmax(preds)
            mood = emotion_labels[mood_index].capitalize()

        except:
            mood = "Unknown"

        # -------- DISPLAY --------
        cv2.putText(frame, f"EAR: {ear:.2f}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Mood: {mood}", (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Attention: {attention}", (30,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)

    cv2.imshow("Final System (SVM + CNN)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()