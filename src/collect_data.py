import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance as dist
import csv

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
    center = (frame_size[1] / 2, frame_size[0] / 2)

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

    return angles  # pitch, yaw, roll

# -------- PATH --------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")

# -------- LOAD --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# -------- CSV SETUP --------
csv_file = os.path.join(base_dir, "dataset.csv")

# Create file with header if not exists
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["EAR", "Yaw", "Label"])  # Label: 1=Attentive, 0=Not

# -------- WEBCAM --------
cap = cv2.VideoCapture(0)

print("Press A for Attentive, D for Not Attentive, ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        pitch, yaw, roll = get_head_pose(landmarks, frame.shape)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):  # Attentive
            label = 1
        elif key == ord('d'):  # Not attentive
            label = 0
        else:
            label = None

        if label is not None:
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ear, yaw, label])
            print(f"Saved: EAR={ear:.2f}, Yaw={yaw:.2f}, Label={label}")

    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()