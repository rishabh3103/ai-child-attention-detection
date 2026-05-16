import cv2
import dlib
import os
from scipy.spatial import distance as dist

# -------- EAR FUNCTION --------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -------- PATH SETUP --------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")

# -------- LOAD MODELS --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Eye landmark indexes
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# -------- WEBCAM --------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = []
        right_eye = []

        for i in LEFT_EYE:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            left_eye.append((x, y))

        for i in RIGHT_EYE:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            right_eye.append((x, y))

        # Calculate EAR
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        # Display EAR
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Check eye state
        if ear < 0.20:
            status = "Eyes Closed"
        else:
            status = "Eyes Open"

        cv2.putText(frame, status, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()