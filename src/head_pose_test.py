import cv2
import dlib
import numpy as np
import os

# -------- PATH --------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")

# -------- LOAD --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# -------- CAMERA MATRIX (approximation) --------
def get_head_pose(shape, frame_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye corner
        (shape.part(45).x, shape.part(45).y),  # Right eye corner
        (shape.part(48).x, shape.part(48).y),  # Left mouth
        (shape.part(54).x, shape.part(54).y)   # Right mouth
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

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    return rotation_vector

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

        rotation_vector = get_head_pose(landmarks, frame.shape)

        # Convert to angles
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        pitch, yaw, roll = angles

        # Display
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Logic
        if abs(yaw) < 10:
            status = "Looking Forward"
        else:
            status = "Looking Away"

        cv2.putText(frame, status, (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("Head Pose", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()