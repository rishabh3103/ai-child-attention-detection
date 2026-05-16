import cv2
import dlib
import os

# Get base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to shape predictor
predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")

# Load detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not working")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Draw all 68 points
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Facial Landmarks", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()