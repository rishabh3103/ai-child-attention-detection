import cv2
import os

# Get base directory (project folder)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to haarcascade file
cascade_path = os.path.join(base_dir, 'haarcascade.xml')

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if loaded properly
if face_cascade.empty():
    print("Error: Haarcascade file not loaded. Check path.")
    exit()

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

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangle on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Face Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
cap.release()
cv2.destroyAllWindows()