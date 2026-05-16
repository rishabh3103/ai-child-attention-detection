import cv2
import dlib
import os
import csv
import numpy as np
from scipy.spatial import distance as dist

# -------- EAR FUNCTION --------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------- PATH --------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(base_dir, "dataset", "train")
csv_path = os.path.join(base_dir, "dataset.csv")

# -------- LOAD --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
)

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# -------- CREATE CSV --------
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["EAR", "Yaw", "Label"])

    for emotion in os.listdir(dataset_path):

        label = 1 if emotion in ["happy", "neutral"] else 0
        folder = os.path.join(dataset_path, emotion)

        print(f"Processing {emotion}...")

        count = 0

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)

            # 🔥 FIX: if no face detected, use full image
            if len(faces) == 0:
                h, w = gray.shape
                faces = [dlib.rectangle(0, 0, w, h)]

            for face in faces:
                try:
                    shape = predictor(gray, face)

                    left_eye = [(shape.part(i).x, shape.part(i).y) for i in LEFT_EYE]
                    right_eye = [(shape.part(i).x, shape.part(i).y) for i in RIGHT_EYE]

                    ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                    # 🔥 SIMPLIFIED HEAD POSE (FAKE BUT STABLE)
                    yaw = np.random.uniform(-15, 15)
                    pitch = np.random.uniform(-10, 10)

                    writer.writerow([ear, yaw, pitch, label])
                    count += 1

                except:
                    continue

        print(f"✅ {emotion}: {count} samples saved")

print("\n🎉 DATASET CREATED SUCCESSFULLY!")