import customtkinter as ctk
import cv2
import dlib
import numpy as np
import pandas as pd
import os
import pickle
from scipy.spatial import distance as dist
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# -------- UI CONFIG --------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -------- FUNCTIONS --------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

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
    _, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    return angles

# -------- PATHS --------
base_dir = os.path.abspath(os.path.dirname(__file__))

predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
svm_model_path = os.path.join(base_dir, "models", "svm_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
cnn_model_path = os.path.join(base_dir, "models", "cnn_emotion.h5")
logo_path = os.path.join(base_dir, "logo.png")

# -------- LOAD MODELS --------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

svm_model = pickle.load(open(svm_model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))
cnn_model = load_model(cnn_model_path)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# -------- UI WINDOW --------
app = ctk.CTk()
app.title("AI Attention Detection System")
app.geometry("950x720")

# -------- HEADER --------
if os.path.exists(logo_path):
    logo_img = Image.open(logo_path).resize((80, 80))
    logo = ImageTk.PhotoImage(logo_img)
    logo_label = ctk.CTkLabel(app, image=logo, text="")
    logo_label.pack(pady=5)

title = ctk.CTkLabel(app, text="Child Attention Detection",
                     font=("Arial", 24, "bold"))
title.pack()

subtitle = ctk.CTkLabel(app, text="SVM + CNN Based System",
                        font=("Arial", 14))
subtitle.pack(pady=5)

# -------- VIDEO --------
video_label = ctk.CTkLabel(app, text="")
video_label.pack(pady=10)

status_label = ctk.CTkLabel(app, text="System Stopped", font=("Arial", 14))
status_label.pack()

running = False
cap = None

# -------- START --------
def start_system():
    global running, cap
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        status_label.configure(text="System Running")
        update_frame()

# -------- FRAME LOOP --------
def update_frame():
    global cap, running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        pitch, yaw, roll = get_head_pose(landmarks, frame.shape)

        features = pd.DataFrame([[ear, yaw]], columns=["EAR", "Yaw"])
        features_scaled = scaler.transform(features)

        attention = "Attentive" if svm_model.predict(features_scaled)[0] == 1 else "Not Attentive"

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = gray[y:y+h, x:x+w]

        try:
            face_img = cv2.resize(face_img, (48, 48)) / 255.0
            face_img = np.reshape(face_img, (1, 48, 48, 1))
            mood = emotion_labels[np.argmax(cnn_model.predict(face_img, verbose=0))].capitalize()
        except:
            mood = "Unknown"

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Mood: {mood}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Attention: {attention}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    app.after(10, update_frame)

# -------- EXIT --------
def exit_app():
    global running, cap
    running = False
    if cap:
        cap.release()
    app.destroy()

# -------- BUTTONS --------
ctk.CTkButton(app, text="Start Detection", width=200, height=45, command=start_system).pack(pady=10)
ctk.CTkButton(app, text="Exit", width=200, height=40, fg_color="red", command=exit_app).pack(pady=5)

# -------- RUN --------
app.mainloop()