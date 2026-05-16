import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# -------- PATH --------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "dataset", "train")
results_path = os.path.join(base_dir, "results", "cnn")

os.makedirs(results_path, exist_ok=True)

# -------- IMAGE SETTINGS --------
img_size = (48, 48)
batch_size = 32

# -------- LOAD DATA --------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -------- NORMALIZATION --------
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

# -------- BUILD CNN --------
model = models.Sequential([
    layers.Input(shape=(48,48,1)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# -------- COMPILE --------
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -------- TRAIN --------
print("Training CNN...")
history = model.fit(train_ds, epochs=5)

# -------- SAVE MODEL --------
model_path = os.path.join(base_dir, "models", "cnn_emotion.h5")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

# -------- ACCURACY GRAPH --------
plt.plot(history.history['accuracy'])
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(results_path, "accuracy.png"))
plt.close()

# -------- LOSS GRAPH --------
plt.plot(history.history['loss'])
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(results_path, "loss.png"))
plt.close()

# -------- CONFUSION MATRIX --------
y_true = []
y_pred = []

for images, labels in train_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN Confusion Matrix")
plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
plt.close()

# -------- CLASSIFICATION REPORT --------
report = classification_report(y_true, y_pred, target_names=class_names)

with open(os.path.join(results_path, "classification_report.txt"), "w") as f:
    f.write(report)

print("✅ CNN Model + Metrics Saved in /results/cnn/")