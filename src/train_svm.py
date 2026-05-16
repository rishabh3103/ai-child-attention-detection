import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, f1_score
)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

# -------- PATH --------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(base_dir, "dataset.csv")
model_path = os.path.join(base_dir, "models", "svm_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

# -------- RESULT FOLDER --------
results_path = os.path.join(base_dir, "results", "svm")
os.makedirs(results_path, exist_ok=True)

print("CSV PATH:", csv_path)

# -------- LOAD DATA --------
data = pd.read_csv(csv_path)
print("Dataset Loaded:", data.shape)

# -------- CLEAN DATA --------
data = data[(data['EAR'] > 0.1) & (data['EAR'] < 0.5)]

# -------- FEATURES --------
X = data[['EAR', 'Yaw']]
y = data['Label']

# -------- NORMALIZATION --------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -------- STRATIFIED SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------- TRAIN MODEL --------
print("Training SVM...")
model = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train, y_train)

# -------- PREDICTION --------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------- METRICS --------
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n🎯 Accuracy:", accuracy)
print("\n📊 Classification Report:\n", report)

# -------- SAVE TEXT --------
with open(os.path.join(results_path, "accuracy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}")

with open(os.path.join(results_path, "classification_report.txt"), "w") as f:
    f.write(report)

with open(os.path.join(results_path, "f1_score.txt"), "w") as f:
    f.write(f"F1 Score: {f1}")

# -------- TRAIN vs TEST --------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

with open(os.path.join(results_path, "train_test_accuracy.txt"), "w") as f:
    f.write(f"Train Accuracy: {train_acc}\nTest Accuracy: {test_acc}")

# -------- CONFUSION MATRIX --------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(results_path, "confusion_matrix.png"))
plt.close()

# -------- ROC CURVE --------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(results_path, "roc_curve.png"))
plt.close()

# -------- PRECISION-RECALL CURVE --------
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.savefig(os.path.join(results_path, "pr_curve.png"))
plt.close()

# -------- LEARNING CURVE (STRATIFIED) --------
cv = StratifiedKFold(n_splits=5)

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=cv, n_jobs=-1
)

plt.figure()
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Test")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.savefig(os.path.join(results_path, "learning_curve.png"))
plt.close()

# -------- SAVE MODEL --------
os.makedirs(os.path.dirname(model_path), exist_ok=True)

pickle.dump(model, open(model_path, "wb"))
pickle.dump(scaler, open(scaler_path, "wb"))

print("\n✅ Clean Model + All Metrics Saved in /results/svm/")