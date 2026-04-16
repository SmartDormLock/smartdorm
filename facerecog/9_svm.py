import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ================== LOAD EMBEDDING ==================
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

X = []
y = []

# ================== PREPARE DATA ==================
for name, embeddings in data.items():
    for emb in embeddings:
        X.append(emb)
        y.append(name)

X = np.array(X)
y = np.array(y)

print(f"Total data: {len(X)}")
print(f"Classes: {np.unique(y)}")

# ================== SPLIT DATA ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================== TRAIN SVM ==================
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

print("\n?? SVM trained")

# ================== PREDICT ==================
y_pred = model.predict(X_test)

# ================== METRICS ==================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

cm = confusion_matrix(y_test, y_pred)

print("\n=== EVALUATION RESULT ===")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix (angka):")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ================== VISUAL CONFUSION MATRIX ==================
labels = np.unique(y)

plt.figure(figsize=(6,5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (SVM)")

plt.tight_layout()

# SAVE GAMBAR
plt.savefig("confusion_matrix_svm.png")

print("\n?? Confusion matrix saved as confusion_matrix_svm.png")

plt.show()

# ================== SAVE MODEL ==================
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("?? Model saved as svm_model.pkl")
