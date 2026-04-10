import cv2
import torch
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ================== LOAD MODEL ==================
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

# ================== LOAD EMBEDDING ==================
with open("embeddings.pkl", "rb") as f:
    database = pickle.load(f)

print("Database loaded:", list(database.keys()))

# ================== COSINE SIM ==================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ================== RECOGNITION ==================
def recognize(face_embedding, database, threshold=0.7):
    best_match = "UNKNOWN"
    best_score = -1

    for name, embeddings in database.items():
        for emb in embeddings:
            score = cosine_similarity(face_embedding, emb)

            if score > best_score:
                best_score = score
                best_match = name

    if best_score > threshold:
        return best_match, best_score
    else:
        return "UNKNOWN", best_score

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)

print("Real-time Recognition ON ??")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror biar natural
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)

    if boxes is not None:
        box = boxes[0]
        x1, y1, x2, y2 = [int(b) for b in box]

        face = frame[y1:y2, x1:x2]

        if face.size != 0:
            face = cv2.resize(face, (160,160))

            # ================== EMBEDDING ==================
            face_img = Image.fromarray(face).convert("RGB")
            face_np = np.array(face_img) / 255.0
            face_tensor = torch.tensor(face_np).permute(2,0,1).float().unsqueeze(0)

            with torch.no_grad():
                emb = model(face_tensor).squeeze().numpy()

            # ================== RECOGNITION ==================
            name, score = recognize(emb, database)

            # ================== DRAW ==================
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            label = f"{name} ({score:.2f})"

            # outline
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)

            # text putih
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
