import cv2
import torch
import numpy as np
import pickle
import time
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# ================== CONFIG ==================
FRAME_SKIP = 3
EMBEDDING_DELAY = 0.5
THRESHOLD = 0.7

# ukuran display (ikut window lu)
DISPLAY_WIDTH = 900
DISPLAY_HEIGHT = 700

# ================== INIT ==================
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

with open("embeddings.pkl", "rb") as f:
    database = pickle.load(f)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

# ================== UTILS ==================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize(face_embedding):
    best_name = "UNKNOWN"
    best_score = -1

    for name, embs in database.items():
        for emb in embs:
            score = cosine_similarity(face_embedding, emb)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score > THRESHOLD:
        return best_name, best_score
    return "UNKNOWN", best_score

# ================== STATE ==================
frame_count = 0
last_embedding_time = 0

cached_box = None
cached_label = "Detecting..."

print("🔥 Optimized Recognition Running")

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # mirror biar natural
    frame = cv2.flip(frame, 1)

    frame_count += 1

    # ================== DETECTION ==================
    if frame_count % FRAME_SKIP == 0:

        small = cv2.resize(frame, (240,180))
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(rgb_small)

        if boxes is not None:
            box = boxes[0]

            # scale ke frame asli
            h_ratio = frame.shape[0] / 180
            w_ratio = frame.shape[1] / 240

            x1, y1, x2, y2 = box
            x1 = int(x1 * w_ratio)
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio)
            y2 = int(y2 * h_ratio)

            cached_box = (x1, y1, x2, y2)

    # ================== EMBEDDING ==================
    if cached_box is not None and time.time() - last_embedding_time > EMBEDDING_DELAY:
        x1, y1, x2, y2 = cached_box

        face = frame[y1:y2, x1:x2]

        if face.size != 0:
            face = cv2.resize(face, (160,160))

            face_img = Image.fromarray(face).convert("RGB")
            face_np = np.array(face_img) / 255.0
            face_tensor = torch.tensor(face_np).permute(2,0,1).float().unsqueeze(0)

            with torch.no_grad():
                emb = model(face_tensor).squeeze().numpy()

            name, score = recognize(emb)
            cached_label = f"{name} ({score:.2f})"

        last_embedding_time = time.time()

    # ================== DRAW ==================
    if cached_box is not None:
        x1, y1, x2, y2 = cached_box

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # outline
        cv2.putText(frame, cached_label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)

        # text putih
        cv2.putText(frame, cached_label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # ================== 🔥 FIX DISPLAY ==================
    frame_display = cv2.resize(
        frame,
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )

    cv2.imshow("Optimized Face Recognition", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
