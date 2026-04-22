import os
import cv2
import time
import pickle
import torch
import shutil
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import lgpio

# ================== RELAY SETUP ==================
RELAY_PIN = 27

h = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(h, RELAY_PIN)

# default LOCK
lgpio.gpio_write(h, RELAY_PIN, 0)

def buka_pintu():
    print("🚪 Membuka pintu...")
    lgpio.gpio_write(h, RELAY_PIN, 1)

def kunci_pintu():
    print("🔒 Mengunci pintu...")
    lgpio.gpio_write(h, RELAY_PIN, 0)

def open_door():
    try:
        buka_pintu()
        time.sleep(5)
    finally:
        kunci_pintu()
        print("Pintu terkunci kembali")

# ================== INIT ==================
mtcnn = MTCNN(keep_all=False)
model = InceptionResnetV1(pretrained='vggface2').eval()

DATASET_PATH = "dataset"
EMBED_PATH = "embeddings.pkl"

os.makedirs(DATASET_PATH, exist_ok=True)

# ================== UTILS ==================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_database():
    if os.path.exists(EMBED_PATH):
        with open(EMBED_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_database(data):
    with open(EMBED_PATH, "wb") as f:
        pickle.dump(data, f)

# ================== MENU ==================
def show_menu():
    print("\n====== FACE SYSTEM ======")
    print("[1] Tambah user")
    print("[2] Scan wajah")
    print("[3] Lihat user")
    print("[4] Hapus user")
    print("[5] Rebuild embedding")
    print("[q] Keluar")

# ================== ADD USER ==================
def add_user():
    poses = ["lurus", "kanan", "kiri", "atas", "bawah"]
    photos_per_pose = 10
    delay_per_photo = 0.5

    user_name = input("Masukkan nama user: ").strip()
    dataset_dir = os.path.join(DATASET_PATH, user_name)

    for pose in poses:
        os.makedirs(os.path.join(dataset_dir, pose), exist_ok=True)

    cap = cv2.VideoCapture(0)

    print("\n[INFO] Tekan 'S' untuk mulai capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Generate embedding...")
    build_embedding()

# ================== BUILD EMBEDDING ==================
def build_embedding():
    data = {}

    for user in os.listdir(DATASET_PATH):
        user_path = os.path.join(DATASET_PATH, user)

        if not os.path.isdir(user_path):
            continue

        embeddings = []

        for pose in os.listdir(user_path):
            pose_path = os.path.join(user_path, pose)

            for img_name in os.listdir(pose_path):
                img_path = os.path.join(pose_path, img_name)

                img = Image.open(img_path).convert("RGB")
                img = img.resize((160,160))

                img = np.array(img)/255.0
                img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

                with torch.no_grad():
                    emb = model(img).squeeze().numpy()

                embeddings.append(emb)

        data[user] = embeddings
        print(f"{user} → {len(embeddings)} embeddings")

    save_database(data)
    print("Embedding updated")

# ================== SCAN ==================
def scan_face():
    database = load_database()
    cap = cv2.VideoCapture(0)

    print("Scan wajah ON")

    last_open_time = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(rgb)

        if boxes is not None:
            x1,y1,x2,y2 = map(int, boxes[0])
            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                face = cv2.resize(face, (160,160))

                img = np.array(face)/255.0
                img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

                with torch.no_grad():
                    emb = model(img).squeeze().numpy()

                best_name = "UNKNOWN"
                best_score = -1

                for name, embs in database.items():
                    for e in embs:
                        score = cosine_similarity(emb, e)
                        if score > best_score:
                            best_score = score
                            best_name = name

                if best_score < 0.7:
                    best_name = "UNKNOWN"

                label = f"{best_name} ({best_score:.2f})"

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                # 🔥 TRIGGER RELAY
                if best_name != "UNKNOWN":
                    if time.time() - last_open_time > 5:
                        print(f"✅ AKSES: {best_name}")
                        open_door()
                        last_open_time = time.time()

        cv2.imshow("Scan", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ================== LIST ==================
def list_users():
    users = os.listdir(DATASET_PATH)
    print("\nUser:")
    for u in users:
        print("-", u)

# ================== DELETE ==================
def delete_user():
    name = input("Nama user: ")
    path = os.path.join(DATASET_PATH, name)

    if os.path.exists(path):
        shutil.rmtree(path)
        print("User dihapus")
        build_embedding()
    else:
        print("User tidak ditemukan")

# ================== MAIN ==================
try:
    while True:
        show_menu()
        cmd = input("Pilih: ")

        if cmd == "1":
            add_user()
        elif cmd == "2":
            scan_face()
        elif cmd == "3":
            list_users()
        elif cmd == "4":
            delete_user()
        elif cmd == "5":
            build_embedding()
        elif cmd.lower() == "q":
            break
        else:
            print("Invalid")

except KeyboardInterrupt:
    print("\nDihentikan user")

finally:
    lgpio.gpio_write(h, RELAY_PIN, 0)
    lgpio.gpiochip_close(h)
