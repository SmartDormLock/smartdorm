import os
import cv2
import time
import pickle
import torch
import shutil
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

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

# ================== ADD USER (SESUAI DATASET LU) ==================
def add_user():
    poses = ["lurus", "kanan", "kiri", "atas", "bawah"]
    photos_per_pose = 10
    delay_per_photo = 0.5

    user_name = input("Masukkan nama user: ").strip()
    dataset_dir = os.path.join(DATASET_PATH, user_name)

    for pose in poses:
        os.makedirs(os.path.join(dataset_dir, pose), exist_ok=True)

    cap = cv2.VideoCapture(0)

    detect_on = False
    current_pose_idx = 0
    capture_count = 0
    waiting_for_next = False

    # ===== ALIGN FUNCTION =====
    def align_face(img, landmarks):
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = (
            int((left_eye[0] + right_eye[0]) / 2),
            int((left_eye[1] + right_eye[1]) / 2)
        )

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    print("\n[INFO] Tekan 'S' untuk mulai capture")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ===== DETECT =====
        if detect_on and not waiting_for_next:
            boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)

            if boxes is not None:
                box = boxes[0]
                landmark = landmarks[0]

                x1, y1, x2, y2 = [int(b) for b in box]

                # clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

                if capture_count < photos_per_pose:

                    aligned = align_face(frame, landmark)
                    face = aligned[y1:y2, x1:x2]

                    if face.size != 0:
                        face = cv2.resize(face, (160,160))

                        pose_name = poses[current_pose_idx]
                        pose_path = os.path.join(dataset_dir, pose_name)

                        existing = len(os.listdir(pose_path))

                        filename = os.path.join(
                            pose_path,
                            f"{pose_name}_{user_name}_{existing+1}.jpg"
                        )

                        cv2.imwrite(filename, face)

                        capture_count += 1
                        time.sleep(delay_per_photo)

                        print(f"[SAVE] {filename}")

                    if capture_count >= photos_per_pose:
                        waiting_for_next = True
                        next_pose = poses[current_pose_idx + 1] if current_pose_idx + 1 < len(poses) else "Selesai"
                        print(f"[DONE] Pose '{pose_name}' selesai → Next: {next_pose} (tekan S)")

        # ===== UI =====
        if waiting_for_next:
            status = f"Next: {poses[current_pose_idx+1]}" if current_pose_idx+1 < len(poses) else "All done!"
        elif detect_on:
            status = f"Capturing {poses[current_pose_idx]}"
        else:
            status = "Press S to start"

        cv2.putText(frame,
            f"{user_name} | {poses[current_pose_idx]} | {capture_count}/{photos_per_pose}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)

        cv2.putText(frame,
            f"{user_name} | {poses[current_pose_idx]} | {capture_count}/{photos_per_pose}",
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, status, (10,60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("Dataset Capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('s'):
            if not detect_on:
                detect_on = True
                current_pose_idx = 0
                capture_count = 0
                waiting_for_next = False
                print(f"[START] {user_name}")

            elif waiting_for_next:
                current_pose_idx += 1
                capture_count = 0
                waiting_for_next = False

                if current_pose_idx >= len(poses):
                    print(f"[DONE] Semua pose selesai untuk {user_name}")
                    detect_on = False
                    break

        elif key == ord('q'):
            detect_on = False
            waiting_for_next = False

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
