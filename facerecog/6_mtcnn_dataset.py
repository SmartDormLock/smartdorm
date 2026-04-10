import cv2
import os
import time
import numpy as np
from facenet_pytorch import MTCNN

# ================== CONFIG ==================
poses = ["lurus", "kanan", "kiri", "atas", "bawah"]
photos_per_pose = 10
delay_per_photo = 0.5

# ================== USER INPUT ==================
user_name = input("Masukkan nama user: ").strip()
dataset_dir = os.path.join("dataset", user_name)

for pose in poses:
    os.makedirs(os.path.join(dataset_dir, pose), exist_ok=True)

# ================== INIT ==================
mtcnn = MTCNN(keep_all=False)
cap = cv2.VideoCapture(0)

detect_on = False
current_pose_idx = 0
capture_count = 0
waiting_for_next = False

# ================== ALIGN FUNCTION ==================
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
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera frame detected!")
        break

    # mirror biar natural
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================== DETECTION ==================
    if detect_on and not waiting_for_next:
        boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)

        if boxes is not None:
            box = boxes[0]
            landmark = landmarks[0]

            x1, y1, x2, y2 = [int(b) for b in box]

            # clamp biar aman
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            if capture_count < photos_per_pose:

                # ================== ALIGN ==================
                aligned = align_face(frame, landmark)

                # ================== CROP ==================
                face = aligned[y1:y2, x1:x2]

                if face.size != 0:
                    face = cv2.resize(face, (160,160))

                    pose_name = poses[current_pose_idx]
                    pose_path = os.path.join(dataset_dir, pose_name)

                    # hitung file existing
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
                    print(f"[DONE] Pose '{pose_name}' selesai. Next: {next_pose} (tekan S)")

    # ================== UI ==================
    if waiting_for_next:
        status_text = f"Next: {poses[current_pose_idx+1]}" if current_pose_idx+1 < len(poses) else "All done!"
    elif detect_on:
        status_text = f"Capturing '{poses[current_pose_idx]}'"
    else:
        status_text = "Press S to start"

    cv2.putText(frame,
                f"{user_name} | Pose: {poses[current_pose_idx]} | {capture_count}/{photos_per_pose}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)

    cv2.putText(frame,
                f"{user_name} | Pose: {poses[current_pose_idx]} | {capture_count}/{photos_per_pose}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame,
                status_text,
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================== KEY ==================
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

    elif key == ord('q'):
        detect_on = False
        waiting_for_next = False
        print("[STOP] paused")

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()