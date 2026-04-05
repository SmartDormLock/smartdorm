import cv2
import os
import time
from facenet_pytorch import MTCNN

# ================== CONFIG ==================
poses = ["lurus", "kanan", "kiri", "atas", "bawah"]
photos_per_pose = 10
delay_per_photo = 0.5  # detik

# ================== USER INPUT ==================
user_name = input("Masukkan nama user: ").strip()
dataset_dir = os.path.join("dataset", user_name)
os.makedirs(dataset_dir, exist_ok=True)
for pose in poses:
    os.makedirs(os.path.join(dataset_dir, pose), exist_ok=True)

# ================== INIT ==================
mtcnn = MTCNN(keep_all=False)
cap = cv2.VideoCapture(0)

detect_on = False
current_pose_idx = 0
capture_count = 0
waiting_for_next = False

# ================== MAIN LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        print("No camera frame detected!")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================== DETECTION ==================
    if detect_on and not waiting_for_next:
        boxes, probs = mtcnn.detect(rgb_frame)
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # SAVE IMAGE
            if capture_count < photos_per_pose:
                pose_name = poses[current_pose_idx]
                filename = os.path.join(dataset_dir, pose_name, f"{pose_name}_{capture_count+1}.jpg")
                cv2.imwrite(filename, frame)
                capture_count += 1
                time.sleep(delay_per_photo)
                if capture_count >= photos_per_pose:
                    waiting_for_next = True
                    print(f"[DONE] Pose '{pose_name}' capture complete. Press 'S' to continue.")

    # ================== UI TEXT ==================
    pose_text = poses[current_pose_idx] if current_pose_idx < len(poses) else "Selesai"
    status_text = "Waiting user to continue..." if waiting_for_next else "Capturing..."
    cv2.putText(frame, f"User: {user_name} | Pose: {pose_text} | Foto: {capture_count}/{photos_per_pose} | {status_text}",
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, "Press 'S' to Start/Next Pose | 'Q' to Stop | ESC to Exit",
                (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # ================== KEY HANDLER ==================
    if key == 27:  # ESC
        print("Exiting...")
        break
    elif key == ord('s'):
        if not detect_on:
            detect_on = True
            current_pose_idx = 0
            capture_count = 0
            waiting_for_next = False
            print(f"[START] Capture started for {user_name}")
        elif waiting_for_next:
            current_pose_idx += 1
            capture_count = 0
            waiting_for_next = False
            if current_pose_idx >= len(poses):
                print(f"[ALL DONE] Dataset capture for {user_name} finished!")
                detect_on = False
    elif key == ord('q'):
        detect_on = False
        waiting_for_next = False
        print("[STOP] Capture paused. Press 'S' to resume.")

cap.release()
cv2.destroyAllWindows()
