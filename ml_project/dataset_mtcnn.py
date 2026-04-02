import cv2
import os
import numpy as np
from mtcnn import MTCNN

# ================= INPUT =================
name = input("Masukkan nama user: ")
dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

# ================= INIT =================
detector = MTCNN()
cap = cv2.VideoCapture(0)

cap.set(3, 480)
cap.set(4, 360)

count = 0
saving = False

print("S = Start Save | E = Stop Save | ESC = Exit")

# ================= ALIGN =================
def align_face(img, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))
    center = (img.shape[1] // 2, img.shape[0] // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except:
        faces = []

    for face in faces:
        x, y, w, h = face['box']
        keypoints = face['keypoints']

        x = max(0, x)
        y = max(0, y)

        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        aligned = align_face(face_crop, keypoints)
        face_resized = cv2.resize(aligned, (160, 160))

        # tampilkan wajah final
        cv2.imshow("Face (160x160)", face_resized)

        # gambar kotak
        cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)

        # ================= SAVE =================
        if saving:
            filename = f"{dataset_path}/{count}.jpg"
            cv2.imwrite(filename, face_resized)
            print("Saved:", filename)
            count += 1

    # ================= STATUS TEXT =================
    status_text = "Saving: ON" if saving else "Saving: OFF"
    cv2.putText(display, status_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if saving else (0,0,255), 2)

    cv2.imshow("Camera", display)

    # ================= CONTROL =================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        saving = True
        print("Start Saving...")

    elif key == ord('e'):
        saving = False
        print("Stop Saving.")

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
