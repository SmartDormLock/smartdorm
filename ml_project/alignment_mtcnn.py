import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)

cap.set(3, 480)
cap.set(4, 360)

print("STEP 2: ALIGNMENT ??")

def align_face(img, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # hitung perbedaan posisi
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    # pusat gambar
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # rotate
    M = cv2.getRotationMatrix2D(center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned

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

        # ================= ALIGN =================
        aligned = align_face(face_crop, keypoints)

        # tampilkan
        cv2.imshow("Aligned Face", aligned)

        # kotak
        cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Camera", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
