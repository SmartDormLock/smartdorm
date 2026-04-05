import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

# ================== STABILITY ==================
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ================== INIT ==================
mtcnn = MTCNN(
    keep_all=True,
    device='cpu',
    min_face_size=80
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 320)
cap.set(4, 240)

print("Alignment ON 🔥")

# ================== FUNCTION ALIGN ==================
def align_face(img, landmarks):
    # ambil mata
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # hitung sudut
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # tengah wajah
    center = (int((left_eye[0] + right_eye[0]) / 2),
              int((left_eye[1] + right_eye[1]) / 2))

    # rotate
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (240, 180))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs, landmarks = mtcnn.detect(rgb_frame, landmarks=True)

    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):
            x1, y1, x2, y2 = map(int, box)

            # bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # ================== ALIGN ==================
            aligned = align_face(frame, landmark)

            # crop wajah setelah align
            face_crop = aligned[y1:y2, x1:x2]

            # tampilkan hasil align
            if face_crop.size != 0:
                cv2.imshow("Aligned Face", face_crop)

    cv2.imshow("Original", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
