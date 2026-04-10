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

# ================== CAMERA ==================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(3, 640)
cap.set(4, 480)

print("Alignment + Clean Display ??")

# ================== WINDOW ==================
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", 900, 700)

cv2.namedWindow("Aligned Face", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aligned Face", 300, 300)

# ================== ALIGN FUNCTION ==================
def align_face(img, landmarks):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (int((left_eye[0] + right_eye[0]) / 2),
              int((left_eye[1] + right_eye[1]) / 2))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return aligned

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ?? mirror biar natural (optional tapi recommended)
    frame = cv2.flip(frame, 1)

    # ================== PROCESSING KECIL ==================
    small = cv2.resize(frame, (240, 180))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes, probs, landmarks = mtcnn.detect(rgb_small, landmarks=True)

    # scale ratio
    h_ratio = frame.shape[0] / small.shape[0]
    w_ratio = frame.shape[1] / small.shape[1]

    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):

            # scale box
            x1, y1, x2, y2 = box
            x1 = int(x1 * w_ratio)
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio)
            y2 = int(y2 * h_ratio)

            # clamp biar aman
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # scale landmark
            scaled_landmark = []
            for (x, y) in landmark:
                scaled_landmark.append((x * w_ratio, y * h_ratio))
            scaled_landmark = np.array(scaled_landmark)

            # ================== DRAW BOX ==================
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # ================== ALIGN ==================
            aligned = align_face(frame, scaled_landmark)

            # ================== CROP ==================
            face_crop = aligned[y1:y2, x1:x2]

            if face_crop.size != 0:
                # resize biar konsisten (FaceNet nanti butuh 160x160)
                face_crop = cv2.resize(face_crop, (160,160))

                cv2.imshow("Aligned Face", face_crop)

    # ================== SHOW ==================
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
