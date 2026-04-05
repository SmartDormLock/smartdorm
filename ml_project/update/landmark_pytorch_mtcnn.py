import cv2
import torch
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
cap.set(3, 320)
cap.set(4, 240)

print("Detect + Landmark + Label ON 🔥")

# label nama titik
labels = ["L_Eye", "R_Eye", "Nose", "Mouth_L", "Mouth_R"]

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

            # kotak wajah
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # confidence
            cv2.putText(frame, f"{prob:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # ================== LANDMARK + LABEL ==================
            for i, (x, y) in enumerate(landmark):
                x, y = int(x), int(y)

                # titik merah
                cv2.circle(frame, (x, y), 2, (0,0,255), -1)

                # label
                cv2.putText(frame, labels[i], (x+3, y-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)

    cv2.imshow("MTCNN Landmark + Label", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
