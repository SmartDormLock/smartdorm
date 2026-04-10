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

# resolusi display (gede)
cap.set(3, 640)
cap.set(4, 480)

print("Detect + Landmark + Label (CLEAN UI) ??")

# label landmark
labels = ["L_Eye", "R_Eye", "Nose", "Mouth_L", "Mouth_R"]

# window resizeable
cv2.namedWindow("MTCNN Clean UI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MTCNN Clean UI", 900, 700)

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # ================== PROCESSING (KECIL BIAR RINGAN) ==================
    small = cv2.resize(frame, (240, 180))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    boxes, probs, landmarks = mtcnn.detect(rgb_small, landmarks=True)

    # ================== SCALE ==================
    h_ratio = frame.shape[0] / small.shape[0]
    w_ratio = frame.shape[1] / small.shape[1]

    # ================== DRAW ==================
    if boxes is not None:
        for box, prob, landmark in zip(boxes, probs, landmarks):

            # scale box
            x1, y1, x2, y2 = box
            x1 = int(x1 * w_ratio)
            y1 = int(y1 * h_ratio)
            x2 = int(x2 * w_ratio)
            y2 = int(y2 * h_ratio)

            # kotak wajah
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # ================== CONFIDENCE ==================
            text_conf = f"{prob:.2f}"

            # outline hitam
            cv2.putText(frame, text_conf, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
            # teks putih
            cv2.putText(frame, text_conf, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # ================== LANDMARK ==================
            for i, (x, y) in enumerate(landmark):
                x = int(x * w_ratio)
                y = int(y * h_ratio)

                # titik merah
                cv2.circle(frame, (x, y), 3, (0,0,255), -1)

                # label putih + outline
                label = labels[i]

                cv2.putText(frame, label, (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)

                cv2.putText(frame, label, (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    # ================== SHOW ==================
    cv2.imshow("MTCNN Clean UI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()
