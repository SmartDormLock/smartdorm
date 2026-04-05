import cv2
import torch
from facenet_pytorch import MTCNN

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

mtcnn = MTCNN(
    keep_all=True,
    device='cpu',
    min_face_size=80
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(3, 320)
cap.set(4, 240)

print("Detect Bounding Box Only 🔥")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (240, 180))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{prob:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("MTCNN Bounding Box Only", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
