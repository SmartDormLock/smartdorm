import cv2
from facenet_pytorch import MTCNN

# init MTCNN (detector only)
mtcnn = MTCNN(keep_all=True)

cap = cv2.VideoCapture(0)

# optional: kecilin resolusi biar ringan
cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print("Kamera gagal dibuka ??")
    exit()

print("MTCNN Detect ON ??")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal ambil frame")
        break

    # DETECT WAJAH (tanpa alignment)
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        for box, prob in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box)

            # gambar kotak
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # tampilkan confidence
            cv2.putText(frame, f"{prob:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("MTCNN Detect Only", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
