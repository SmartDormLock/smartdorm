import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera gagal dibuka ??")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame gagal diambil")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC buat keluar
        break

cap.release()
cv2.destroyAllWindows()
