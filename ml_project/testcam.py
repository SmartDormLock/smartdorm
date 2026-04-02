import cv2

# load model face detection bawaan OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# set resolusi biar ringan
cap.set(3, 320)
cap.set(4, 240)

if not cap.isOpened():
    print("Kamera gagal dibuka ??")
    exit()

print("Camera + Face Detection ON ??")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal ambil frame")
        break

    # convert ke grayscale (wajib buat Haar)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect wajah
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # gambar kotak
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    # tekan ESC buat keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
