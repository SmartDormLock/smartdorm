import cv2
from mtcnn import MTCNN

# init
detector = MTCNN()
cap = cv2.VideoCapture(0)

# resolusi kamera (tetap agak gede biar display enak)
cap.set(3, 640)
cap.set(4, 480)

# optimasi
cv2.setUseOptimized(True)

frame_count = 0
faces = []
last_faces = []

print("MTCNN Optimized Camera ON 🔥")

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    # ================= RESIZE UNTUK DETECTION =================
    small = cv2.resize(frame, (320, 240))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # ================= DETECT TIAP BEBERAPA FRAME =================
    if frame_count % 5 == 0:
        try:
            faces = detector.detect_faces(rgb)
            last_faces = faces
        except:
            faces = last_faces
    else:
        faces = last_faces

    # ================= SCALE BALIK KE FRAME ASLI =================
    frame_display = frame.copy()
    h_frame, w_frame, _ = frame.shape

    scale_x = w_frame / 320
    scale_y = h_frame / 240

    for face in faces:
        x, y, w, h = face['box']

        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # clamp biar aman
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # ================= DISPLAY =================
    cv2.imshow("MTCNN Optimized", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
