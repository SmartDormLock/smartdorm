import cv2
from mtcnn import MTCNN

# ================= INIT =================
detector = MTCNN()
cap = cv2.VideoCapture(0)

cap.set(3, 480)
cap.set(4, 360)

cv2.setUseOptimized(True)

tracker = None
tracking = False
frame_count = 0

print("MTCNN + Tracking ON ??")

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    frame_display = frame.copy()

    # ================= DETECT TIAP BEBERAPA FRAME =================
    if frame_count % 10 == 0 or not tracking:

        small = cv2.resize(frame, (224, 168))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        try:
            faces = detector.detect_faces(rgb)
        except:
            faces = []

        if len(faces) > 0:
            face = faces[0]

            x, y, w, h = face['box']

            # scale balik
            scale_x = frame.shape[1] / 224
            scale_y = frame.shape[0] / 168

            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            # clamp
            x = max(0, x)
            y = max(0, y)

            # ================= INIT TRACKER =================
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True

    # ================= TRACKING =================
    if tracking:
        success, box = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame_display, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            tracking = False  # kalau gagal, detect ulang

    cv2.imshow("MTCNN + Tracking", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
