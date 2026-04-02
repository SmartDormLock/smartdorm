import cv2
from mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

frame_count = 0

print("MTCNN Camera ON ??")

while True:
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 2 != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if rgb.shape[0] < 40 or rgb.shape[1] < 40:
        continue

    try:
        faces = detector.detect_faces(rgb)
    except Exception as e:
        print("Skip frame:", e)
        faces = []

    frame_display = frame.copy()
    h_frame, w_frame, _ = frame.shape

    for face in faces:
        x, y, w, h = face['box']

        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        cv2.rectangle(frame_display, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("MTCNN Face Detection", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
