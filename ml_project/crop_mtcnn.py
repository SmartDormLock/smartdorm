import cv2
from mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)

cap.set(3, 480)
cap.set(4, 360)

print("STEP 1: CROP WAJAH ??")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        faces = detector.detect_faces(rgb)
    except:
        faces = []

    for face in faces:
        x, y, w, h = face['box']

        # clamp biar aman
        x = max(0, x)
        y = max(0, y)

        face_crop = frame[y:y+h, x:x+w]

        if face_crop.size > 0:
            cv2.imshow("Face Crop", face_crop)

        cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Camera", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
