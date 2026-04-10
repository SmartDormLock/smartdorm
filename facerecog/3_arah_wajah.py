import cv2
from facenet_pytorch import MTCNN

# =====================
# INIT
mtcnn = MTCNN(keep_all=True)

cap = cv2.VideoCapture(0)

# resolusi (balance)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Kamera gagal dibuka ??")
    exit()

print("MTCNN Direction (FIXED & STABLE) ??")

# window resize
cv2.namedWindow("MTCNN Direction", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MTCNN Direction", 900, 700)

# =====================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal ambil frame")
        break

    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

    if boxes is not None:
        for box, prob, lm in zip(boxes, probs, landmarks):
            x1, y1, x2, y2 = map(int, box)

            # kotak wajah
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # =====================
            # LANDMARK
            left_eye = lm[0]
            right_eye = lm[1]
            nose = lm[2]

            # =====================
            # HITUNG POSISI
            eye_center_x = (left_eye[0] + right_eye[0]) / 2
            eye_center_y = (left_eye[1] + right_eye[1]) / 2

            diff_x = nose[0] - eye_center_x
            diff_y = nose[1] - eye_center_y

            # =====================
            # ?? NORMALISASI (biar stabil)
            face_width = x2 - x1
            face_height = y2 - y1

            if face_width > 0 and face_height > 0:
                diff_x /= face_width
                diff_y /= face_height

            # =====================
            # THRESHOLD
            th_x = 0.05
            th_y = 0.08

            # =====================
            # ?? LOGIC FINAL (PRIORITAS HORIZONTAL)
            if abs(diff_x) > th_x:
                if diff_x > 0:
                    direction = "Kiri"
                else:
                    direction = "Kanan"

            elif abs(diff_y) > th_y:
                if diff_y > 0:
                    direction = "Bawah"
                else:
                    direction = "Atas"

            else:
                direction = "Depan"

            # =====================
            # TEKS (PUTIH + OUTLINE)
            cv2.putText(frame, direction, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)

            cv2.putText(frame, direction, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # confidence
            cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # =====================
    # SHOW
    cv2.imshow("MTCNN Direction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================
# CLEANUP
cap.release()
cv2.destroyAllWindows()
