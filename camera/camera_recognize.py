import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "face_db.npy"
THRESHOLD = 0.45

face_database = np.load(DB_PATH, allow_pickle=True).item()

known_embeddings = []
known_students = []

for student_no, info in face_database.items():
    known_embeddings.append(info["embedding"])
    known_students.append({
        "student_no": info["student_no"],
        "name": info["name"]
    })

known_embeddings = np.array(known_embeddings)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用 NVIDIA GPU

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("摄像头已启动，按 Esc 退出")

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头画面")
        break

    height, width = frame.shape[:2]

    frame_center_x = width // 2
    frame_center_y = height // 2

    faces = app.get(frame)

    for face in faces:
        embedding = face.normed_embedding

        similarities = np.dot(known_embeddings, embedding)

        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score >= THRESHOLD:
            student = known_students[best_index]
            label = f"{student['student_no']} {student['name']} {best_score:.2f}"
            color = (0, 255, 0)

            x1, y1, x2, y2 = face.bbox.astype(int)

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # ===== 关键：误差坐标 =====
            error_x = center_x - frame_center_x
            error_y = center_y - frame_center_y

            print(f"误差坐标: ({error_x}, {error_y})")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

    cv2.imshow("School Face Access GPU", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()