import cv2
import numpy as np
from insightface.app import FaceAnalysis

DB_PATH = "face_db.npy"
THRESHOLD = 0.45

class KalmanFilter2D:
    def __init__(self):
        # 状态量：[x, y, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float32)

        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # 观测矩阵：只能观测到 x, y
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # 过程噪声，越大越跟手，但越抖
        self.Q = np.eye(4, dtype=np.float32) * 0.03

        # 测量噪声，越大越平滑，但越滞后
        self.R = np.eye(2, dtype=np.float32) * 8.0

        # 估计误差协方差
        self.P = np.eye(4, dtype=np.float32) * 100

        self.initialized = False

    def update(self, measured_x, measured_y):
        z = np.array([[measured_x], [measured_y]], dtype=np.float32)

        if not self.initialized:
            self.x[0, 0] = measured_x
            self.x[1, 0] = measured_y
            self.x[2, 0] = 0
            self.x[3, 0] = 0
            self.initialized = True
            return measured_x, measured_y

        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 更新
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ P_pred

        filtered_x = int(self.x[0, 0])
        filtered_y = int(self.x[1, 0])

        return filtered_x, filtered_y

    def reset(self):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 100
        self.initialized = False

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

kalman_filter = KalmanFilter2D()

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

            filtered_error_x, filtered_error_y = kalman_filter.update(
                error_x,
                error_y
            )

            print(f"误差坐标: ({filtered_error_x}, {filtered_error_y})")

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
