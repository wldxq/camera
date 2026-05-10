import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

PHOTO_DIR = "photos"
DB_PATH = "face_db.npy"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 使用 NVIDIA GPU

face_database = {}

for filename in os.listdir(PHOTO_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(PHOTO_DIR, filename)

    name_part = os.path.splitext(filename)[0]
    try:
        student_no, student_name = name_part.split("_", 1)
    except ValueError:
        print(f"文件名格式不正确，跳过：{filename}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片：{filename}")
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"未检测到人脸：{filename}")
        continue

    if len(faces) > 1:
        print(f"检测到多张人脸，跳过：{filename}")
        continue

    face = faces[0]
    embedding = face.normed_embedding

    face_database[student_no] = {
        "student_no": student_no,
        "name": student_name,
        "embedding": embedding
    }

    print(f"注册成功：{student_no} {student_name}")

np.save(DB_PATH, face_database)
print(f"\n人脸特征库已保存到：{DB_PATH}")