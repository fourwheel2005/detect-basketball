"""
สอน 3D CNN สำหรับตรวจจับฟาวล์ประเภทสัมผัส

โครงสร้าง Dataset ที่ต้องเตรียม:
    trainmodel/dataset/
        normal/        ← คลิปปกติ ไม่มีฟาวล์
        arm_hit/       ← คลิปตีแขน
        body_contact/  ← คลิปชนตัว
    แต่ละโฟลเดอร์ใส่ไฟล์ .mp4 หรือ .avi
    ความยาวคลิปละ ~1 วินาที (16–30 เฟรม)
"""
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

CLIP_LEN   = 16
FRAME_SIZE = (64, 64)
DATASET_DIR = "trainmodel/dataset"
LABELS      = ["normal", "arm_hit", "body_contact"]
MODEL_OUT   = "trainmodel/contact_model.h5"


def load_clip(video_path: str) -> np.ndarray:
    """โหลด Video แล้วตัดเป็น Clip ขนาด (CLIP_LEN, H, W, 3)"""
    cap    = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < CLIP_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, FRAME_SIZE).astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()

    while len(frames) < CLIP_LEN:
        frames.append(frames[-1])

    return np.stack(frames[:CLIP_LEN])


def build_dataset():
    X, y = [], []
    for label_idx, label in enumerate(LABELS):
        folder = os.path.join(DATASET_DIR, label)
        for fname in os.listdir(folder):
            if not fname.endswith(('.mp4', '.avi')):
                continue
            clip = load_clip(os.path.join(folder, fname))
            X.append(clip)
            y.append(label_idx)
    return np.array(X), tf.keras.utils.to_categorical(y, len(LABELS))


def build_model():
    """3D CNN อย่างง่าย — ปรับ layer เพิ่มได้ถ้าข้อมูลเยอะ"""
    inp = tf.keras.Input(shape=(CLIP_LEN, *FRAME_SIZE, 3))
    x   = tf.keras.layers.Conv3D(32, (3,3,3), activation='relu', padding='same')(inp)
    x   = tf.keras.layers.MaxPool3D((2,2,2))(x)
    x   = tf.keras.layers.Conv3D(64, (3,3,3), activation='relu', padding='same')(x)
    x   = tf.keras.layers.MaxPool3D((2,2,2))(x)
    x   = tf.keras.layers.GlobalAvgPool3D()(x)
    x   = tf.keras.layers.Dense(128, activation='relu')(x)
    x   = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(len(LABELS), activation='softmax')(x)
    return tf.keras.Model(inp, out)


if __name__ == "__main__":
    print("📂 Loading dataset...")
    X, y = build_dataset()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=30, batch_size=8)
    model.save(MODEL_OUT)
    print(f"✅ Model saved → {MODEL_OUT}")