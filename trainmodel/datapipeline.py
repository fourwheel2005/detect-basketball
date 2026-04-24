import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical

class VideoDataGenerator(Sequence):
    def __init__(self, dataset_path, batch_size=8, frames_per_clip=16, 
                 target_size=(112, 112), classes=None, shuffle=True):
        """
        Custom Data Generator สำหรับโหลดภาพ 16 เฟรมเข้า 3D CNN
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.frames_per_clip = frames_per_clip
        self.target_size = target_size
        self.classes = classes if classes else ['normal', 'arm_hit', 'body_contact']
        self.shuffle = shuffle
        
        # 1. รวบรวม Path ของทุกคลิปและ Label
        self.clip_paths = []
        self.labels = []
        
        for label_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_dir):
                continue
                
            # วนลูปหาโฟลเดอร์ย่อย (1 โฟลเดอร์ = 1 คลิป = 16 ภาพ)
            for clip_folder in os.listdir(class_dir):
                clip_path = os.path.join(class_dir, clip_folder)
                if os.path.isdir(clip_path):
                    self.clip_paths.append(clip_path)
                    self.labels.append(label_idx)
                    
        # สร้าง Index สำหรับสุ่ม (Shuffle) ข้อมูล
        self.indexes = np.arange(len(self.clip_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # คำนวณจำนวน Batch ต่อ 1 Epoch
        return int(np.floor(len(self.clip_paths) / self.batch_size))

    def on_epoch_end(self):
        # สลับข้อมูลใหม่ทุกครั้งที่จบ Epoch ลด Overfitting
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # ดึง Index สำหรับ Batch ปัจจุบัน
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # เตรียม Array ว่างสำหรับใส่ข้อมูล X (ภาพ) และ Y (Label)
        # X shape: (batch_size, 16, 112, 112, 3)
        X = np.empty((self.batch_size, self.frames_per_clip, *self.target_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        
        for i, idx in enumerate(batch_indexes):
            clip_path = self.clip_paths[idx]
            y[i] = self.labels[idx]
            
            # ดึงรายชื่อภาพ และ "เรียงลำดับตามชื่อ" (สำคัญมาก ไม่งั้นวิดีโอจะเล่นสลับไปมา)
            frame_files = sorted(os.listdir(clip_path))
            
            # ป้องกันกรณีคลิปมีภาพไม่ถึง 16 เฟรม
            frame_files = frame_files[:self.frames_per_clip]
            
            for j, frame_file in enumerate(frame_files):
                img_path = os.path.join(clip_path, frame_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # แปลง BGR (OpenCV) เป็น RGB (Standard AI)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize เป็น 112x112
                    img = cv2.resize(img, self.target_size)
                    # Normalization (0-1) แล้วยัดใส่ Array
                    X[i, j, :, :, :] = img / 255.0
                    
        # แปลง Label Y ให้เป็น One-Hot Encoding (เช่น [0, 1, 0])
        return X, to_categorical(y, num_classes=len(self.classes))

# ==========================================
# วิธีเรียกใช้งาน
# ==========================================
if __name__ == '__main__':
    train_gen = VideoDataGenerator(dataset_path='dataset/train/', batch_size=4)
    
    # ทดสอบดึง Data ออกมา 1 Batch เพื่อดู Shape
    X_batch, y_batch = train_gen[0]
    print("X_batch shape (Batch, Frames, H, W, Channels):", X_batch.shape) 
    print("y_batch shape (Batch, Classes):", y_batch.shape)