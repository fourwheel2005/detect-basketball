"""
fouls/contact_3dcnn_model.py — โหลดและรัน 3D CNN Model (High Accuracy Version)
--------------------------------------------------------------------------------
ฟีเจอร์ที่เพิ่มเพื่อความแม่นยำ:
    1. BGR to RGB Conversion: ป้องกันโมเดลสับสนสี
    2. Auto Normalization: แปลงค่าพิกเซล 0-255 เป็น 0-1 อัตโนมัติ
    3. Model Warm-up: สั่ง predict หลอก 1 ครั้งตอนเริ่ม เพื่อลดอาการกระตุกในเฟรมแรก
    4. Dynamic Thresholds: แยกความมั่นใจขั้นต่ำของแต่ละคลาสให้เหมาะกับความยากง่าย
"""

import tensorflow as tf
import numpy as np
import time

LABELS = ["normal", "reaching_foul", "blocking_charging", "holding"]

THRESHOLDS = {
    "reaching_foul": 0.60,      # ตีแขน (เห็นยาก ขยับเร็ว ให้ Threshold ต่ำหน่อย)
    "blocking_charging": 0.75,  # ชนลำตัว (เห็นชัด ให้ Threshold สูงเพื่อกัน False Positive)
    "holding": 0.70             # ดึงเสื้อ
}

class ContactFoulModel:
    """
    Wrapper สำหรับ 3D CNN Model
    รับ Clip (1, 16, 64, 64, 3) → ทำนาย Foul Type
    """

    def __init__(self, model_path: str = "trainmodel/contact_model.h5"):

        global LABELS
        print(f"⏳ Loading 3D CNN Model from: {model_path} ...")
        self.model = tf.keras.models.load_model(model_path)
        
        model_outputs = self.model.output_shape[-1]
        if model_outputs != len(LABELS):
            print(f"⚠️ คำเตือน: โมเดลมี {model_outputs} คลาส แต่ตั้ง LABELS ไว้ {len(LABELS)} คลาส!")
            LABELS = ["normal", "arm_hit", "body_contact"]
            
        self._warmup_model()
        print("✅ 3D CNN is Ready and Warmed up!")

    def _warmup_model(self):
        """รัน Inference หลอก 1 ครั้งตอนโหลด เพื่อไม่ให้ไปกระตุกตอนจับฟาวล์จริงในวินาทีแรก"""
        dummy_input = np.zeros((1, 16, 64, 64, 3), dtype=np.float32)
        _ = self.model.predict(dummy_input, verbose=0)

    def predict(self, clip_bgr: np.ndarray) -> tuple:
        """
        ทำนายประเภทฟาวล์จาก Clip วิดีโอ
        
        Parameters:
            clip_bgr: numpy array BGR วิดีโอ (1, 16, 64, 64, 3) 
                      ค่าพิกเซลสามารถเป็น 0-255 (uint8) หรือ 0-1 (float) ก็ได้
        """
        # ── 1. Data Preprocessing (สำคัญมากต่อ Accuracy) ──
        clip = np.copy(clip_bgr)
        
        # แปลง BGR เป็น RGB (OpenCV อ่านมาเป็น BGR แต่โมเดลส่วนใหญ่เทรนด้วย RGB)
        # ใช้ numpy slicing เพื่อสลับ channel สุดท้าย
        clip = clip[..., ::-1] 

        # ตรวจสอบและแปลงเป็น float32 (0.0 - 1.0)
        if clip.dtype != np.float32:
            clip = clip.astype(np.float32)
        if clip.max() > 1.0:
            clip /= 255.0  

        # ── 2. รัน Inference ──
        probs = self.model.predict(clip, verbose=0)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])
        label_name = LABELS[idx]

        # ── 3. ตรวจสอบ Threshold ──
        if label_name == "normal":
            return None, conf
            
        # ดึงค่า Threshold ของคลาสนั้นๆ (ถ้าไม่มีใน Dict ให้ใช้ 0.65 เป็นค่ามาตรฐาน)
        required_conf = THRESHOLDS.get(label_name, 0.65)

        if conf < required_conf:
            return None, conf

        return label_name, conf