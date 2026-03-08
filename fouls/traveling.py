# fouls/traveling.py
from collections import deque
import numpy as np

class TravelingDetector:
    def __init__(self):
        self.steps = 0
        self.prev_l_y = 0
        self.prev_r_y = 0
        self.is_holding_prev = False
        
    def check(self, landmarks_px, mp_pose, is_holding):
        # 📌 แก้ไข: ใช้ [1] เพื่อดึงค่า Y จาก Tuple (x, y)
        l_y = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE][1]
        r_y = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE][1]

        if is_holding and not self.is_holding_prev:
            self.steps = 0 # Gather Step (เริ่มนับ 0)
        
        if is_holding:
            # Logic: ถ้าระยะ Y เปลี่ยนแปลงเยอะ = ก้าว
            diff_l = abs(l_y - self.prev_l_y)
            diff_r = abs(r_y - self.prev_r_y)
            
            # Sensitivity (Pixel movement)
            threshold = 15.0 
            
            if diff_l > threshold or diff_r > threshold:
                self.steps += 0.1 # เพิ่มจำนวนก้าว (Sample rate มันถี่ ต้องบวกทีละน้อย)

        self.prev_l_y = l_y
        self.prev_r_y = r_y
        self.is_holding_prev = is_holding

        if int(self.steps) > 2:
            return True, f"TRAVELING ({int(self.steps)})"
        
        return False, f"Steps: {int(self.steps)}"