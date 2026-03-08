import numpy as np
import time
import csv
import os

def get_dist(p1, p2):
    """ระยะห่างระหว่างจุด 2 จุด (Pixel)"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(a, b, c):
    """คำนวณมุม (องศา)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    return angle

# ==========================================
# 📝 คลาสสำหรับบันทึก Log การทำฟาวล์
# ==========================================
class FoulLogger:
    def __init__(self, filename="basketball_foul_logs.csv", cooldown_sec=3.0):
        self.filename = filename
        self.cooldown_sec = cooldown_sec
        self.last_logged = {} # เก็บเวลาล่าสุดที่ log ฟาวล์นั้นๆ เพื่อกันการบันทึกซ้ำรัวๆ
        
        # สร้างไฟล์และหัวข้อคอลัมน์ถ้ายังไม่มีไฟล์นี้
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Date_Time", "Player_ID", "Foul_Type"])

    def log_foul(self, player_id, foul_type):
        current_time = time.time()
        # สร้าง Key เป็น ID_ฟาวล์ เช่น "1_TRAVELING"
        key = f"{player_id}_{foul_type}"

        # เช็คว่าเลยช่วง Cooldown หรือยัง (กันการ log ซ้ำรัวๆ ในเฟรมติดกัน)
        if key not in self.last_logged or (current_time - self.last_logged[key] > self.cooldown_sec):
            self.last_logged[key] = current_time
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            
            # บันทึกลงไฟล์ CSV
            with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp_str, f"Player_{player_id}", foul_type])
            
            # แจ้งเตือนใน Terminal ด้วยว่ามีการบันทึกลงไฟล์แล้ว
            print(f"📝 [LOG SAVED] {timestamp_str} | Player: {player_id} | Foul: {foul_type}")