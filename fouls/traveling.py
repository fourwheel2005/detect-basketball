# ไฟล์ fouls/traveling.py
from utils import EMAFilter, TemporalVoter

class TravelingDetector:
    def __init__(self):
        self.steps = 0
        self.ema_l = EMAFilter(alpha=0.3) # 📌 ลด Jitter ข้อเท้าซ้าย
        self.ema_r = EMAFilter(alpha=0.3) # 📌 ลด Jitter ข้อเท้าขวา
        self.voter = TemporalVoter(window_size=5, threshold=4) # 📌 ระบบ Voting
        
        self.prev_l_y = None
        self.prev_r_y = None
        self.is_holding_prev = False
        
    def check(self, landmarks_px, mp_pose, is_holding, shoulder_width):
        # ดึงค่า Y และผ่าน Filter ลดการสั่น
        raw_l_y = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE][1]
        raw_r_y = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE][1]
        l_y = self.ema_l.update(raw_l_y)
        r_y = self.ema_r.update(raw_r_y)

        if self.prev_l_y is None:
            self.prev_l_y = l_y
            self.prev_r_y = r_y

        if is_holding and not self.is_holding_prev:
            self.steps = 0 
        
        if is_holding:
            diff_l = abs(l_y - self.prev_l_y)
            diff_r = abs(r_y - self.prev_r_y)
            
            # 📌 Dynamic Threshold: ระยะก้าวต้องเกิน 15% ของความกว้างไหล่
            dynamic_threshold = shoulder_width * 0.15 
            
            if diff_l > dynamic_threshold or diff_r > dynamic_threshold:
                self.steps += 0.1 

        self.prev_l_y = l_y
        self.prev_r_y = r_y
        self.is_holding_prev = is_holding

        # 📌 ระบบ Voting: เช็คว่าก้าวเกิน 2 ก้าวไหม ถ้าเกินส่งไปโหวต
        is_foul_raw = int(self.steps) > 2
        is_foul_confirmed = self.voter.vote(is_foul_raw)

        if is_foul_confirmed:
            return True, f"TRAVELING ({int(self.steps)})"
        
        return False, f"Steps: {int(self.steps)}"