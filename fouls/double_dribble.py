# fouls/double_dribble.py
import numpy as np
from utils import get_dist

class DoubleDribbleDetector:
    def __init__(self):
        # State: IDLE -> DRIBBLING -> HOLDING -> VIOLATION
        self.state = "IDLE"

    def check(self, landmarks, mp_pose, ball_center):
        if ball_center is None: return False, ""

        # ดึงพิกัดมือ (Global Coordinates ที่ส่งมาจาก main)
        r_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, 
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y])
        l_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, 
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y])

        # ระยะห่าง (Threshold ต้องจูนตามความละเอียดกล้อง)
        # สมมติว่าส่งพิกัดเป็น Pixel มาแล้ว
        dist_r = get_dist(r_wrist, ball_center)
        dist_l = get_dist(l_wrist, ball_center)
        threshold = 100 

        is_touching_two = (dist_r < threshold) and (dist_l < threshold)
        is_touching_one = (dist_r < threshold) or (dist_l < threshold)

        violation = False

        # State Machine Logic
        if is_touching_two:
            self.state = "HOLDING" # จับสองมือ = หยุดเลี้ยง
        
        elif is_touching_one:
            if self.state == "HOLDING":
                # เคยหยุดเลี้ยงแล้ว กลับมาเลี้ยงใหม่ = ฟาวล์
                violation = True
            elif self.state == "IDLE":
                self.state = "DRIBBLING"

        # Auto Reset ถ้าบอลหายไปนาน (Optional)
        
        if violation:
            return True, "DOUBLE DRIBBLE"
        return False, ""