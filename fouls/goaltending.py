# fouls/goaltending.py
from collections import deque

class GoaltendingDetector:
    def __init__(self):
        self.ball_history = deque(maxlen=15)

    def check(self, ball_center, frame_h):
        if ball_center is None: return False, ""
        
        y = ball_center[1]
        self.ball_history.append(y)
        
        if len(self.ball_history) < 5: return False, ""

        # 1. เช็คว่าบอลกำลัง "ตกลง" (Downward)
        # ค่า Y ล่าสุดต้องมากกว่าค่า Y อดีต (เพราะ Y เริ่ม 0 จากด้านบน)
        curr = self.ball_history[-1]
        prev = self.ball_history[-5]
        
        is_falling = curr > (prev + 10) # ต้องตกลงมาชัดเจน

        # 2. เช็คความสูง (Height)
        # บอลต้องอยู่สูงกว่าครึ่งจอ หรือระดับห่วง (ต้อง Calibrate หน้างาน)
        is_above_rim_level = y < (frame_h * 0.4) 

        if is_falling and is_above_rim_level:
            return True, "GOALTENDING WARNING"
            
        return False, ""