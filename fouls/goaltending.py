import numpy as np
from collections import deque
import math

class GoaltendingDetector:
    def __init__(self):
        self.HISTORY_LEN = 15     # เก็บประวัติ 15 เฟรมเพื่อสร้างกราฟ
        self.MIN_SAMPLES = 8
        self.RIM_RATIO = 0.42
        self.HAND_BALL_THRESHOLD_PX = 80 # ระยะห่างระหว่างมือกับบอลที่ถือว่า "แตะ"
        
        self._y_history = deque(maxlen=self.HISTORY_LEN)

    def check(self, ball_center, hands_positions, frame_h):
        """
        hands_positions: list ของ (x, y) ของมือผู้เล่นทุกคนในเฟรม (ดึงจาก MediaPipe)
        """
        if ball_center is None:
            self._y_history.clear()
            return False, ""

        y = ball_center[1]
        self._y_history.append(y)

        if len(self._y_history) < self.MIN_SAMPLES:
            return False, ""

        # 1. เช็คความสูง (ต้องอยู่เหนือห่วง)
        if y >= frame_h * self.RIM_RATIO:
            return False, ""

        # 2. สร้างสมการ Parabola (y = ax^2 + bx + c) จากประวัติ Y
        # เนื่องจากพิกัด Y ของจอคอม ยิ่งลงล่างยิ่งค่าบวก กราฟลูกบาสตกพื้นจะเป็นตัว U หงาย (a > 0)
        x_axis = np.arange(len(self._y_history))
        y_axis = np.array(self._y_history)
        a, b, c = np.polyfit(x_axis, y_axis, 2)

        # ถ้า a > 0 แสดงว่าเป็นส่วนโค้งพาราโบลาหงาย (วิถีลูกบาส)
        # และจุดล่าสุด y_axis[-1] > y_axis[-2] แสดงว่ากำลังอยู่ใน "ขาลง"
        is_downward_arc = (a > 0.5) and (y_axis[-1] > y_axis[-2])

        if not is_downward_arc:
            return False, ""

        # 3. เช็คว่ามี "มือ" เข้าใกล้ลูกบาสในระยะที่กำหนดหรือไม่ (ป้องกันนกบินผ่าน)
        is_touched = False
        if hands_positions:
            for hand_x, hand_y in hands_positions:
                dist = math.hypot(hand_x - ball_center[0], hand_y - ball_center[1])
                if dist < self.HAND_BALL_THRESHOLD_PX:
                    is_touched = True
                    break

        if is_downward_arc and is_touched:
            # ล้างประวัติเพื่อไม่ให้เตือนซ้ำรัวๆ
            self._y_history.clear() 
            return True, "GOALTENDING ⚠️"

        return False, ""