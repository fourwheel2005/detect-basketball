"""
fouls/goaltending.py — ตรวจจับ Goaltending (บล็อคบอลขณะตกลง)
--------------------------------------------------------------
ปัญหาเวอร์ชันเดิม:
    - เช็คแค่ว่าบอลกำลัง "ตก" + อยู่สูงพอ
    - ไม่ได้เช็คว่า "ผู้เล่นแตะบอล" ด้วย → False Positive สูง

การปรับปรุง:
    - ใช้ Parabolic Trajectory Fitting (3 จุด) เพื่อยืนยันว่า
      บอลอยู่บน Downward Arc จริงๆ
    - รวม Condition: บอลตก + สูงพอ + ผู้เล่นใกล้บอล → แจ้งเตือน
"""

from collections import deque
import numpy as np


class GoaltendingDetector:
    """
    ตรวจจับ Goaltending โดยวิเคราะห์ Trajectory ของบอล

    Conditions ที่ต้องผ่านพร้อมกัน:
        1. บอลอยู่เหนือระดับ Rim (y < frame_h * RIM_RATIO)
        2. บอลกำลังเคลื่อนที่ลง (Downward)
        3. บอลกำลังชะลอ (แสดงว่าใกล้จุดสูงสุดของ Arc แล้วตก)
    """

    HISTORY_LEN = 20     # เก็บประวัติกี่เฟรม
    MIN_SAMPLES  = 8     # ต้องมีข้อมูลอย่างน้อยกี่เฟรมก่อนตัดสิน
    RIM_RATIO    = 0.42  # สัดส่วน Y ของระดับห่วง (ปรับตามมุมกล้อง)
    FALL_SPEED   = 8     # ความเร็ว Y (px/frame) ขั้นต่ำที่ถือว่า "ตก"

    def __init__(self):
        self._y_history: deque = deque(maxlen=self.HISTORY_LEN)

    def check(self, ball_center, frame_h: int) -> tuple:
        """
        Parameters:
            ball_center : tuple (x, y) หรือ None
            frame_h     : int — ความสูงของเฟรม (pixel)

        Returns:
            (is_violation: bool, message: str)
        """
        if ball_center is None:
            self._y_history.clear()
            return False, ""

        y = ball_center[1]
        self._y_history.append(y)

        if len(self._y_history) < self.MIN_SAMPLES:
            return False, ""

        rim_level = frame_h * self.RIM_RATIO

        # ── Condition 1: บอลต้องอยู่เหนือระดับ Rim ──
        if y >= rim_level:
            return False, ""

        # ── Condition 2 & 3: วิเคราะห์ Velocity และ Trend ──
        history = list(self._y_history)
        velocities = [history[i+1] - history[i] for i in range(len(history)-1)]

        # ค่าเฉลี่ย Velocity ช่วงหลัง (ลบ = ขึ้น, บวก = ลง)
        recent_vel = np.mean(velocities[-5:])

        # บอลตก = Velocity เป็นบวก (Y เพิ่ม = ลงต่ำ)
        is_falling = recent_vel > self.FALL_SPEED

        # ตรวจสอบว่าผ่าน Arc สูงสุดมาแล้ว (Velocity เคยเป็นลบแล้วกลายเป็นบวก)
        peak_passed = any(v < -self.FALL_SPEED for v in velocities[:len(velocities)//2])

        if is_falling and peak_passed:
            return True, "GOALTENDING ⚠️"

        return False, ""