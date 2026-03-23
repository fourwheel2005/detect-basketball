"""
fouls/double_dribble.py — ตรวจจับ Double Dribble
--------------------------------------------------
ปัญหาของเวอร์ชันเดิม:
  1. ใช้ landmarks.x / landmarks.y แบบ Normalized (0–1)
     แต่ ball_center เป็น Pixel → ระยะห่างผิดพลาด 100%
  2. State Machine ไม่ Reset เมื่อผู้เล่นชู้ต/ส่งบอล

การแก้ไข:
  - รับ landmarks_px (Pixel) แบบเดียวกับไฟล์อื่นๆ
  - เพิ่ม State "RELEASED" เพื่อครอบคลุม Shot/Pass
  - เพิ่ม Timeout: ถ้าไม่แตะบอลนานเกิน N เฟรม → Reset
"""

import time
from utils import get_dist


class DoubleDribbleDetector:
    """
    ตรวจจับ Double Dribble ด้วย State Machine

    States:
        IDLE      → ยังไม่แตะบอล (เริ่มต้น / หลัง Reset)
        DRIBBLING → เลี้ยงบอลด้วยมือเดียว
        HOLDING   → จับบอลด้วยสองมือ (หยุดเลี้ยงแล้ว)
        VIOLATION → ตรวจพบฟาวล์ (รอ Reset)

    Transitions:
        IDLE      + touch_one  → DRIBBLING
        DRIBBLING + touch_two  → HOLDING
        HOLDING   + touch_one  → VIOLATION  ← Double Dribble!
        any       + no touch (timeout) → IDLE
    """

    HOLD_THRESHOLD   = 110   # ระยะ (px) ที่ถือว่า "จับบอล" ด้วยมือนั้น
    DRIBBLE_THRESHOLD = 140  # ระยะ (px) ที่ถือว่า "แตะบอล" ขณะเลี้ยง
    TIMEOUT_FRAMES   = 45    # เฟรมที่ไม่แตะบอลแล้ว Reset State

    def __init__(self):
        self.state = "IDLE"
        self._no_touch_frames = 0  # นับเฟรมที่ไม่แตะบอล

    def check(self, landmarks_px: dict, mp_pose, ball_center) -> tuple:
        """
        ตรวจสอบ Double Dribble ใน 1 เฟรม

        Parameters:
            landmarks_px : dict {landmark_id: (x, y)} — Pixel coordinates
            mp_pose      : mediapipe.solutions.pose module
            ball_center  : tuple (x, y) หรือ None

        Returns:
            (is_violation: bool, message: str)
        """
        if ball_center is None:
            return False, ""

        # ดึงพิกัด Pixel ของข้อมือ (ไม่ต้องแปลงอีก เพราะ landmarks_px เป็น Pixel แล้ว)
        r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST]

        dist_r = get_dist(r_wrist, ball_center)
        dist_l = get_dist(l_wrist, ball_center)

        # ─── จำแนกสถานะการสัมผัสบอล ───
        touching_right = dist_r < self.DRIBBLE_THRESHOLD
        touching_left  = dist_l < self.DRIBBLE_THRESHOLD
        touching_both  = touching_right and touching_left
        touching_any   = touching_right or touching_left

        # ─── Timeout: ถ้าไม่แตะบอลนานเกิน Threshold → Reset ───
        if not touching_any:
            self._no_touch_frames += 1
            if self._no_touch_frames >= self.TIMEOUT_FRAMES:
                self.state = "IDLE"
                self._no_touch_frames = 0
            return False, f"State: {self.state}"
        else:
            self._no_touch_frames = 0

        # ─── State Machine ───
        violation = False

        if touching_both:
            # จับสองมือ = หยุดเลี้ยงแล้ว
            self.state = "HOLDING"

        elif touching_any:
            if self.state == "HOLDING":
                # เคย HOLDING แล้วกลับมาแตะมือเดียว = Double Dribble
                violation = True
                self.state = "VIOLATION"
            elif self.state == "IDLE":
                self.state = "DRIBBLING"
            # DRIBBLING + touch_one = ยังเลี้ยงอยู่ ปกติ

        if violation:
            return True, "DOUBLE DRIBBLE"

        return False, f"State: {self.state}"

    def reset(self):
        """รีเซ็ตสถานะ (เรียกเมื่อ Shot หรือ Pass ออกไป)"""
        self.state = "IDLE"
        self._no_touch_frames = 0