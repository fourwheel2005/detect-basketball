"""
fouls/traveling.py — ตรวจจับ Traveling (ก้าวเกิน 2 ก้าว & Up-and-Down)
-------------------------------------------------------------------------
การปรับปรุงล่าสุด:
  - (เดิม) นับก้าวจริงด้วย Peak Detection + Smoothing 
  - (ใหม่) เพิ่มระบบตรวจจับ "Up-and-Down" (กระโดดแล้วไม่ปล่อยบอลก่อนเท้าแตะพื้น)
"""

from collections import deque

class TravelingDetector:
    """
    ตรวจจับ Traveling 2 รูปแบบหลัก:
        1. Too Many Steps : ก้าวเกิน MAX_STEPS ขณะถือบอล
        2. Up-and-Down    : กระโดดขณะถือบอล แล้วลงพื้นโดยยังไม่ปล่อยบอล
    """

    # ── Config การนับก้าว ──
    MAX_STEPS   = 2      # จำนวนก้าวสูงสุดที่อนุญาต
    SMOOTH_WIN  = 5      # ขนาดหน้าต่าง Smoothing (เฟรม)
    STEP_THRESH = 12     # ระยะ Y ขั้นต่ำ (px) ที่ถือว่าเป็น "ก้าว"

    # ── Config การกระโดด (Up-and-Down) ──
    JUMP_THRESH = 40     # ระยะพิกัด Y ที่ลอยขึ้นไปแล้วถือว่า "กระโดดพ้นพื้น"

    def __init__(self):
        # ── State สำหรับนับก้าว ──
        self._l_hist: deque = deque(maxlen=self.SMOOTH_WIN)
        self._r_hist: deque = deque(maxlen=self.SMOOTH_WIN)
        self.steps = 0
        self._prev_l_smooth = None
        self._prev_r_smooth = None
        self._l_direction   = 0   # +1 = กำลังยก, -1 = กำลังวาง
        self._r_direction   = 0

        # ── State สำหรับ Up-and-Down ──
        self._was_holding      = False
        self._jumped_with_ball = False
        self._ground_l_y       = 0.0  # เก็บ Baseline พื้นของเท้าซ้าย
        self._ground_r_y       = 0.0  # เก็บ Baseline พื้นของเท้าขวา

    # ─── Public API ───────────────────────────────────

    def check(self, landmarks_px: dict, mp_pose, is_holding: bool) -> tuple:
        """
        ตรวจสอบ Traveling ใน 1 เฟรม
        Returns: (is_violation: bool, message: str)
        """
        # ดึงพิกัด Y ของข้อเท้า (Y ยิ่งมาก = ยิ่งอยู่ต่ำ/ติดพื้น)
        l_y = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE][1]
        r_y = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE][1]

        # ── 1. จัดการจังหวะเริ่มถือบอล (Gather Step) ──
        if is_holding and not self._was_holding:
            self._reset_states(l_y, r_y)

        # ถ้าปล่อยบอลไปแล้ว (Pass/Shoot) รีเซ็ตสถานะการกระโดด
        if not is_holding:
            self._jumped_with_ball = False
            self._was_holding = False
            return False, f"Steps: {self.steps}"

        self._was_holding = is_holding

        # ── 2. อัปเดต Baseline พื้น (Ground Level) ──
        # คอยอัปเดตค่า Y ที่ต่ำที่สุด (ค่ามากสุดในแกน Y) ไว้เป็น "พื้น" เสมอ
        self._ground_l_y = max(self._ground_l_y, l_y)
        self._ground_r_y = max(self._ground_r_y, r_y)

        # ── 3. ตรวจสอบ "Up-and-Down" (กระโดดแล้วไม่ปล่อยบอล) ──
        # ตรวจว่าเท้าทั้งสองลอยพ้นพื้น (ค่า Y ลดลงเกิน Threshold)
        is_airborne = (self._ground_l_y - l_y > self.JUMP_THRESH) and \
                      (self._ground_r_y - r_y > self.JUMP_THRESH)

        if is_airborne:
            self._jumped_with_ball = True

        # ตรวจการลงพื้น: เท้าข้างใดข้างหนึ่งกลับมาใกล้ระดับพื้น
        is_landed = (self._ground_l_y - l_y < self.JUMP_THRESH * 0.3) or \
                    (self._ground_r_y - r_y < self.JUMP_THRESH * 0.3)

        if self._jumped_with_ball and is_landed:
            return True, "TRAVELING (Up & Down)"

        # ── 4. ตรวจสอบ "จำนวนก้าว" (Too Many Steps) ──
        self._l_hist.append(l_y)
        self._r_hist.append(r_y)

        if len(self._l_hist) == self.SMOOTH_WIN:
            l_smooth = sum(self._l_hist) / self.SMOOTH_WIN
            r_smooth = sum(self._r_hist) / self.SMOOTH_WIN

            if self._prev_l_smooth is not None:
                self.steps += self._count_step(self._prev_l_smooth, l_smooth, "_l_direction")
                self.steps += self._count_step(self._prev_r_smooth, r_smooth, "_r_direction")

            self._prev_l_smooth = l_smooth
            self._prev_r_smooth = r_smooth

            if self.steps > self.MAX_STEPS:
                return True, f"TRAVELING ({self.steps} steps)"

        return False, f"Steps: {self.steps}"

    # ─── Private Helpers ──────────────────────────────

    def _count_step(self, prev_y: float, curr_y: float, direction_attr: str) -> int:
        """นับ 1 ก้าวเมื่อ Y เปลี่ยนทิศจาก "ลง" → "ขึ้น" """
        diff = curr_y - prev_y
        current_dir = getattr(self, direction_attr)

        if abs(diff) < self.STEP_THRESH:
            return 0  

        new_dir = 1 if diff > 0 else -1

        if current_dir == 1 and new_dir == -1:
            setattr(self, direction_attr, new_dir)
            return 1

        setattr(self, direction_attr, new_dir)
        return 0

    def _reset_states(self, l_y: float, r_y: float):
        """รีเซ็ตตัวนับก้าว, สถานะการกระโดด และระดับพื้น เมื่อเริ่มถือบอลใหม่"""
        self.steps = 0
        self._l_direction = 0
        self._r_direction = 0
        self._prev_l_smooth = None
        self._prev_r_smooth = None
        self._l_hist.clear()
        self._r_hist.clear()
        
        self._jumped_with_ball = False
        self._ground_l_y = l_y
        self._ground_r_y = r_y