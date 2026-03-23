"""
fouls/traveling.py — ตรวจจับ Traveling (ก้าวเท้าเกิน 2 ก้าว ขณะถือบอล)
-------------------------------------------------------------------------
ปัญหาของเวอร์ชันเดิม:
  1. บวก 0.1 ต่อเฟรม → ขึ้นอยู่กับ FPS ทำให้ผลต่างกันในแต่ละเครื่อง
  2. ไม่มี Debounce → นับซ้ำถ้าเท้าสั่น (Jitter) เล็กน้อย
  3. ไม่ Reset เมื่อหยุดถือบอล

การแก้ไข:
  - ใช้ Peak Detection (จาก Low → High) แทนการนับต่อเฟรม
    → นับ "ก้าวจริง" โดยดูว่าเท้าข้างไหน Y ขึ้นสูงสุดแล้วลงมา
  - เพิ่ม Smoothing ด้วย Rolling Average เพื่อกรอง Noise
  - Reset steps เมื่อปล่อยบอล
"""

from collections import deque


class TravelingDetector:
    """
    ตรวจจับ Traveling โดยนับ "จำนวนก้าวจริง" ด้วย Peak Detection

    Logic:
        1. เก็บประวัติค่า Y ของข้อเท้าซ้าย/ขวา (Smoothed)
        2. ก้าว = เมื่อ Y ของเท้าข้างใดข้างหนึ่งเปลี่ยนทิศ (Peak/Valley)
        3. ถ้าก้าวเกิน MAX_STEPS ขณะถือบอล → TRAVELING
    """

    MAX_STEPS   = 2      # จำนวนก้าวสูงสุดที่อนุญาต
    SMOOTH_WIN  = 5      # ขนาดหน้าต่าง Smoothing (เฟรม)
    STEP_THRESH = 12     # ระยะ Y ขั้นต่ำ (px) ที่ถือว่าเป็น "ก้าว"

    def __init__(self):
        # ประวัติ Y สำหรับ Smoothing
        self._l_hist: deque = deque(maxlen=self.SMOOTH_WIN)
        self._r_hist: deque = deque(maxlen=self.SMOOTH_WIN)

        self.steps = 0
        self._prev_l_smooth = None
        self._prev_r_smooth = None
        self._l_direction   = 0   # +1 = กำลังยก, -1 = กำลังวาง
        self._r_direction   = 0

        self._was_holding = False  # สถานะการถือบอลเฟรมก่อน

    # ─── Public API ───────────────────────────────────

    def check(self, landmarks_px: dict, mp_pose, is_holding: bool):
        """
        ตรวจสอบ Traveling ใน 1 เฟรม

        Parameters:
            landmarks_px : dict {landmark_id: (x, y)}
            mp_pose      : mediapipe.solutions.pose module
            is_holding   : bool — ผู้เล่นถือบอลอยู่หรือไม่

        Returns:
            (is_violation: bool, message: str)
        """
        # ดึงพิกัด Y ของข้อเท้า
        l_y = landmarks_px[mp_pose.PoseLandmark.LEFT_ANKLE][1]
        r_y = landmarks_px[mp_pose.PoseLandmark.RIGHT_ANKLE][1]

        # ── รีเซ็ตเมื่อเพิ่งเริ่มถือบอล (Gather Step = 0) ──
        if is_holding and not self._was_holding:
            self._reset_steps()

        self._was_holding = is_holding

        if not is_holding:
            return False, f"Steps: {self.steps}"

        # ── Smoothing: เฉลี่ย Y ใน Rolling Window ──
        self._l_hist.append(l_y)
        self._r_hist.append(r_y)

        if len(self._l_hist) < self.SMOOTH_WIN:
            return False, f"Steps: {self.steps}"  # รอข้อมูลพอสำหรับ Smooth

        l_smooth = sum(self._l_hist) / len(self._l_hist)
        r_smooth = sum(self._r_hist) / len(self._r_hist)

        # ── Peak Detection: นับก้าวเมื่อทิศทาง Y กลับทาง ──
        if self._prev_l_smooth is not None:
            self.steps += self._count_step(self._prev_l_smooth, l_smooth,
                                            "_l_direction")
            self.steps += self._count_step(self._prev_r_smooth, r_smooth,
                                            "_r_direction")

        self._prev_l_smooth = l_smooth
        self._prev_r_smooth = r_smooth

        # ── ตัดสิน ──
        if self.steps > self.MAX_STEPS:
            return True, f"TRAVELING ({self.steps} steps)"

        return False, f"Steps: {self.steps}"

    # ─── Private Helpers ──────────────────────────────

    def _count_step(self, prev_y: float, curr_y: float,
                    direction_attr: str) -> int:
        """
        นับ 1 ก้าวเมื่อ Y เปลี่ยนทิศจาก "ลง" → "ขึ้น"
        (เท้าวางลงดิน แล้วยกขึ้นมา = 1 ก้าว)
        """
        diff = curr_y - prev_y  # บวก = ลงต่ำ, ลบ = ยกขึ้น
        current_dir = getattr(self, direction_attr)

        if abs(diff) < self.STEP_THRESH:
            return 0  # เคลื่อนที่น้อยเกิน → ไม่นับ (กรอง Jitter)

        new_dir = 1 if diff > 0 else -1

        # ทิศกลับจาก "ลง" (1) → "ขึ้น" (-1) = เท้ายกขึ้น = ก้าวใหม่
        if current_dir == 1 and new_dir == -1:
            setattr(self, direction_attr, new_dir)
            return 1

        setattr(self, direction_attr, new_dir)
        return 0

    def _reset_steps(self):
        """รีเซ็ตตัวนับก้าวและ Direction State"""
        self.steps = 0
        self._l_direction = 0
        self._r_direction = 0
        self._prev_l_smooth = None
        self._prev_r_smooth = None
        self._l_hist.clear()
        self._r_hist.clear()