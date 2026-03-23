"""
fouls/carrying.py — ตรวจจับ Carrying (Palming)
-----------------------------------------------
Logic:
    ขณะถือบอล ถ้าข้อมือ (Wrist) อยู่ต่ำกว่านิ้ว (Index Finger) ชัดเจน
    → แสดงว่า "ฝ่ามือคว่ำรองบอล" = Carrying

การแก้ไขจากเวอร์ชันเดิม:
    - เพิ่ม Confirmation Buffer: ต้องเกิดขึ้นต่อเนื่อง N เฟรม ถึงจะแจ้ง
      เพื่อกรองกรณีมือไหวชั่วครู่
"""


class CarryingDetector:
    """
    ตรวจจับ Carrying โดยเปรียบเทียบตำแหน่ง Y ของ Wrist vs Index Finger

    ถ้า wrist_y > index_y + BUFFER → มือคว่ำ = Carrying
    (Y มากกว่า = ต่ำกว่าในพิกัดภาพ)
    """

    Y_BUFFER        = 20   # ระยะ Y (px) ที่ข้อมือต้องต่ำกว่านิ้ว
    CONFIRM_FRAMES  = 5    # ต้องเกิดต่อเนื่องกี่เฟรมถึง Confirm

    def __init__(self):
        self._consecutive = 0  # นับเฟรมที่ตรงเงื่อนไขติดต่อกัน

    def check(self, landmarks_px: dict, mp_pose, is_holding: bool) -> tuple:
        """
        Parameters:
            landmarks_px : dict {landmark_id: (x, y)} — Pixel
            mp_pose      : mediapipe.solutions.pose
            is_holding   : bool — ผู้เล่นถือบอลอยู่หรือไม่

        Returns:
            (is_violation: bool, message: str)
        """
        if not is_holding:
            self._consecutive = 0
            return False, ""

        # ── ดึงพิกัด Y ของข้อมือและนิ้วชี้ ──
        r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_index = landmarks_px[mp_pose.PoseLandmark.RIGHT_INDEX]
        l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST]
        l_index = landmarks_px[mp_pose.PoseLandmark.LEFT_INDEX]

        # ข้อมืออยู่ต่ำกว่านิ้วชี้ + Buffer = มือคว่ำ
        r_carry = r_wrist[1] > (r_index[1] + self.Y_BUFFER)
        l_carry = l_wrist[1] > (l_index[1] + self.Y_BUFFER)

        if r_carry or l_carry:
            self._consecutive += 1
        else:
            self._consecutive = 0

        # ต้องเกิดต่อเนื่องเพื่อ Confirm (กรอง Noise)
        if self._consecutive >= self.CONFIRM_FRAMES:
            return True, "CARRYING"

        return False, ""