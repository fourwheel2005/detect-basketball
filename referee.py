"""
referee.py — ศูนย์กลางการตัดสิน (Orchestrator)
------------------------------------------------
BasketballRef รับข้อมูล Pose + Ball ของแต่ละผู้เล่น
แล้วส่งต่อให้ Detector แต่ละตัว และรวบรวมผลลัพธ์
"""

import numpy as np
from utils import get_dist, AccuracyTracker
from fouls.double_dribble import DoubleDribbleDetector
from fouls.traveling      import TravelingDetector
from fouls.carrying       import CarryingDetector
from fouls.goaltending    import GoaltendingDetector


# ─────────────────────────────────────────────────────
#  ฟังก์ชัน Module-level (อยู่นอก class — เรียกได้จากทุกที่)
# ─────────────────────────────────────────────────────

_MAX_MISSING = 2  # ขาด Landmark หลักได้สูงสุดกี่จุดก่อนถือว่า Invalid


def _is_pose_valid(landmarks_px: dict, mp_pose) -> tuple:
    """
    ตรวจสอบว่า Pose มี Landmark สำคัญครบหรือไม่
    ถ้าไม่ครบ = น่าจะเป็น False Detection (มือ/แขนลอยๆ ไม่ใช่คน)

    Parameters:
        landmarks_px : dict {landmark_id (int): (x, y)}
        mp_pose      : mediapipe.solutions.pose

    Returns:
        (is_valid: bool, reason: str)
    """
    required = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]

    # ใช้ .value เพราะ landmarks_px key เป็น int ไม่ใช่ Enum
    missing = [lm for lm in required if lm.value not in landmarks_px]

    if len(missing) > _MAX_MISSING:
        return False, f"Pose incomplete ({len(missing)} key points missing)"

    # เช็ค Aspect Ratio: คนจริงต้องสูงกว่ากว้าง
    xs = [v[0] for v in landmarks_px.values()]
    ys = [v[1] for v in landmarks_px.values()]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)

    if w > 0 and h > 0 and (h / w) < 0.5:
        return False, "Aspect ratio invalid (likely arm/hand)"

    return True, ""


# ─────────────────────────────────────────────────────
#  Class หลัก
# ─────────────────────────────────────────────────────

class BasketballRef:
    """
    จัดการ Detector ของแต่ละผู้เล่น (แยกกันตาม Player ID)
    และรวบรวม Violations + Info Texts ส่งกลับให้ main.py

    Attributes:
        HOLDING_DIST (int): ระยะ (px) ที่ถือว่า "ถือบอล"
    """

    HOLDING_DIST = 120  # pixel

    def __init__(self):
        # dict ของ Detector แยกตาม Player ID
        # {player_id: {"dd": ..., "tr": ..., "ca": ..., "gt": ...}}
        self._players : dict = {}

        # ระบบติดตามความแม่นยำ (shared ข้ามผู้เล่น)
        self.accuracy = AccuracyTracker()

    # ─── Public API ───────────────────────────────────

    def process(self, player_id: int, landmarks_px: dict,
                mp_pose, ball_box, frame_w: int, frame_h: int) -> tuple:
        """
        ประมวลผลผู้เล่น 1 คน ใน 1 เฟรม

        Parameters:
            player_id   : int  — TrackID จาก YOLO
            landmarks_px: dict — {landmark_id (int): (x, y)} Pixel coordinates
            mp_pose     : mediapipe.solutions.pose module
            ball_box    : array [x1, y1, x2, y2] หรือ None
            frame_w/h   : ขนาดเฟรม (pixel)

        Returns:
            violations (list[str]): รายชื่อฟาวล์ที่พบ
            info_texts (list[str]): ข้อมูล Debug เช่น "Steps: 1"
        """
        # ── ด่านที่ 1: ตรวจสอบความสมบูรณ์ของ Pose ──
        # ถ้า Landmark ไม่ครบ (เช่น เห็นแค่มือ/แขน) → ข้ามการตัดสิน
        is_valid, reason = _is_pose_valid(landmarks_px, mp_pose)
        if not is_valid:
            return [], [f"[SKIP] {reason}"]

        detectors  = self._get_detectors(player_id)
        violations : list = []
        info_texts : list = []

        # ── คำนวณ Ball Center จาก Bounding Box ──
        ball_center = None
        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

        # ── ตรวจสอบว่าผู้เล่นถือบอลอยู่หรือไม่ ──
        is_holding = self._check_holding(landmarks_px, mp_pose, ball_center)

        # ── Rule 1: Double Dribble ──
        is_dd, msg_dd = detectors["dd"].check(landmarks_px, mp_pose, ball_center)
        self.accuracy.record("DOUBLE_DRIBBLE", is_dd)
        if is_dd:
            violations.append(msg_dd)

        # ── Rule 2: Traveling ──
        is_tr, msg_tr = detectors["tr"].check(landmarks_px, mp_pose, is_holding)
        self.accuracy.record("TRAVELING", is_tr)
        if is_tr:
            violations.append(msg_tr)
        elif msg_tr.startswith("Steps"):
            info_texts.append(msg_tr)

        # ── Rule 3: Carrying ──
        is_ca, msg_ca = detectors["ca"].check(landmarks_px, mp_pose, is_holding)
        self.accuracy.record("CARRYING", is_ca)
        if is_ca:
            violations.append(msg_ca)

        # ── Rule 4: Goaltending (เฉพาะตอนถือบอลและบอลอยู่สูง) ──
        is_gt, msg_gt = detectors["gt"].check(ball_center, frame_h)
        self.accuracy.record("GOALTENDING", is_gt and is_holding)
        if is_gt and is_holding:
            violations.append(msg_gt)

        # แสดงสถานะ Holding เป็น Info
        if is_holding:
            info_texts.append("Holding Ball")

        return violations, info_texts

    # ─── Private Helpers ──────────────────────────────

    def _get_detectors(self, player_id: int) -> dict:
        """
        ดึง (หรือสร้างใหม่) ชุด Detectors สำหรับผู้เล่นคนนั้น
        แยก Instance ต่อผู้เล่น เพื่อไม่ให้ State ปนกัน
        """
        if player_id not in self._players:
            self._players[player_id] = {
                "dd": DoubleDribbleDetector(),
                "tr": TravelingDetector(),
                "ca": CarryingDetector(),
                "gt": GoaltendingDetector(),
            }
        return self._players[player_id]

    def _check_holding(self, landmarks_px: dict, mp_pose,
                       ball_center) -> bool:
        """
        ตรวจสอบว่าผู้เล่นถือบอลหรือไม่
        โดยดูระยะห่างระหว่างข้อมือกับจุดกึ่งกลางบอล
        """
        if ball_center is None:
            return False

        # ใช้ .value เพราะ key ใน landmarks_px เป็น int
        r_key = mp_pose.PoseLandmark.RIGHT_WRIST.value
        l_key = mp_pose.PoseLandmark.LEFT_WRIST.value

        r_wrist = landmarks_px.get(r_key)
        l_wrist = landmarks_px.get(l_key)

        if r_wrist is None or l_wrist is None:
            return False

        return (get_dist(r_wrist, ball_center) < self.HOLDING_DIST or
                get_dist(l_wrist, ball_center) < self.HOLDING_DIST)