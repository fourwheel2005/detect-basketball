"""
referee.py — ศูนย์กลางการตัดสิน (Orchestrator)
------------------------------------------------
กฎทั้งหมดที่ตรวจสอบ:
    Rule 1 — Double Dribble   (State Machine)
    Rule 2 — Traveling        (Peak Detection + Rolling Average)
    Rule 3 — Carrying         (Wrist vs Index Y + Confirm Buffer)
    Rule 4 — Goaltending      (Parabolic Trajectory Analysis)
    Rule 5 — Jump Ball Foul   (Phase Detection + Velocity + Elbow Angle)
    Rule 6 — Contact Foul     (3D CNN — ตีแขน / ชนตัว)
"""

from utils import get_dist, AccuracyTracker
from fouls.double_dribble import DoubleDribbleDetector
from fouls.traveling      import TravelingDetector
from fouls.carrying       import CarryingDetector
from fouls.goaltending    import GoaltendingDetector
# from fouls.jump_ball      import JumpBallDetector

# Contact Foul (3D CNN) — import แบบ Optional
# ถ้ายังไม่มี model ไฟล์ หรือยังไม่ได้ติดตั้ง tensorflow
# ระบบจะข้าม Rule นี้โดยอัตโนมัติ ไม่ทำให้โปรแกรม crash
try:
    from fouls.contact_foul import ContactFoulDetector
    CONTACT_FOUL_AVAILABLE = True
except ImportError:
    CONTACT_FOUL_AVAILABLE = False
    print("⚠️  ContactFoulDetector ไม่พร้อมใช้งาน "
          "(ติดตั้ง tensorflow และ train model ก่อน)")


# ─────────────────────────────────────────────────────
#  Pose Validity Check (Module-level function)
# ─────────────────────────────────────────────────────

_MAX_MISSING = 2  # ขาด Landmark หลักได้สูงสุดกี่จุด


def _is_pose_valid(landmarks_px: dict, mp_pose) -> tuple:
    """
    ตรวจสอบว่า Pose มี Landmark สำคัญครบหรือไม่
    ถ้าไม่ครบ = น่าจะเป็น False Detection (มือ/แขนลอยๆ ไม่ใช่คน)

    Conditions:
        1. Landmark หลัก 5 จุด ต้องขาดไม่เกิน _MAX_MISSING จุด
        2. Aspect Ratio ต้องสูงกว่ากว้าง (คนยืน/เดิน)

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

    # ใช้ .value เพราะ key ใน landmarks_px เป็น int
    missing = [lm for lm in required if lm.value not in landmarks_px]
    if len(missing) > _MAX_MISSING:
        return False, f"Pose incomplete ({len(missing)} key points missing)"

    xs = [v[0] for v in landmarks_px.values()]
    ys = [v[1] for v in landmarks_px.values()]
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)

    if w > 0 and h > 0 and (h / w) < 0.5:
        return False, "Aspect ratio invalid (likely arm/hand)"

    return True, ""


# ─────────────────────────────────────────────────────
#  BasketballRef — Main Orchestrator
# ─────────────────────────────────────────────────────

class BasketballRef:
    """
    จัดการ Detector ของแต่ละผู้เล่น (แยกกันตาม Player ID)
    และรวบรวม Violations + Info Texts ส่งกลับให้ main.py

    Attributes:
        HOLDING_DIST (int) : ระยะ (px) ที่ถือว่า "ถือบอล"
        _players     (dict): {player_id: {detector_key: DetectorInstance}}
        _latest_landmarks  : {player_id: landmarks_px} — ใช้หา opponent
    """

    HOLDING_DIST = 120  # px

    def __init__(self):
        self._players           : dict = {}
        self._latest_landmarks  : dict = {}
        self.accuracy           = AccuracyTracker()

    # ─── Public API ───────────────────────────────────

   # ไฟล์ referee.py (ตัดมาเฉพาะส่วนฟังก์ชัน process เพื่ออัปเดต)
    def process(self, p_id, landmarks_px, mp_pose, ball_box, frame_w, frame_h):
        detectors = self.get_detectors(p_id)
        violations = []
        info_texts = []

        # 📌 1. Dynamic Scale: หาความกว้างไหล่ (Shoulder Width) เพื่อใช้เป็นเกณฑ์วัด
        l_shoulder = landmarks_px[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks_px[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        from utils import get_dist, is_point_near_box
        shoulder_width = get_dist(l_shoulder, r_shoulder)
        if shoulder_width < 10: shoulder_width = 10 # ป้องกันค่า 0

        # 📌 2. Dynamic Possession: เช็คการครองบอลจาก Bounding Box และปลายนิ้ว
        is_holding = False
        ball_center = None
        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            ball_center = ((bx1+bx2)/2, (by1+by2)/2)
            
            r_index = landmarks_px[mp_pose.PoseLandmark.RIGHT_INDEX]
            l_index = landmarks_px[mp_pose.PoseLandmark.LEFT_INDEX]
            
            # Margin แปรผันตามขนาดตัว (ไหล่)
            margin = int(shoulder_width * 0.2) 
            
            if is_point_near_box(r_index[0], r_index[1], ball_box, margin) or \
               is_point_near_box(l_index[0], l_index[1], ball_box, margin):
                is_holding = True

        # --- Check Rules ---
        
        # Double Dribble
        is_dd, msg_dd = detectors["dd"].check(landmarks_px, mp_pose, ball_center)
        if is_dd: violations.append(msg_dd)

        # 📌 3. Traveling (ส่ง shoulder_width ไปคำนวณ Dynamic Threshold ด้วย)
        is_tr, msg_tr = detectors["tr"].check(landmarks_px, mp_pose, is_holding, shoulder_width)
        if is_tr: violations.append(msg_tr)
        elif "Steps" in msg_tr: info_texts.append(msg_tr)

        # Carrying
        is_ca, msg_ca = detectors["ca"].check(landmarks_px, mp_pose, is_holding)
        if is_ca: violations.append(msg_ca)

        return violations, info_texts

    def cleanup_player(self, player_id: int):
        """
        เรียกเมื่อ Player ID หายออกจากเฟรม
        - Reset JumpBallDetector (Baseline ข้อเท้า)
        - ลบ landmarks ออกจาก cache

        ควรเรียกจาก main.py เมื่อ valid_ids ลดลง
        """
      #  if player_id in self._players:
         #   self._players[player_id]["jb"].reset()
        self._latest_landmarks.pop(player_id, None)

    # ─── Private Helpers ──────────────────────────────

    def _get_detectors(self, player_id: int) -> dict:
        """
        ดึง (หรือสร้างใหม่) ชุด Detectors สำหรับผู้เล่นคนนั้น
        แยก Instance ต่อผู้เล่น → State ไม่ปนกัน
        """
        if player_id not in self._players:
            detectors = {
                "dd": DoubleDribbleDetector(),
                "tr": TravelingDetector(),
                "ca": CarryingDetector(),
                "gt": GoaltendingDetector(),
                #"jb": JumpBallDetector(),
            }
            # เพิ่ม Contact Foul เฉพาะเมื่อพร้อม
            if CONTACT_FOUL_AVAILABLE:
                detectors["cf"] = ContactFoulDetector()

            self._players[player_id] = detectors

        return self._players[player_id]

    def _check_holding(self, landmarks_px: dict, mp_pose,
                       ball_center) -> bool:
        """
        ตรวจสอบว่าผู้เล่นถือบอลหรือไม่
        เช็คระยะห่างระหว่างข้อมือกับจุดกึ่งกลางบอล
        """
        if ball_center is None:
            return False

        r_key = mp_pose.PoseLandmark.RIGHT_WRIST.value
        l_key = mp_pose.PoseLandmark.LEFT_WRIST.value

        r_wrist = landmarks_px.get(r_key)
        l_wrist = landmarks_px.get(l_key)

        if r_wrist is None or l_wrist is None:
            return False

        return (get_dist(r_wrist, ball_center) < self.HOLDING_DIST or
                get_dist(l_wrist, ball_center) < self.HOLDING_DIST)

    def _find_nearest_opponent(self, player_id: int,
                                my_landmarks: dict):
        """
        ค้นหาผู้เล่นที่อยู่ใกล้ที่สุดเพื่อใช้เป็น opponent
        สำหรับ JumpBallDetector ตรวจ Push contact

        ใช้ LEFT_HIP (id=23) เป็นจุดกึ่งกลางตัว
        คืน None ถ้าไม่มีผู้เล่นอื่น หรืออยู่ไกลเกิน 300px
        """
        my_hip = my_landmarks.get(23)   # LEFT_HIP.value = 23
        if my_hip is None:
            return None

        nearest_lm   = None
        nearest_dist = float('inf')

        for pid, lm in self._latest_landmarks.items():
            if pid == player_id:
                continue

            opp_hip = lm.get(23)
            if opp_hip is None:
                continue

            d = get_dist(my_hip, opp_hip)
            if d < nearest_dist:
                nearest_dist = d
                nearest_lm   = lm

        # ไกลเกิน 300px = ไม่ใช่คู่ Jump Ball
        return nearest_lm if nearest_dist <= 300 else None