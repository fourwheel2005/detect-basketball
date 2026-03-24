"""
fouls/jump_ball.py — ตรวจจับ Jump Ball Foul (Fixed Version)
============================================================
แก้ไขปัญหา False Positive:

  ปัญหาเดิม:
    1. นับ Jump ขณะนั่ง — JUMP_THRESHOLD ต่ำเกิน (25px)
       และ Baseline เก็บจาก Rolling Average ปกติ
       แต่ขณะนั่ง ankle Y นิ่ง → Baseline ผิด
    2. PUSH FOUL รัว — threshold 18px/frame ต่ำมาก
       การขยับหัว/ลำตัวปกติก็ทำให้ Wrist Landmark กระเด้ง
    3. ไม่มี Cooldown ระหว่าง Push Foul

  การแก้ไข:
    1. JUMP_THRESHOLD เพิ่มเป็น 60px (ต้องยกเท้าสูงชัดเจน)
    2. ต้องมี Hip ลอยขึ้นด้วย ≥ JUMP_THRESHOLD*0.6
    3. AIRBORNE_MIN เพิ่มเป็น 5 เฟรม (ต้องลอยอย่างน้อย ~0.17s)
    4. PUSH_VEL_THRESH เพิ่มเป็น 80px/frame
    5. เพิ่ม PUSH_COOLDOWN ระดับ Detector (30 เฟรม ≈ 1 วินาที)
    6. ตรวจ Push เฉพาะตอน AIRBORNE จริงๆ เท่านั้น
    7. ต้องกระโดดติดต่อกัน (REQUIRE_BOTH_FEET) ห้ามขาข้างเดียว
"""

from collections import deque
from utils import get_dist, calculate_angle


class JumpBallDetector:
    """
    ตรวจจับ Jump Ball Foul ด้วยการวิเคราะห์ Phase การกระโดด

    Foul ที่ตรวจ:
        PUSH FOUL     — ข้อมือเคลื่อนเร็วเกิน threshold ขณะ Airborne
        ILLEGAL HANDS — ข้อศอกกางเกิน 155° ขณะ Airborne

    Thresholds (ปรับได้ตามความสูงกล้องและ Resolution):
        JUMP_THRESHOLD   = 60px   ← ยกสูงจาก Baseline กี่ px ถึงเรียก Jump
        PUSH_VEL_THRESH  = 80     ← px/frame — การขยับมือปกติ ≈ 10–30px/f
        ELBOW_ANGLE_MAX  = 155    ← องศา ข้อศอกกาง
        AIRBORNE_MIN     = 5      ← เฟรมที่ต้องลอยขั้นต่ำ
        PUSH_COOLDOWN    = 30     ← เฟรม cooldown หลัง Push ถูก trigger
    """

    # ── Jump Detection ──────────────────────────────
    JUMP_THRESHOLD   = 60    # px — ยกจาก Baseline กี่ px ถึงเรียก Jump
    HIP_LIFT_RATIO   = 0.6   # Hip ต้องลอยขึ้น ≥ JUMP_THRESHOLD × ratio นี้
    AIRBORNE_MIN     = 5     # เฟรมขั้นต่ำที่ต้องลอยอยู่
    BASELINE_FRAMES  = 40    # เฟรมสำหรับเก็บ Baseline (เพิ่มจาก 30)
    BASELINE_MIN     = 20    # ต้องมีข้อมูลอย่างน้อยกี่เฟรมก่อนตัดสิน

    # ── Foul Detection ───────────────────────────────
    PUSH_VEL_THRESH  = 80    # px/frame — เพิ่มจาก 18 → 80
    ELBOW_ANGLE_MAX  = 155   # องศา
    CONFIRM_FRAMES   = 3     # เฟรมต่อเนื่องก่อน Confirm
    PUSH_COOLDOWN    = 30    # เฟรม cooldown หลัง Push (กัน Log รัว)

    # ── Phases ───────────────────────────────────────
    PHASE_IDLE     = "IDLE"
    PHASE_RISING   = "RISING"
    PHASE_AIRBORNE = "AIRBORNE"
    PHASE_LANDING  = "LANDING"

    def __init__(self):
        # Baseline ของข้อเท้า (เก็บเฉพาะตอน IDLE)
        self._l_ankle_baseline : deque = deque(maxlen=self.BASELINE_FRAMES)
        self._r_ankle_baseline : deque = deque(maxlen=self.BASELINE_FRAMES)
        # Baseline ของ Hip (เพิ่มมาเพื่อตรวจ Hip lift ด้วย)
        self._hip_baseline     : deque = deque(maxlen=self.BASELINE_FRAMES)

        # Wrist ก่อนหน้า (สำหรับคำนวณ velocity)
        self._prev_r_wrist = None
        self._prev_l_wrist = None

        # Jump Phase State
        self.phase          = self.PHASE_IDLE
        self._airborne_cnt  = 0

        # Confirm Counters
        self._push_confirm   = 0
        self._elbow_confirm  = 0

        # Cooldown counter (นับถอยหลัง)
        self._push_cd_frames = 0

        # สถิติ
        self.jump_count = 0

    # ─── Public API ────────────────────────────────────────

    def check(self, landmarks_px: dict, mp_pose,
              opponent_landmarks: dict = None) -> tuple:
        """
        ตรวจสอบ Jump Ball Foul ใน 1 เฟรม

        Returns:
            (is_violation: bool, message: str)
        """
        lm = landmarks_px

        # ดึง Landmarks ที่ต้องใช้
        l_ankle    = lm.get(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        r_ankle    = lm.get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        l_hip      = lm.get(mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip      = lm.get(mp_pose.PoseLandmark.RIGHT_HIP.value)
        l_wrist    = lm.get(mp_pose.PoseLandmark.LEFT_WRIST.value)
        r_wrist    = lm.get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        l_elbow    = lm.get(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        r_elbow    = lm.get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        l_shoulder = lm.get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_shoulder = lm.get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)

        # Landmark ไม่ครบ → ข้ามเฟรมนี้
        if not all([l_ankle, r_ankle, l_hip, r_hip]):
            return False, f"Phase: {self.phase} | Jumps: {self.jump_count}"

        # ── อัปเดต Baseline เฉพาะตอน IDLE ──
        if self.phase == self.PHASE_IDLE:
            self._l_ankle_baseline.append(l_ankle[1])
            self._r_ankle_baseline.append(r_ankle[1])
            hip_y = (l_hip[1] + r_hip[1]) / 2
            self._hip_baseline.append(hip_y)

        # ── ตรวจ Phase ──
        self._update_phase(l_ankle, r_ankle, l_hip, r_hip)

        # ── ลด Cooldown ──
        if self._push_cd_frames > 0:
            self._push_cd_frames -= 1

        # ── ตรวจฟาวล์เฉพาะตอน AIRBORNE ──
        result = (False, "")

        if self.phase == self.PHASE_AIRBORNE:
            # Push Foul
            if l_wrist and r_wrist:
                push = self._check_push(l_wrist, r_wrist, opponent_landmarks)
                if push[0]:
                    self._push_confirm += 1
                    if self._push_confirm >= self.CONFIRM_FRAMES:
                        result = push
                else:
                    self._push_confirm = 0

            # Illegal Hands
            if all([l_elbow, r_elbow, l_shoulder, r_shoulder,
                    l_wrist, r_wrist]):
                elbow = self._check_elbow(
                    l_shoulder, r_shoulder,
                    l_elbow,    r_elbow,
                    l_wrist,    r_wrist,
                )
                if elbow[0]:
                    self._elbow_confirm += 1
                    if self._elbow_confirm >= self.CONFIRM_FRAMES:
                        result = elbow
                else:
                    self._elbow_confirm = 0

        elif self.phase == self.PHASE_IDLE:
            # Reset Confirm เมื่อลงพื้นแล้ว
            self._push_confirm  = 0
            self._elbow_confirm = 0

        # อัปเดต Wrist สำหรับเฟรมถัดไป
        self._prev_r_wrist = r_wrist
        self._prev_l_wrist = l_wrist

        msg = f"Phase: {self.phase} | Jumps: {self.jump_count}"
        return result if result[0] else (False, msg)

    # ─── Phase Detection ───────────────────────────────────

    def _update_phase(self, l_ankle, r_ankle, l_hip, r_hip):
        """
        อัปเดต Phase การกระโดดจาก Ankle Y และ Hip Y

        เงื่อนไขที่ต้องผ่านพร้อมกัน ถึงจะเรียก AIRBORNE:
            1. มีข้อมูล Baseline ≥ BASELINE_MIN เฟรม
            2. ข้อเท้าทั้งสองลอยขึ้นจาก Baseline ≥ JUMP_THRESHOLD
            3. Hip ลอยขึ้นด้วย ≥ JUMP_THRESHOLD × HIP_LIFT_RATIO
               (กัน False Positive กรณีนั่งยองหรือก้มตัว)
        """
        # รอข้อมูล Baseline ก่อน
        if len(self._l_ankle_baseline) < self.BASELINE_MIN:
            return

        base_l   = sum(self._l_ankle_baseline) / len(self._l_ankle_baseline)
        base_r   = sum(self._r_ankle_baseline) / len(self._r_ankle_baseline)
        base_hip = sum(self._hip_baseline)     / len(self._hip_baseline)

        # ระยะที่ยกขึ้นจาก Baseline (บวก = สูงขึ้น ค่า Y น้อยลง)
        lift_l   = base_l   - l_ankle[1]
        lift_r   = base_r   - r_ankle[1]
        hip_y    = (l_hip[1] + r_hip[1]) / 2
        hip_lift = base_hip - hip_y

        # ต้องทั้งสองเท้าพ้นพื้นพร้อมกัน AND Hip ลอยขึ้นด้วย
        both_up  = (lift_l > self.JUMP_THRESHOLD and
                    lift_r > self.JUMP_THRESHOLD)
        hip_ok   = hip_lift > (self.JUMP_THRESHOLD * self.HIP_LIFT_RATIO)

        is_jumping = both_up and hip_ok

        if is_jumping:
            if self.phase in (self.PHASE_IDLE, self.PHASE_RISING):
                self._airborne_cnt += 1
                if self._airborne_cnt >= self.AIRBORNE_MIN:
                    if self.phase != self.PHASE_AIRBORNE:
                        self.jump_count += 1
                    self.phase = self.PHASE_AIRBORNE
            else:
                self.phase = self.PHASE_AIRBORNE
        else:
            if self.phase == self.PHASE_AIRBORNE:
                self.phase = self.PHASE_LANDING
            elif self.phase == self.PHASE_LANDING:
                self.phase         = self.PHASE_IDLE
                self._airborne_cnt = 0
            elif self.phase == self.PHASE_IDLE:
                self._airborne_cnt = 0

    # ─── Foul Checks ───────────────────────────────────────

    def _check_push(self, l_wrist, r_wrist,
                    opponent_lm: dict) -> tuple:
        """
      
        """
        # ยังอยู่ใน Cooldown → ไม่ตรวจ
        if self._push_cd_frames > 0:
            return False, ""

        # ── ตรวจ Velocity ──
        if self._prev_r_wrist is not None and self._prev_l_wrist is not None:
            vel_r = get_dist(r_wrist, self._prev_r_wrist)
            vel_l = get_dist(l_wrist, self._prev_l_wrist)
            max_vel = max(vel_r, vel_l)

            if max_vel > self.PUSH_VEL_THRESH:
                self._push_cd_frames = self.PUSH_COOLDOWN  # เริ่ม Cooldown
                hand = "R" if vel_r > vel_l else "L"
                return True, f"PUSH FOUL ({hand} vel:{max_vel:.0f}px/f)"

        # ── ตรวจ Contact กับคู่ต่อสู้ ──
        if opponent_lm:
            l_s = opponent_lm.get(11)  # LEFT_SHOULDER
            r_s = opponent_lm.get(12)  # RIGHT_SHOULDER
            if l_s and r_s:
                chest = ((l_s[0]+r_s[0])/2, (l_s[1]+r_s[1])/2)
                if min(get_dist(r_wrist, chest),
                       get_dist(l_wrist, chest)) < 80:
                    self._push_cd_frames = self.PUSH_COOLDOWN
                    return True, "PUSH FOUL (contact)"

        return False, ""

    def _check_elbow(self, l_shoulder, r_shoulder,
                     l_elbow, r_elbow,
                     l_wrist, r_wrist) -> tuple:
        """
        ตรวจ Illegal Hands: ข้อศอกกางเกิน ELBOW_ANGLE_MAX ขณะ Airborne
        เพิ่มเงื่อนไข: ต้อง "ยกแขนขึ้น" ด้วย ถึงจะนับว่ากางศอกขวางคนอื่น
        """
        angle_r = calculate_angle(r_shoulder, r_elbow, r_wrist)
        angle_l = calculate_angle(l_shoulder, l_elbow, l_wrist)
        max_ang = max(angle_r, angle_l)
        side    = "R" if angle_r > angle_l else "L"

        # เช็คว่ายกแขนไหม (เพื่อกันเคสยืนปล่อยแขนห้อยลงพื้นเฉยๆ)
        # ในแกนภาพ Y ยิ่งน้อยคือยิ่งอยู่สูง (เราเช็คว่าข้อมือต้องอยู่สูงกว่าระดับหน้าอก)
        is_r_raised = r_wrist[1] < r_shoulder[1] + 50
        is_l_raised = l_wrist[1] < l_shoulder[1] + 50

        if max_ang > self.ELBOW_ANGLE_MAX and (is_r_raised or is_l_raised):
            # เอาตัวเลของศาออกจากข้อความ Log เพื่อให้ Cooldown ของ FoulLogger ทำงานได้ปกติ
            return True, f"ILLEGAL HANDS ({side})"
            
        return False, ""

    # ─── Helpers ───────────────────────────────────────────

    def is_airborne(self) -> bool:
        return self.phase == self.PHASE_AIRBORNE

    def reset(self):
        """Reset ทั้งหมด — เรียกเมื่อ Player ID หายออกจากเฟรม"""
        self.phase           = self.PHASE_IDLE
        self._airborne_cnt   = 0
        self._push_confirm   = 0
        self._elbow_confirm  = 0
        self._push_cd_frames = 0
        self._prev_r_wrist   = None
        self._prev_l_wrist   = None
        self.jump_count      = 0
        self._l_ankle_baseline.clear()
        self._r_ankle_baseline.clear()
        self._hip_baseline.clear()