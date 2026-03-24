"""
main.py — AI Basketball Referee (Complete Version)
===================================================
ระบบ AI ตรวจจับฟาวล์บาสเกตบอลแบบ Real-time ด้วย:
    - YOLOv8         : ตรวจจับผู้เล่นและลูกบอล
    - MediaPipe Pose : วิเคราะห์ท่าทางร่างกาย
    - Rule Engine    : 6 กฎ (Traveling / Carrying / Double Dribble /
                       Goaltending / Jump Ball / Contact Foul)

ฟีเจอร์:
    ✅ บันทึก Log ลง CSV
    ✅ Accuracy Report หลัง Session จบ
    ✅ Noise Filtering 3 ด่าน (IoU + Stability + Pose Valid)
    ✅ รองรับกล้องมือถือผ่าน IP Webcam
    ✅ cleanup_player() เมื่อผู้เล่นออกจากเฟรม
    ✅ ส่ง roi_bgr ให้ Contact Foul (3D CNN) อัตโนมัติ

Controls:
    q  — ออก (พิมพ์ Accuracy Report)
    r  — Reset Accuracy Tracker
"""

import cv2
import mediapipe as mp
from ultralytics import YOLO

from referee import BasketballRef
from utils   import (FoulLogger, PlayerStabilityFilter,
                     filter_duplicate_boxes)


# ════════════════════════════════════════════════════
#  Config — ปรับค่าทั้งหมดที่นี่ที่เดียว
# ════════════════════════════════════════════════════

# ── แหล่งกล้อง ──
# "webcam"  = กล้องคอม
# "phone"   = มือถือผ่าน IP Webcam (Android) / DroidCam
CAMERA_SOURCE = "webcam"

CAMERA_INDEX  = 0     # ใช้เมื่อ CAMERA_SOURCE = "webcam"

# IP Webcam — ดู IP จากแอปแล้วใส่ตรงนี้
PHONE_IP      = "192.168.1.5"
PHONE_PORT    = 8080
PHONE_URL     = f"http://{PHONE_IP}:{PHONE_PORT}/video"

FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
YOLO_CONF     = 0.35    # Confidence ขั้นต่ำของ YOLO
LOG_FILE      = "basketball_foul_logs.csv"
LOG_COOLDOWN  = 3.0     # วินาที — ป้องกัน Log ซ้ำรัวๆ
FLIP_CAMERA   = True    # True = กล้องหน้า (Mirror) — ปิดเมื่อใช้กล้องมือถือ


# ════════════════════════════════════════════════════
#  ฟังก์ชันเปิดกล้อง
# ════════════════════════════════════════════════════

def open_camera() -> cv2.VideoCapture:
    """
    เปิดกล้องตาม CAMERA_SOURCE
    รองรับ: webcam, phone (IP Webcam / DroidCam)

    Returns:
        cv2.VideoCapture หรือ None ถ้าเปิดไม่ได้
    """
    if CAMERA_SOURCE == "phone":
        print(f"📱 Connecting to phone camera: {PHONE_URL}")
        cap = cv2.VideoCapture(PHONE_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ลด Latency
    else:
        print(f"💻 Opening webcam index: {CAMERA_INDEX}")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        if CAMERA_SOURCE == "phone":
            print("❌ เชื่อมต่อมือถือไม่ได้ — ตรวจสอบ:")
            print(f"   1. มือถือและคอมอยู่ WiFi เดียวกัน")
            print(f"   2. IP ถูกต้อง: {PHONE_IP}")
            print(f"   3. กด 'Start server' ในแอป IP Webcam แล้ว")
        else:
            print(f"❌ ไม่พบกล้อง index {CAMERA_INDEX}")
        return None

    print("✅ Camera connected!")
    return cap


# ════════════════════════════════════════════════════
#  ฟังก์ชัน Draw UI
# ════════════════════════════════════════════════════

def draw_player(frame, box, player_id, violations, info_texts):
    """
    วาด UI สำหรับผู้เล่น 1 คน: กรอบ, ข้อมูล Debug, Violation Banner

    Parameters:
        frame      : frame หลัก (BGR)
        box        : (x1, y1, x2, y2)
        player_id  : int
        violations : list[str]
        info_texts : list[str]
    """
    x1, y1, x2, y2 = map(int, box)
    color = (0, 0, 255) if violations else (0, 220, 80)

    # กรอบผู้เล่น
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Player ID Badge
    cv2.putText(frame, f"P{player_id}", (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Info Texts เหนือกรอบ (กรอง [SKIP] ออก)
    label_y = y1 - 10
    for info in info_texts:
        if info.startswith("[SKIP]"):
            continue
        cv2.putText(frame, info, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 230, 50), 2)
        label_y -= 20

    # Violation Banner พร้อมพื้นหลังกึ่งโปร่งใส
    if violations:
        text = " + ".join(violations)
        font         = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.75, 2
        (tw, th), _  = cv2.getTextSize(text, font, scale, thick)

        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x1, y1 - th - 16),
                      (x1 + tw + 8, y1 - 4),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, text, (x1 + 4, y1 - 8),
                    font, scale, (0, 80, 255), thick)


def draw_ball(frame, ball_box):
    """วาดกรอบรอบลูกบอล"""
    bx1, by1, bx2, by2 = map(int, ball_box)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
    cv2.putText(frame, "Ball", (bx1, by1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


def draw_hud(frame, frame_count, num_players):
    """HUD มุมซ้ายบน"""
    cv2.putText(frame,
                f"Frame: {frame_count}  Players: {num_players}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════

def main():
    # ── โหลด AI Models ──
    print("⏳ Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("⏳ Initializing MediaPipe Pose...")
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    )

    # ── เตรียม Systems ──
    referee   = BasketballRef()
    logger    = FoulLogger(filename=LOG_FILE, cooldown_sec=LOG_COOLDOWN)
    stability = PlayerStabilityFilter()

    # ── เปิดกล้อง ──
    cap = open_camera()
    if cap is None:
        return

    print("✅ System Ready! กด 'q' เพื่อออก, 'r' เพื่อ Reset Accuracy")
    frame_count      = 0
    prev_valid_ids   = set()   # ใช้ตรวจจับ Player ID ที่หายออกไป

    # ════════════════════════════════════════════════
    #  Main Loop
    # ════════════════════════════════════════════════
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("⚠️ ไม่ได้รับ Frame — กล้องอาจตัดการเชื่อมต่อ")
            break

        if FLIP_CAMERA:
            frame = cv2.flip(frame, 1)

        frame_h, frame_w, _ = frame.shape
        frame_count += 1
        referee.accuracy.tick_frame()

        # ── YOLO: ตรวจจับผู้เล่น (class 0) + บอล (class 32) ──
        results = model.track(
            frame,
            persist=True,
            classes=[0, 32],
            conf=YOLO_CONF,
            verbose=False,
        )

        ball_box     = None
        person_boxes = []
        person_ids   = []

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, pid in zip(boxes, classes, ids):
                if int(cls) == 32:
                    ball_box = box
                    draw_ball(frame, box)
                elif int(cls) == 0:
                    person_boxes.append(box)
                    person_ids.append(pid)

        # ── ด่าน 1: กรอง Box ซ้อนทับ (มือบังหน้า → YOLO นับเป็นคน) ──
        person_boxes, person_ids = filter_duplicate_boxes(
            person_boxes, person_ids, iou_threshold=0.4
        )

        # ── ด่าน 2: กรอง ID กระพริบ (Noise จากมือ/แขน) ──
        valid_ids    = stability.update(person_ids)
        filtered     = [(b, p) for b, p in zip(person_boxes, person_ids)
                        if p in valid_ids]
        person_boxes = [x[0] for x in filtered]
        person_ids   = [x[1] for x in filtered]

        # ── Cleanup: Player ID ที่หายออกจากเฟรมนี้ ──
        # Reset JumpBallDetector Baseline และลบ landmarks cache
        disappeared_ids = prev_valid_ids - set(person_ids)
        for gone_id in disappeared_ids:
            referee.cleanup_player(gone_id)
        prev_valid_ids = set(person_ids)

        # ── Process ผู้เล่นทีละคน ──
        for box, pid in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = map(int, box)

            # ขยาย ROI เล็กน้อยเพื่อให้ MediaPipe มีบริบทรอบข้าง
            margin = 30
            px1 = max(0, x1 - margin)
            py1 = max(0, y1 - margin)
            px2 = min(frame_w, x2 + margin)
            py2 = min(frame_h, y2 + margin)
            roi = frame[py1:py2, px1:px2]

            if roi.size == 0:
                continue

            # ── MediaPipe Pose บน ROI ──
            roi_rgb     = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(roi_rgb)

            if not pose_result.pose_landmarks:
                continue

            # วาดโครงกระดูกบน ROI (แสดงบนภาพหลักด้วย)
            mp_drawing.draw_landmarks(
                roi, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ── แปลงพิกัด: Normalized ROI → Global Pixel ──
            roi_h, roi_w, _ = roi.shape
            landmarks_px    = {}

            for idx, lm in enumerate(pose_result.pose_landmarks.landmark):
                # ขั้น 1: Normalized × ขนาด ROI = Pixel ใน ROI
                cx = int(lm.x * roi_w)
                cy = int(lm.y * roi_h)
                # ขั้น 2: บวก ROI offset = Global Pixel ของภาพหลัก
                landmarks_px[idx] = (cx + px1, cy + py1)

            # ── ด่าน 3: ส่งให้ Referee ตัดสิน ──
            # ส่ง roi (BGR) ด้วยเพื่อให้ ContactFoulDetector (3D CNN) ใช้งานได้
            violations, info_texts = referee.process(
                pid, landmarks_px, mp_pose,
                ball_box, frame_w, frame_h,
                roi_bgr=roi        # ← 3D CNN จะใช้ตรงนี้
            )

            # ── บันทึก Log ถ้ามีฟาวล์ ──
            for v in violations:
                logger.log_foul(pid, v)

            # ── วาด UI ──
            draw_player(frame, box, pid, violations, info_texts)

        # ── HUD ──
        draw_hud(frame, frame_count, len(person_boxes))

        cv2.imshow("AI Basketball Referee", frame)

        # ── Keyboard Controls ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 กด 'q' — กำลังปิดระบบ...")
            break
        elif key == ord('r'):
            referee.accuracy = type(referee.accuracy)()
            print("🔄 Accuracy Tracker ถูก Reset แล้ว")

    # ════════════════════════════════════════════════
    #  Cleanup และรายงานผล
    # ════════════════════════════════════════════════
    cap.release()
    cv2.destroyAllWindows()

    referee.accuracy.print_report()
    print(f"📁 Log บันทึกไว้ที่: {LOG_FILE}")


if __name__ == "__main__":
    main()