"""
main.py — AI Basketball Referee (Final Clean Version)
======================================================
ระบบ AI ตรวจจับฟาวล์บาสเกตบอลแบบ Real-time ด้วย:
    - YOLOv8         : ตรวจจับผู้เล่นและลูกบอล
    - MediaPipe Pose : วิเคราะห์ท่าทางร่างกาย
    - Rule Engine    : ตรวจสอบกฎ Traveling / Carrying /
                       Double Dribble / Goaltending

ฟีเจอร์:
    ✅ บันทึก Log ลง CSV
    ✅ แสดง Accuracy Report หลัง Session จบ
    ✅ กรอง Noise มือบังหน้า ด้วย IoU + Stability Filter
    ✅ โค้ดสะอาด อ่านง่าย Comment ทุกจุด

Controls:
    q  — ออกจากโปรแกรม
    r  — รีเซ็ต Accuracy Tracker
"""

import cv2
import mediapipe as mp
from ultralytics import YOLO

from referee import BasketballRef
from utils   import (FoulLogger, PlayerStabilityFilter,
                     filter_duplicate_boxes)


# ════════════════════════════════════════════════════
#  Config — ปรับค่าได้ที่นี่ที่เดียว
# ════════════════════════════════════════════════════

CAMERA_INDEX  = 0      # 0 = Webcam หลัก
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
YOLO_CONF     = 0.35   # ความมั่นใจขั้นต่ำของ YOLO
LOG_FILE      = "basketball_foul_logs.csv"
LOG_COOLDOWN  = 3.0    # วินาที — ป้องกัน Log ซ้ำรัวๆ
FLIP_CAMERA   = True   # True = กล้องหน้า (Mirror)


# ════════════════════════════════════════════════════
#  ฟังก์ชัน Draw UI
# ════════════════════════════════════════════════════

def draw_player(frame, box, player_id, violations, info_texts):
    """
    วาด UI สำหรับผู้เล่น 1 คน: กรอบ, ข้อมูล, ฟาวล์

    Parameters:
        frame      : frame หลัก (BGR)
        box        : (x1, y1, x2, y2) bounding box ของผู้เล่น
        player_id  : int
        violations : list[str] — ฟาวล์ที่ตรวจพบ
        info_texts : list[str] — ข้อมูล Debug เช่น Steps
    """
    x1, y1, x2, y2 = map(int, box)

    # สี: เขียว = ปกติ, แดง = ฟาวล์
    color = (0, 0, 255) if violations else (0, 220, 80)

    # กรอบผู้เล่น
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Player ID Badge
    cv2.putText(frame, f"P{player_id}", (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Info Texts (Steps, State, ฯลฯ) — แสดงเหนือกรอบ
    label_y = y1 - 10
    for info in info_texts:
        # กรอง [SKIP] ออก ไม่แสดงบนหน้าจอ (ดูใน Terminal แทน)
        if info.startswith("[SKIP]"):
            continue
        cv2.putText(frame, info, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 230, 50), 2)
        label_y -= 22

    # Violation Banner — แสดงถ้ามีฟาวล์
    if violations:
        text = " + ".join(violations)
        font        = cv2.FONT_HERSHEY_SIMPLEX
        scale, thick = 0.8, 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

        # พื้นหลังกึ่งโปร่งใส
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
    """แสดง HUD มุมซ้ายบน"""
    cv2.putText(frame, f"Frame: {frame_count}  Players: {num_players}",
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
    stability = PlayerStabilityFilter()   # กรอง ID กระพริบ (สร้างครั้งเดียวนอก Loop)

    # ── เปิดกล้อง ──
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ ไม่สามารถเปิดกล้องได้ ตรวจสอบ CAMERA_INDEX")
        return

    print("✅ System Ready! กด 'q' เพื่อออก, 'r' เพื่อ Reset Accuracy")
    frame_count = 0

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

        # ── YOLO: ตรวจจับผู้เล่น (class 0) และบอล (class 32) ──
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
                if int(cls) == 32:      # ลูกบอล
                    ball_box = box
                    draw_ball(frame, box)
                elif int(cls) == 0:     # ผู้เล่น
                    person_boxes.append(box)
                    person_ids.append(pid)

        # ── ด่านที่ 1: กรอง Box ที่ซ้อนทับกัน (มือบังหน้า) ──
        person_boxes, person_ids = filter_duplicate_boxes(
            person_boxes, person_ids, iou_threshold=0.4
        )

        # ── ด่านที่ 2: กรอง ID ที่กระพริบ (Noise จากมือ/แขน) ──
        valid_ids    = stability.update(person_ids)
        filtered     = [(b, p) for b, p in zip(person_boxes, person_ids)
                        if p in valid_ids]
        person_boxes = [x[0] for x in filtered]
        person_ids   = [x[1] for x in filtered]

        # ── Process ผู้เล่นทีละคน ──
        for box, pid in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = map(int, box)
            margin = 30  # ขยาย ROI เล็กน้อยเพื่อให้ Pose มีข้อมูลรอบข้าง

            # กำหนดขอบเขต ROI ไม่ให้เกินขอบ Frame
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

            # วาดโครงกระดูกบน ROI
            mp_drawing.draw_landmarks(roi, pose_result.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

            # ── แปลงพิกัด Landmark: Normalized ROI → Global Pixel ──
            roi_h, roi_w, _ = roi.shape
            landmarks_px    = {}

            for idx, lm in enumerate(pose_result.pose_landmarks.landmark):
                # Normalized (0–1) × ขนาด ROI = Pixel ใน ROI
                cx = int(lm.x * roi_w)
                cy = int(lm.y * roi_h)
                # บวก Offset ของ ROI = Pixel ในภาพหลัก
                landmarks_px[idx] = (cx + px1, cy + py1)

            # ── ด่านที่ 3: ส่งให้ Referee ตัดสิน (_is_pose_valid อยู่ใน referee.py) ──
            violations, info_texts = referee.process(
                pid, landmarks_px, mp_pose,
                ball_box, frame_w, frame_h
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

    # 📊 พิมพ์รายงานความแม่นยำหลัง Session จบ
    referee.accuracy.print_report()
    print(f"📁 Log ถูกบันทึกไว้ที่: {LOG_FILE}")


if __name__ == "__main__":
    main()