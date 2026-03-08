import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from referee import BasketballRef

# 📌 1. นำเข้า FoulLogger จากไฟล์ utils
from utils import FoulLogger 

def main():
    # 1. Setup
    print("Loading AI Models...")
    model = YOLO('yolov8n.pt') 
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    
    referee = BasketballRef()
    
    # 📌 2. เปิดใช้งานระบบ Logger 
    # สร้างตัวแปร logger เพื่อเตรียมเขียนไฟล์บันทึก โดยตั้งหน่วงเวลา (Cooldown) 3 วินาทีเพื่อไม่ให้บันทึกซ้ำรัวๆ
    print("Initializing Foul Logger...")
    logger = FoulLogger(filename="basketball_foul_logs.csv", cooldown_sec=3.0)
    
    # เปิดกล้อง (0 = Webcam)
    cap = cv2.VideoCapture(0) 
    cap.set(3, 1280) 
    cap.set(4, 720)

    print("Starting System... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # กลับด้านภาพ (ถ้าใช้กล้องหน้า)
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 2. YOLO Detection
        # Class 0=Person, 32=Sports ball
        results = model.track(frame, persist=True, classes=[0, 32], conf=0.3, verbose=False)
        
        ball_box = None
        person_boxes = []
        person_ids = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls, pid in zip(boxes, classes, ids):
                if int(cls) == 32: # Ball
                    ball_box = box
                    bx1, by1, bx2, by2 = map(int, box)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                    cv2.putText(frame, "Ball", (bx1, by1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                elif int(cls) == 0: # Person
                    person_boxes.append(box)
                    person_ids.append(pid)

        # 3. Process Each Person
        for box, pid in zip(person_boxes, person_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Crop ROI (ตัดภาพเฉพาะคน)
            m = 30 # Margin
            px1, py1 = max(0, x1-m), max(0, y1-m)
            px2, py2 = min(w, x2+m), min(h, y2+m)
            roi = frame[py1:py2, px1:px2]
            
            if roi.size == 0: continue
            
            # Pose Estimation บน ROI
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(roi_rgb)

            if pose_results.pose_landmarks:
                # วาดเส้นกระดูกบน ROI (จะไปโผล่ใน Frame หลัก)
                mp_drawing.draw_landmarks(roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # *** CRITICAL FIX: Convert ROI Coords to Global Pixel Coords ***
                # สร้าง Dict เก็บพิกัดจริงเพื่อส่งให้ Referee
                landmarks_px = {}
                roi_h, roi_w, _ = roi.shape
                
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    # 1. แปลง Normalized ของ ROI -> Pixel ของ ROI
                    cx, cy = int(lm.x * roi_w), int(lm.y * roi_h)
                    # 2. บวก Offset ของ ROI -> Pixel ของภาพใหญ่
                    global_x, global_y = cx + px1, cy + py1
                    
                    landmarks_px[idx] = (global_x, global_y)

                # 4. เรียก Referee ตัดสิน
                violations, infos = referee.process(pid, landmarks_px, mp_pose, ball_box, w, h)
                
                # 5. แสดงผล (UI)
                color = (0, 255, 0) # เขียว (ปกติ)
                
                # แสดงข้อมูล (Steps)
                label_y = y1 - 10
                for info in infos:
                    cv2.putText(frame, info, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    label_y -= 25

                # แสดงฟาวล์ (Violations)
                if violations:
                    color = (0, 0, 255) # แดง (ฟาวล์)
                    text = " & ".join(violations)
                    
                    # 📌 3. สั่งบันทึกลง Log (CSV)
                    # วนลูปเผื่อกรณีผู้เล่น 1 คน ทำฟาวล์ 2 อย่างพร้อมกัน
                    for v in violations:
                        logger.log_foul(pid, v)
                    
                    # พื้นหลังตัวหนังสือ
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 40), (x1 + tw, y1 - 15), (0,0,0), -1)
                    cv2.putText(frame, text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # วาดกรอบรอบตัวคน
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.imshow('AI Basketball Referee (Final)', frame)
        
        # *** FIX: Break อยู่ใน Loop ถูกต้องแล้ว ***
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()