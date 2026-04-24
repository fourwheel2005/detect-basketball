import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import multiprocessing as mp_lib
import time
import random

# ==========================================
# 🚀 1. The Consumer: 3D CNN Worker (รันอยู่หลังบ้าน)
# ==========================================
def cnn_worker(input_queue, result_queue):
    print("🔥 3D CNN Worker Started! (Running in Dummy Mode)")
    
    # 📝 TODO: เมื่อเทรนโมเดลเสร็จ ให้ Uncomment บรรทัดนี้
    # import onnxruntime as ort
    # session = ort.InferenceSession("trainmodel/contact_model.onnx", providers=['CoreMLExecutionProvider'])
    
    while True:
        frames_buffer = input_queue.get() 
        if frames_buffer is None: # สัญญาณสั่งปิด
            break
            
        # ⏱️ จำลองเวลา Inference ของ AI (0.05 วินาที)
        time.sleep(0.05)
        
        # 🎲 จำลองผลลัพธ์ (สุ่ม) เพื่อให้คุณเทส UI ไปพลางๆ
        # 📝 TODO: ของจริงต้องรัน session.run() ตรงนี้
        is_foul = random.choice([False, False, False, True]) # สุ่มฟาวล์ 25%
        
        result_queue.put({"is_foul": is_foul, "time": 0.05})

# ==========================================
# 📸 2. The Producer: ระบบหลัก
# ==========================================
def main():
    # สร้างท่อส่งข้อมูลระหว่าง Process
    input_queue = mp_lib.Queue()
    result_queue = mp_lib.Queue()
    
    # สตาร์ท Thread ของ 3D CNN
    cnn_process = mp_lib.Process(target=cnn_worker, args=(input_queue, result_queue))
    cnn_process.start()

    print("🤖 โหลด YOLO Nano (Track Mode) และ MediaPipe...")
    yolo_model = YOLO("yolov8n.pt") 
    mp_pose = mp.solutions.pose
    pose_mac = mp_pose.Pose(min_detection_confidence=0.7)

    cap_mac = cv2.VideoCapture(0)
    
    frame_buffer = []
    latest_contact_foul = False
    foul_display_timer = 0
    prev_time = time.time()

    print("✅ ระบบพร้อมทำงาน! กด 'q' เพื่อออก")

    while cap_mac.isOpened():
        ret, frame = cap_mac.read()
        if not ret: break
        
        # ---------------------------------------------------------
        # 📌 1. YOLOv8n Tracking Mode (ป้องกันกล่องกะพริบ)
        # ---------------------------------------------------------
        results_yolo = yolo_model.track(
            frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=[0, 32], 
            verbose=False
        )
        
        # สมมติว่ามี Logic กรองและ Crop ผู้เล่นที่ปะทะกัน (คุณต้องเขียนเพิ่ม)
        # เพื่อความเรียบง่าย เราจะโยนเฟรมเต็มเข้าไปเก็บใน Buffer (ของจริงต้อง Crop 112x112)
        frame_buffer.append(cv2.resize(frame, (112, 112)))
        
        # ---------------------------------------------------------
        # 📌 2. ส่งข้อมูลให้ 3D CNN ผ่าน Queue
        # ---------------------------------------------------------
        if len(frame_buffer) == 16:
            input_queue.put(frame_buffer) # โยนแล้วลืมไปเลย ไม่รอ!
            frame_buffer = [] # รีเซ็ต Buffer
            
        # ---------------------------------------------------------
        # 📌 3. เช็คผลลัพธ์จาก 3D CNN (Non-Blocking)
        # ---------------------------------------------------------
        if not result_queue.empty():
            res = result_queue.get()
            if res["is_foul"]:
                latest_contact_foul = True
                foul_display_timer = 30 # แสดงแจ้งเตือนค้างไว้ 30 เฟรม

        # จัดการ UI แจ้งเตือน
        if latest_contact_foul and foul_display_timer > 0:
            cv2.putText(frame, "CONTACT FOUL DETECTED!", (50, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            foul_display_timer -= 1
        else:
            latest_contact_foul = False

        # ---------------------------------------------------------
        # 📌 แสดงผล FPS
        # ---------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("🏀 AI Referee (Mac Camera)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิด Process คืน RAM ให้ระบบ
    input_queue.put(None) 
    cnn_process.join()
    cap_mac.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()