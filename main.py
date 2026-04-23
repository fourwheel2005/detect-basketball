import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from threading import Thread
import time

# ==========================================
# 1. คลาสจัดการกล้องแบบ Multithreading (เพิ่ม FPS)
# ==========================================
class CameraStream:
    def __init__(self, src=0, name="Camera"):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.name = name
        self.stopped = False
        
        # เริ่ม Thread แยกต่างหากเพื่ออ่านภาพตลอดเวลา
        Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                break
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ==========================================
# 2. ระบบคำนวณสมอง 3 มิติ
# ==========================================
def load_stereo_params(filepath="stereo_params.xml"):
    print("🧠 กำลังโหลดพารามิเตอร์ 3D...")
    try:
        cv_file = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        mtx_mac = cv_file.getNode("mtx_mac").mat()
        mtx_iphone = cv_file.getNode("mtx_iphone").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        cv_file.release()

        P1 = np.dot(mtx_mac, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(mtx_iphone, np.hstack((R, T)))
        return P1, P2
    except Exception as e:
        print(f"❌ โหลดไฟล์ 3D ไม่สำเร็จ (เช็คไฟล์ stereo_params.xml): {e}")
        return None, None

def get_3d_coordinate(pt_mac, pt_iphone, P1, P2):
    """แปลงจุด 2D สองจุด ให้กลายเป็น 3D (X, Y, Z)"""
    pt1 = np.array([[pt_mac[0]], [pt_mac[1]]], dtype=np.float32)
    pt2 = np.array([[pt_iphone[0]], [pt_iphone[1]]], dtype=np.float32)
    
    pt_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.flatten() # คืนค่า [X, Y, Z]

# ==========================================
# 3. โค้ดรันระบบหลัก
# ==========================================
def main():
    # โหลดสมการยิงแสง 3 มิติ
    P1, P2 = load_stereo_params()
    if P1 is None: return

    # โหลด AI Models
    print("🤖 กำลังโหลด YOLO และ MediaPipe...")
    # TODO: แก้ไข path ไฟล์ pt ให้ตรงกับในเครื่องของคุณ
    yolo_model = YOLO("yolov8n.pt") 
    
    mp_pose = mp.solutions.pose
    pose_mac = mp_pose.Pose(min_detection_confidence=0.7)
    pose_iphone = mp_pose.Pose(min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # เปิดกล้องคู่แบบแยก Thread
    print("🎥 กำลังเปิดกล้องคู่...")
    cam_mac = CameraStream(src=0, name="Mac")
    cam_iphone = CameraStream(src=1, name="iPhone") # ลองเปลี่ยน index ถ้าหา iPhone ไม่เจอ

    # ตัวแปรสำหรับคำนวณ FPS
    prev_time = time.time()

    print("✅ ระบบพร้อมทำงาน! กด 'q' เพื่อออก")

    while True:
        ret_mac, frame_mac = cam_mac.read()
        ret_iphone, frame_iphone = cam_iphone.read()

        if not ret_mac or not ret_iphone:
            continue

        h, w, _ = frame_mac.shape

        # ---------------------------------------------------------
        # AI งานที่ 1: YOLO จับลูกบาส (ทำเฉพาะกล้อง Mac ตามแผนที่ 3A)
        # ---------------------------------------------------------
        ball_pos_2d = None
        results_yolo = yolo_model(frame_mac, classes=[32], verbose=False) # class 32 = sports ball
        for box in results_yolo[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            ball_pos_2d = (int((x1+x2)/2), int((y1+y2)/2)) # จุดศูนย์กลางลูกบาส
            cv2.rectangle(frame_mac, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(frame_mac, "Ball", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # ---------------------------------------------------------
        # AI งานที่ 2: MediaPipe จับกระดูกคน (ทำทั้ง 2 กล้อง)
        # ---------------------------------------------------------
        rgb_mac = cv2.cvtColor(frame_mac, cv2.COLOR_BGR2RGB)
        rgb_iphone = cv2.cvtColor(frame_iphone, cv2.COLOR_BGR2RGB)

        res_mac = pose_mac.process(rgb_mac)
        res_iphone = pose_iphone.process(rgb_iphone)

        # ---------------------------------------------------------
        # AI งานที่ 3: รวมร่าง 3D
        # ---------------------------------------------------------
        player_3d_data = {} # เก็บพิกัด 3D ของข้อต่อที่สำคัญ

        if res_mac.pose_landmarks and res_iphone.pose_landmarks:
            mp_draw.draw_landmarks(frame_mac, res_mac.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame_iphone, res_iphone.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ดึงจุดกระดูกมาเข้าสมการ (ยกตัวอย่าง: ดึงส้นเท้าขวา - RIGHT_HEEL)
            idx_heel = mp_pose.PoseLandmark.RIGHT_HEEL.value
            
            lm_mac = res_mac.pose_landmarks.landmark[idx_heel]
            lm_iphone = res_iphone.pose_landmarks.landmark[idx_heel]

            # ตรวจสอบว่ากล้องเห็นส้นเท้าทั้ง 2 มุมไหม
            if lm_mac.visibility > 0.5 and lm_iphone.visibility > 0.5:
                pt_mac = (int(lm_mac.x * w), int(lm_mac.y * h))
                pt_iphone = (int(lm_iphone.x * w), int(lm_iphone.y * h))
                
                # ✨ วินาทีเวทมนตร์: แปลงเป็น 3D ✨
                pos_3d = get_3d_coordinate(pt_mac, pt_iphone, P1, P2)
                player_3d_data['right_heel'] = pos_3d

                # โชว์ความลึกแกน Z ของส้นเท้าบนจอ Mac
                depth_z = pos_3d[2]
                cv2.circle(frame_mac, pt_mac, 8, (0, 255, 0), -1)
                cv2.putText(frame_mac, f"Z: {depth_z:.1f} cm", (pt_mac[0] + 10, pt_mac[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---------------------------------------------------------
        # แสดงผลภาพ และ FPS
        # ---------------------------------------------------------
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(frame_mac, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # ย่อและต่อภาพเพื่อแสดงผลจอเดียว
        show_mac = cv2.resize(frame_mac, (640, 480))
        show_iphone = cv2.resize(frame_iphone, (640, 480))
        combined = cv2.hconcat([show_mac, show_iphone])
        
        cv2.imshow("🏀 Fourwheel 3D Referee System", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิดการทำงาน
    cam_mac.stop()
    cam_iphone.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()