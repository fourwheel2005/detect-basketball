import cv2
import numpy as np
import mediapipe as mp

# 1. โหลดพารามิเตอร์ 3D ที่เราเพิ่งทำเสร็จ
print("กำลังโหลดสมอง 3 มิติ (stereo_params.xml)...")
cv_file = cv2.FileStorage("stereo_params.xml", cv2.FILE_STORAGE_READ)
mtx_mac = cv_file.getNode("mtx_mac").mat()
mtx_iphone = cv_file.getNode("mtx_iphone").mat()
R = cv_file.getNode("R").mat()
T = cv_file.getNode("T").mat()
cv_file.release()

# สร้าง Projection Matrix (สมการยิงแสงจากกล้อง)
P1 = np.dot(mtx_mac, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(mtx_iphone, np.hstack((R, T)))

def get_3d_coordinate(pt_mac, pt_iphone):
    """แปลงพิกัด 2D จากสองกล้อง ให้เป็นพิกัด 3D โลกจริง"""
    pt1 = np.array([[pt_mac[0]], [pt_mac[1]]], dtype=np.float32)
    pt2 = np.array([[pt_iphone[0]], [pt_iphone[1]]], dtype=np.float32)
    
    # คณิตศาสตร์ Triangulation (ยิงเส้นตรง 2 เส้นไปตัดกัน)
    pt_4d = cv2.triangulatePoints(P1, P2, pt1, pt2)
    pt_3d = pt_4d[:3] / pt_4d[3] # ปรับให้เป็นพิกัด 3D ปกติ
    return pt_3d.flatten()

# 2. ตั้งค่า MediaPipe (ตรวจจับมือ)
mp_hands = mp.solutions.hands
hands_mac = mp_hands.Hands(min_detection_confidence=0.7)
hands_iphone = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# 3. เปิดกล้องคู่
cap_mac = cv2.VideoCapture(0)
cap_iphone = cv2.VideoCapture(1)

print("✅ ระบบพร้อม! ลองชูนิ้วชี้ไปหน้ากล้องดูครับ (กด 'q' เพื่อออก)")

while True:
    ret1, frame1 = cap_mac.read()
    ret2, frame2 = cap_iphone.read()
    if not ret1 or not ret2: break

    # แปลงสีให้ MediaPipe อ่าน
    rgb_mac = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    rgb_iphone = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    res_mac = hands_mac.process(rgb_mac)
    res_iphone = hands_iphone.process(rgb_iphone)

    pt_mac = None
    pt_iphone = None
    h, w, _ = frame1.shape

    # หาปลายนิ้วชี้จากกล้อง Mac
    if res_mac.multi_hand_landmarks:
        for hand_landmarks in res_mac.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            idx_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pt_mac = (int(idx_finger.x * w), int(idx_finger.y * h))
            cv2.circle(frame1, pt_mac, 10, (0, 255, 0), cv2.FILLED)

    # หาปลายนิ้วชี้จากกล้อง iPhone
    if res_iphone.multi_hand_landmarks:
        for hand_landmarks in res_iphone.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            idx_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            pt_iphone = (int(idx_finger.x * w), int(idx_finger.y * h))
            cv2.circle(frame2, pt_iphone, 10, (0, 255, 255), cv2.FILLED)

    # ถ้าเห็นนิ้วชี้ทั้ง 2 กล้อง -> คำนวณระยะ 3D เลย!
    if pt_mac and pt_iphone:
        pos_3d = get_3d_coordinate(pt_mac, pt_iphone)
        
        # pos_3d[0] = แกน X (ซ้าย-ขวา)
        # pos_3d[1] = แกน Y (บน-ล่าง)
        # pos_3d[2] = แกน Z (ระยะความลึกจากหน้ากล้อง)
        
        depth_cm = pos_3d[2] # ค่าจะออกมาเป็นหน่วยเดียวกับที่คุณตั้ง square_size ไว้ (เซนติเมตร)
        
        # แสดงผลความลึกบนหน้าจอ
        cv2.putText(frame1, f"Depth (Z): {depth_cm:.1f} cm", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # แสดงภาพ 2 จอต่อกัน
    frame1_resized = cv2.resize(frame1, (640, 480))
    frame2_resized = cv2.resize(frame2, (640, 480))
    cv2.imshow("3D Depth Test", cv2.hconcat([frame1_resized, frame2_resized]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_mac.release()
cap_iphone.release()
cv2.destroyAllWindows()