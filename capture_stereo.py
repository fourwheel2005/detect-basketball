import cv2
import os

def create_folders():
    # สร้างโฟลเดอร์เก็บภาพถ้ายังไม่มี
    if not os.path.exists("stereo_data/cam_mac"):
        os.makedirs("stereo_data/cam_mac")
    if not os.path.exists("stereo_data/cam_iphone"):
        os.makedirs("stereo_data/cam_iphone")

def main():
    create_folders()
    
    # เปิดกล้อง (อิงจากที่คุณเทสต์ผ่าน)
    cap_mac = cv2.VideoCapture(0)
    cap_iphone = cv2.VideoCapture(1)

    if not cap_mac.isOpened() or not cap_iphone.isOpened():
        print("❌ เปิดกล้องไม่สำเร็จ ตรวจสอบการเชื่อมต่ออีกครั้ง")
        return

    print("✅ ระบบตากล้องพร้อมทำงาน!")
    print("👉 วิธีใช้: ถือกระดาษตารางหมากรุกให้กล้องทั้ง 2 ตัวเห็นชัดเจน แล้วกด 'Spacebar' เพื่อถ่ายรูป")
    print("👉 ควรถ่ายสัก 20-30 รูป โดยหมุนกระดาษ เอียงซ้าย-ขวา-หน้า-หลัง ให้หลากหลายมุม")
    print("👉 กด 'q' เพื่อออก")

    img_counter = 0

    while True:
        ret1, frame1 = cap_mac.read()
        ret2, frame2 = cap_iphone.read()

        if not ret1 or not ret2:
            print("อ่านภาพไม่สำเร็จ...")
            break

        # ย่อภาพให้ดูง่ายๆ บนหน้าจอ
        show1 = cv2.resize(frame1, (640, 480))
        show2 = cv2.resize(frame2, (640, 480))
        
        cv2.putText(show1, "Mac Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(show2, "iPhone Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        combined_frame = cv2.hconcat([show1, show2])
        cv2.imshow("Stereo Camera Capture", combined_frame)

        k = cv2.waitKey(1)
        
        # กด 'q' เพื่อออก
        if k % 256 == ord('q'):
            print("ปิดโปรแกรม...")
            break
        
        # กด 'Spacebar' (รหัส 32) เพื่อถ่ายและบันทึกรูป
        elif k % 256 == 32:
            img_name_mac = f"stereo_data/cam_mac/image_{img_counter:02d}.png"
            img_name_iphone = f"stereo_data/cam_iphone/image_{img_counter:02d}.png"
            
            # บันทึกภาพต้นฉบับ (frame เต็ม) ลงเครื่อง
            cv2.imwrite(img_name_mac, frame1)
            cv2.imwrite(img_name_iphone, frame2)
            
            print(f"📸 แชะ! บันทึกภาพคู่ที่ {img_counter} เรียบร้อยแล้ว!")
            img_counter += 1

    cap_mac.release()
    cap_iphone.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()