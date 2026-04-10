import cv2

def test_dual_cameras():
    print("🔄 กำลังค้นหากล้อง...")
    
    # 📌 Index 0 มักจะเป็นกล้องหน้าของ Mac (FaceTime HD)
    cap_mac = cv2.VideoCapture(0)
    
    # 📌 Index 1 มักจะเป็นกล้อง iPhone (Continuity Camera) 
    # **หมายเหตุ:** ถ้าภาพ iPhone ไม่ขึ้น ให้ลองเปลี่ยนเลข 1 เป็น 2 หรือ 3 ดูนะครับ
    cap_iphone = cv2.VideoCapture(1) 

    if not cap_mac.isOpened():
        print("❌ ไม่พบกล้อง Mac (Index 0)")
        return
    if not cap_iphone.isOpened():
        print("❌ ไม่พบกล้อง iPhone (Index 1) - ลองเปลี่ยนเลข Index ดูครับ")
        return

    print("✅ เปิดกล้องคู่สำเร็จ! กดตัว 'q' เพื่อออกจากหน้าจอ")

    while True:
        # อ่านเฟรมภาพจากกล้องทั้ง 2 ตัว
        ret1, frame1 = cap_mac.read()
        ret2, frame2 = cap_iphone.read()

        if ret1 and ret2:
            # ปรับขนาดภาพให้เท่ากัน (กว้าง 640 x สูง 480) เพื่อให้เอามาต่อกันได้
            frame1_resized = cv2.resize(frame1, (640, 480))
            frame2_resized = cv2.resize(frame2, (640, 480))

            # เขียนข้อความบอกว่ากล้องไหนเป็นกล้องไหน
            cv2.putText(frame1_resized, "Mac Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame2_resized, "iPhone Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # นำภาพมาต่อกันแนวนอน (Horizontal Concatenate)
            combined_frame = cv2.hconcat([frame1_resized, frame2_resized])

            # แสดงผล
            cv2.imshow("Dual Camera Test (Mac + iPhone)", combined_frame)

        # กด 'q' เพื่อออก
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ปิดการเชื่อมต่อ
    cap_mac.release()
    cap_iphone.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_dual_cameras()