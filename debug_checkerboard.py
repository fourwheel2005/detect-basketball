import cv2

# สุ่มหยิบรูปจากกล้อง Mac มา 1 รูป (เปลี่ยนชื่อไฟล์เป็นรูปที่คุณคิดว่าถ่ายชัดที่สุด)
img_path = 'stereo_data/cam_mac/image_04.png' 
img = cv2.imread(img_path)

if img is None:
    print("❌ หาไฟล์รูปไม่เจอ เช็คชื่อและ path ของไฟล์อีกทีครับ")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ลองให้ AI หาตารางทั้ง 2 รูปแบบ (8, 5) และ (5, 8)
patterns_to_try = [(8, 5), (5, 8)]
found = False

print("กำลังวิเคราะห์รูปภาพ...")

for pattern in patterns_to_try:
    # เพิ่ม Flag ช่วยลดปัญหาแสงจ้า (Adaptive Thresh)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern, flags)
    
    if ret:
        print(f"✅ รอดแล้ว! AI มองเห็นตารางหมากรุกด้วยขนาด {pattern}")
        # วาดเส้นรุ้งๆ โชว์จุดตัด
        cv2.drawChessboardCorners(img, pattern, corners, ret)
        cv2.imshow('Debug - Checkerboard Found!', img)
        print("กดปุ่มอะไรก็ได้บนคีย์บอร์ดเพื่อปิดหน้าต่าง")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        found = True
        break

if not found:
    print("❌ หาไม่เจอครับ! รูปนี้ใช้ไม่ได้จริงๆ")
    print("👉 สาเหตุที่เป็นไปได้: แสงจ้าบังตาราง, นิ้วบังจุดตัด, กระดาษงอ, หรือถ่ายใกล้/ไกลเกินไป")