import cv2
import numpy as np
import glob
import os

def main():
    print("เริ่มกระบวนการคำนวณ Stereo Calibration...")

    # 📌 การตั้งค่ากระดานหมากรุก (สำคัญมากต้องตั้งให้ตรงกับกระดาษที่ปริ้นท์)
    # เราใช้ตารางแบบ 9x6 จุดตัดด้านใน (มุมที่เรามองเห็น)
    CHECKERBOARD = (9, 6) 
    
    # ขนาดของช่องสี่เหลี่ยม 1 ช่องบนกระดาษ (หน่วยเป็นเซนติเมตร)
    # **พรุ่งนี้ตอนปริ้นท์เสร็จ ให้เอาไม้บรรทัดวัดดูว่า 1 ช่องกว้างกี่เซนติเมตร แล้วมาแก้เลขตรงนี้นะครับ**
    square_size = 2.5 

    # เตรียมตัวแปรเก็บจุดพิกัด
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # สร้างพิกัด 3D จำลองของกระดานหมากรุก (เช่น (0,0,0), (2.5,0,0), (5.0,0,0) ...)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = [] # เก็บพิกัด 3D โลกจริง
    imgpoints_mac = [] # เก็บพิกัด 2D บนจอ Mac
    imgpoints_iphone = [] # เก็บพิกัด 2D บนจอ iPhone

    # ดึงรายชื่อไฟล์รูปภาพทั้งหมดที่ถ่ายไว้
    images_mac = sorted(glob.glob('stereo_data/cam_mac/*.png'))
    images_iphone = sorted(glob.glob('stereo_data/cam_iphone/*.png'))

    if len(images_mac) == 0 or len(images_iphone) == 0:
        print("❌ ไม่พบรูปภาพในโฟลเดอร์ กรุณารัน capture_stereo.py เพื่อถ่ายรูปก่อนครับ")
        return

    print(f"พบรูปภาพจำนวน {len(images_mac)} คู่ กำลังค้นหาจุดตัดตารางหมากรุก...")

    # วนลูปอ่านรูปทีละคู่
    valid_pairs = 0
    for img_mac_path, img_iphone_path in zip(images_mac, images_iphone):
        img_mac = cv2.imread(img_mac_path)
        img_iphone = cv2.imread(img_iphone_path)

        gray_mac = cv2.cvtColor(img_mac, cv2.COLOR_BGR2GRAY)
        gray_iphone = cv2.cvtColor(img_iphone, cv2.COLOR_BGR2GRAY)

        # ค้นหาจุดตัดตารางหมากรุก
        ret_mac, corners_mac = cv2.findChessboardCorners(gray_mac, CHECKERBOARD, None)
        ret_iphone, corners_iphone = cv2.findChessboardCorners(gray_iphone, CHECKERBOARD, None)

        # ถ้าเจอตารางชัดเจนทั้ง 2 กล้อง ถึงจะเอามาใช้คำนวณ
        if ret_mac and ret_iphone:
            objpoints.append(objp)

            # ขัดเกลาพิกัดให้แม่นยำระดับทศนิยม (Subpixel)
            corners2_mac = cv2.cornerSubPix(gray_mac, corners_mac, (11, 11), (-1, -1), criteria)
            imgpoints_mac.append(corners2_mac)

            corners2_iphone = cv2.cornerSubPix(gray_iphone, corners_iphone, (11, 11), (-1, -1), criteria)
            imgpoints_iphone.append(corners2_iphone)
            valid_pairs += 1

    print(f"✅ ค้นหาจุดตัดสำเร็จ {valid_pairs} คู่ (จากทั้งหมด {len(images_mac)} คู่)")
    
    if valid_pairs < 10:
        print("⚠️ คำเตือน: จำนวนรูปที่ใช้ได้น้อยเกินไป (ควรมีอย่างน้อย 10-15 คู่) อาจทำให้ความแม่นยำต่ำ")

    # ขนาดภาพ
    img_size = gray_mac.shape[::-1]

    print("กำลังคำนวณหาค่าเลนส์กล้องแต่ละตัว...")
    ret_mac, mtx_mac, dist_mac, _, _ = cv2.calibrateCamera(objpoints, imgpoints_mac, img_size, None, None)
    ret_iphone, mtx_iphone, dist_iphone, _, _ = cv2.calibrateCamera(objpoints, imgpoints_iphone, img_size, None, None)

    print("กำลังคำนวณความสัมพันธ์ระหว่างกล้อง 2 ตัว (Stereo Calibration)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_mac, imgpoints_iphone,
        mtx_mac, dist_mac,
        mtx_iphone, dist_iphone,
        img_size, criteria=criteria, flags=flags
    )

    # R = Rotation Matrix (การหมุน/เอียงของกล้อง iPhone เทียบกับ Mac)
    # T = Translation Vector (ระยะห่างระหว่างกล้อง iPhone กับ Mac)

    # บันทึกค่าทั้งหมดลงไฟล์ XML เพื่อเอาไปใช้ตอนจับฟาวล์จริง
    print("💾 กำลังบันทึกข้อมูลลงไฟล์ 'stereo_params.xml'...")
    cv_file = cv2.FileStorage("stereo_params.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("mtx_mac", mtx_mac)
    cv_file.write("dist_mac", dist_mac)
    cv_file.write("mtx_iphone", mtx_iphone)
    cv_file.write("dist_iphone", dist_iphone)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.release()

    print(f"🎉 เสร็จสมบูรณ์! ค่าความคลาดเคลื่อน (Reprojection Error): {ret_stereo:.4f}")
    print("ตอนนี้ระบบของคุณพร้อมเข้าสู่มิติ 3D แล้วครับ!")

if __name__ == "__main__":
    main()