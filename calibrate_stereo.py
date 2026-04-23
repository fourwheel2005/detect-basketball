import cv2
import numpy as np
import glob
import os

# 📌 เพิ่มฟังก์ชันกรองรูปที่ Error สูงทิ้ง
def filter_bad_images(objpoints, imgpoints_a, imgpoints_b, mtx_a, dist_a, mtx_b, dist_b, threshold=1.0):
    good_obj, good_a, good_b = [], [], []
    print(f"\n🔍 เริ่มกระบวนการคัดกรองรูปภาพที่ Error สูงกว่า {threshold} Pixel...")
    
    for i, (op, ip_a, ip_b) in enumerate(zip(objpoints, imgpoints_a, imgpoints_b)):
        # คำนวณ Error ของกล้อง Mac
        _, rvec_a, tvec_a = cv2.solvePnP(op, ip_a, mtx_a, dist_a)
        proj_a, _ = cv2.projectPoints(op, rvec_a, tvec_a, mtx_a, dist_a)
        err_a = cv2.norm(ip_a, proj_a, cv2.NORM_L2) / len(ip_a)

        # คำนวณ Error ของกล้อง iPhone
        _, rvec_b, tvec_b = cv2.solvePnP(op, ip_b, mtx_b, dist_b)
        proj_b, _ = cv2.projectPoints(op, rvec_b, tvec_b, mtx_b, dist_b)
        err_b = cv2.norm(ip_b, proj_b, cv2.NORM_L2) / len(ip_b)

        # ถ้า Error ของทั้งสองกล้องต่ำกว่าเกณฑ์ ถึงจะเก็บรูปคู่นี้ไว้
        if err_a < threshold and err_b < threshold:
            good_obj.append(op)
            good_a.append(ip_a)
            good_b.append(ip_b)
        else:
            print(f"  ❌ คัดรูปคู่ที่ {i+1} ทิ้ง (Error Mac: {err_a:.2f}, iPhone: {err_b:.2f})")
            
    print(f"✅ คัดกรองเสร็จสิ้น! เหลือรูปที่ผ่านเกณฑ์: {len(good_obj)} คู่ (จากเดิม {len(objpoints)} คู่)\n")
    return good_obj, good_a, good_b


def main():
    print("เริ่มกระบวนการคำนวณ Stereo Calibration...")

    CHECKERBOARD = (8, 5) 
    square_size = 1.4 

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * square_size

    objpoints = [] 
    imgpoints_mac = [] 
    imgpoints_iphone = [] 

    images_mac = sorted(glob.glob('stereo_data/cam_mac/*.png'))
    images_iphone = sorted(glob.glob('stereo_data/cam_iphone/*.png'))

    if len(images_mac) == 0 or len(images_iphone) == 0:
        print("❌ ไม่พบรูปภาพในโฟลเดอร์ กรุณารัน capture_stereo.py เพื่อถ่ายรูปก่อนครับ")
        return

    print(f"พบรูปภาพจำนวน {len(images_mac)} คู่ กำลังค้นหาจุดตัดตารางหมากรุก...")

    valid_pairs = 0
    for img_mac_path, img_iphone_path in zip(images_mac, images_iphone):
        img_mac = cv2.imread(img_mac_path)
        img_iphone = cv2.imread(img_iphone_path)

        gray_mac = cv2.cvtColor(img_mac, cv2.COLOR_BGR2GRAY)
        gray_iphone = cv2.cvtColor(img_iphone, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        ret_mac, corners_mac = cv2.findChessboardCorners(gray_mac, CHECKERBOARD, flags)
        ret_iphone, corners_iphone = cv2.findChessboardCorners(gray_iphone, CHECKERBOARD, flags)

        if ret_mac and ret_iphone:
            objpoints.append(objp)

            corners2_mac = cv2.cornerSubPix(gray_mac, corners_mac, (11, 11), (-1, -1), criteria)
            imgpoints_mac.append(corners2_mac)

            corners2_iphone = cv2.cornerSubPix(gray_iphone, corners_iphone, (11, 11), (-1, -1), criteria)
            imgpoints_iphone.append(corners2_iphone)
            valid_pairs += 1

    print(f"✅ ค้นหาจุดตัดสำเร็จ {valid_pairs} คู่ (จากทั้งหมด {len(images_mac)} คู่)")
    
    if valid_pairs < 10:
        print("⚠️ คำเตือน: จำนวนรูปที่ใช้ได้น้อยเกินไป อาจทำให้ความแม่นยำต่ำ")

    img_size = gray_mac.shape[::-1]

    # 📌 1. คำนวณเลนส์แยกทีละตัว (เพื่อเอาค่า mtx ไปใช้ใน Filter)
    print("กำลังคำนวณหาค่าเลนส์กล้องแต่ละตัว...")
    ret_mac, mtx_mac, dist_mac, _, _ = cv2.calibrateCamera(objpoints, imgpoints_mac, img_size, None, None)
    ret_iphone, mtx_iphone, dist_iphone, _, _ = cv2.calibrateCamera(objpoints, imgpoints_iphone, img_size, None, None)

    # 📌 2. นำรูปมาผ่านระบบคัดกรอง (เรียกใช้ฟังก์ชันตรงนี้)
    objpoints, imgpoints_mac, imgpoints_iphone = filter_bad_images(
        objpoints, imgpoints_mac, imgpoints_iphone,
        mtx_mac, dist_mac, mtx_iphone, dist_iphone,
        threshold=1.0 # ตั้งค่าเกณฑ์ตรงนี้ (ถ้าอยากให้เข้มงวดมาก ปรับเป็น 0.5 ก็ได้)
    )

    # เช็คว่าเหลือรูปรอดมาให้คำนวณกี่รูป
    if len(objpoints) < 5:
        print("❌ รูปผ่านเกณฑ์น้อยเกินไป (เหลือน้อยกว่า 5 คู่) ไม่สามารถคำนวณ 3D ได้ กรุณาถ่ายใหม่ครับ")
        return

    # 📌 3. คำนวณ Stereo Calibration จากรูปที่คัดมาแล้ว
    print("กำลังคำนวณความสัมพันธ์ระหว่างกล้อง 2 ตัว (Stereo Calibration)...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_mac, imgpoints_iphone,
        mtx_mac, dist_mac,
        mtx_iphone, dist_iphone,
        img_size, criteria=criteria, flags=flags
    )

    print("💾 กำลังบันทึกข้อมูลลงไฟล์ 'stereo_params.xml'...")
    cv_file = cv2.FileStorage("stereo_params.xml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("mtx_mac", mtx_mac)
    cv_file.write("dist_mac", dist_mac)
    cv_file.write("mtx_iphone", mtx_iphone)
    cv_file.write("dist_iphone", dist_iphone)
    cv_file.write("R", R)
    cv_file.write("T", T)
    cv_file.release()

    print(f"🎉 เสร็จสมบูรณ์! ค่าความคลาดเคลื่อนรวมหลังคัดกรอง (Reprojection Error): {ret_stereo:.4f}")
    print("ตอนนี้ระบบของคุณพร้อมเข้าสู่มิติ 3D แล้วครับ!")
    
if __name__ == "__main__":
    main()