# fouls/carrying.py
class CarryingDetector:
    def check(self, landmarks_px, mp_pose, is_holding):
        if not is_holding: return False, ""

        r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST]
        r_index = landmarks_px[mp_pose.PoseLandmark.RIGHT_INDEX]
        l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST]
        l_index = landmarks_px[mp_pose.PoseLandmark.LEFT_INDEX]

        # 📌 แก้ไข: ใช้ [1] เพื่อดึงค่า Y มาคำนวณ
        # Logic: ข้อมืออยู่ต่ำกว่านิ้ว (แกน Y มากกว่า = ต่ำกว่า) บวก Buffer 20 pixel
        r_carry = r_wrist[1] > (r_index[1] + 20)
        l_carry = l_wrist[1] > (l_index[1] + 20)

        if r_carry or l_carry:
            return True, "CARRYING"
        return False, ""