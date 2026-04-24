import cv2
import numpy as np
from utils import TemporalVoter # สมมติว่ามีอยู่แล้ว

class Kalman1D:
    """ Wrapper สำหรับ cv2.KalmanFilter แบบ 1 มิติ (แกน Y) """
    def __init__(self):
        self.kf = cv2.KalmanFilter(2, 1)
        # State: [y, dy/dt]
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32) 
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-4, 0], [0, 1e-4]], np.float32)
        self.kf.measurementNoiseCov = np.array([[1e-2]], np.float32)
        self.kf.errorCovPost = np.eye(2, dtype=np.float32)
        self.is_initialized = False

    def update(self, measurement, visibility):
        # 1. Predict (ทายตำแหน่งล่วงหน้าเสมอ)
        prediction = self.kf.predict()
        
        if not self.is_initialized and visibility >= 0.5:
            self.kf.statePost = np.array([[measurement], [0]], np.float32)
            self.is_initialized = True
            return measurement

        # 2. Correct (ถ้ากล้องเห็นชัด ค่อยเอาค่ามาแก้ความคลาดเคลื่อน)
        if visibility >= 0.5:
            meas_array = np.array([[measurement]], np.float32)
            self.kf.correct(meas_array)
            return self.kf.statePost[0, 0]
        else:
            # ถ้ากล้องมองไม่เห็น คืนค่าที่ Kalman เดาไว้ (Prediction)
            return prediction[0, 0]

class TravelingDetector:
    def __init__(self):
        self.steps = 0
        self.kf_l = Kalman1D() # 📌 อัปเกรดเป็น Kalman
        self.kf_r = Kalman1D() # 📌 อัปเกรดเป็น Kalman
        self.voter = TemporalVoter(window_size=5, threshold=4) 
        
        self.prev_l_y = None
        self.prev_r_y = None
        self.is_holding_prev = False
        
    def check(self, pose_landmarks, mp_pose, is_holding, shoulder_width, frame_h):
        lm_l = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        lm_r = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # โยนพิกัด Y และ Visibility เข้า Kalman Filter
        l_y = self.kf_l.update(lm_l.y * frame_h, lm_l.visibility)
        r_y = self.kf_r.update(lm_r.y * frame_h, lm_r.visibility)

        if self.prev_l_y is None:
            self.prev_l_y = l_y
            self.prev_r_y = r_y

        if is_holding and not self.is_holding_prev:
            self.steps = 0 
        
        if is_holding:
            diff_l = abs(l_y - self.prev_l_y)
            diff_r = abs(r_y - self.prev_r_y)
            dynamic_threshold = shoulder_width * 0.15 
            
            if diff_l > dynamic_threshold or diff_r > dynamic_threshold:
                self.steps += 0.1 

        self.prev_l_y = l_y
        self.prev_r_y = r_y
        self.is_holding_prev = is_holding

        is_foul_raw = int(self.steps) > 2
        is_foul_confirmed = self.voter.vote(is_foul_raw)

        if is_foul_confirmed:
            return True, f"TRAVELING ({int(self.steps)})"
        
        return False, f"Steps: {int(self.steps)}"