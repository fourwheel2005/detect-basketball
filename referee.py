# referee.py
import numpy as np
from utils import get_dist
from fouls.double_dribble import DoubleDribbleDetector
from fouls.traveling import TravelingDetector
from fouls.carrying import CarryingDetector
from fouls.goaltending import GoaltendingDetector

class BasketballRef:
    def __init__(self):
        self.players = {}

    def get_detectors(self, p_id):
        if p_id not in self.players:
            self.players[p_id] = {
                "dd": DoubleDribbleDetector(),
                "tr": TravelingDetector(),
                "ca": CarryingDetector(),
                "gt": GoaltendingDetector(),
            }
        return self.players[p_id]

    def process(self, p_id, landmarks_px, mp_pose, ball_box, frame_w, frame_h):
        # landmarks_px: Dictionary {LandmarkID: (x, y)} เป็น Pixel จริงๆ
        detectors = self.get_detectors(p_id)
        violations = []
        info_texts = []

        ball_center = None
        if ball_box is not None:
            bx1, by1, bx2, by2 = ball_box
            ball_center = ((bx1+bx2)/2, (by1+by2)/2)

        # 1. Check Holding (ครองบอล)
        is_holding = False
        if ball_center is not None:
            r_wrist = landmarks_px[mp_pose.PoseLandmark.RIGHT_WRIST]
            l_wrist = landmarks_px[mp_pose.PoseLandmark.LEFT_WRIST]
            
            # ระยะถือบอล 120 px
            if get_dist(r_wrist, ball_center) < 120 or get_dist(l_wrist, ball_center) < 120:
                is_holding = True

        # --- Check Rules ---
        
        # Double Dribble
        is_dd, msg_dd = detectors["dd"].check(landmarks_px, mp_pose, ball_center)
        if is_dd: violations.append(msg_dd)

        # Traveling (ส่ง landmarks_px ไปเลย)
        is_tr, msg_tr = detectors["tr"].check(landmarks_px, mp_pose, is_holding)
        if is_tr: violations.append(msg_tr)
        elif "Steps" in msg_tr: info_texts.append(msg_tr)

        # Carrying (ส่ง landmarks_px ไปเลย)
        is_ca, msg_ca = detectors["ca"].check(landmarks_px, mp_pose, is_holding)
        if is_ca: violations.append(msg_ca)

        # Goaltending
        is_gt, msg_gt = detectors["gt"].check(ball_center, frame_h)
        if is_gt and is_holding:
            violations.append(msg_gt)

        return violations, info_texts