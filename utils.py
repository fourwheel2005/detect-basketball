"""
utils.py — ฟังก์ชันและคลาสช่วยเหลือ
--------------------------------------
ประกอบด้วย:
  - get_dist()              : ระยะห่าง Euclidean
  - calculate_angle()       : มุมจาก 3 จุด
  - compute_iou()           : Intersection over Union
  - filter_duplicate_boxes(): กรอง Box ซ้อนทับ
  - filter_mirror_boxes()   : กรอง เงาในกระจก/สะท้อน  ← ใหม่
  - PlayerStabilityFilter   : กรอง ID กระพริบ
  - FoulLogger              : บันทึก CSV
  - AccuracyTracker         : รายงานความแม่นยำ
"""

import numpy as np
import time
import csv
import os
from collections import defaultdict


# ─────────────────────────────────────────────────────
#  คณิตศาสตร์พื้นฐาน
# ─────────────────────────────────────────────────────

def get_dist(p1, p2) -> float:
    """ระยะทาง Euclidean ระหว่าง 2 จุด (pixel)"""
    return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))


def calculate_angle(a, b, c) -> float:
    """
    มุมที่จุด b (องศา 0–180)
    a, b, c: tuple (x, y) ของข้อต่อแต่ละจุด
    """
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = (np.arctan2(c[1]-b[1], c[0]-b[0])
               - np.arctan2(a[1]-b[1], a[0]-b[0]))
    angle = abs(np.degrees(radians))
    return 360.0 - angle if angle > 180.0 else angle


# ─────────────────────────────────────────────────────
#  IoU และ Box Filtering
# ─────────────────────────────────────────────────────

def compute_iou(boxA, boxB) -> float:
    """IoU ของ 2 Bounding Box [x1, y1, x2, y2]"""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


def filter_duplicate_boxes(boxes: list, ids: list,
                            iou_threshold: float = 0.4) -> tuple:
    """
    กรอง Box ซ้อนทับกัน เก็บ Box ใหญ่ไว้ (ตัวจริง)
    ตัด Box เล็กที่ซ้อนอยู่ทิ้ง (มือ/แขนลอย)
    """
    if len(boxes) == 0:
        return boxes, ids

    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    keep = [];  suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in suppressed:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)

    return [boxes[i] for i in keep], [ids[i] for i in keep]


def filter_mirror_boxes(boxes: list, ids: list,
                        frame_w: int,
                        size_ratio_thresh: float = 0.55,
                        brightness_thresh: float = 30.0,
                        frame_bgr=None) -> tuple:
    """
    กรองเงาในกระจก/ภาพสะท้อนออกจาก Person Detection

    หลักการ 3 ชั้น:
        1. Size Ratio  — เงาในกระจกมักมีขนาดเล็กกว่าตัวจริงมาก
                         Box ที่มี Area < size_ratio_thresh × Area ใหญ่สุด → ตัดทิ้ง
                         (ใช้ได้ดีเมื่อกระจกอยู่ข้าง/หลัง)

        2. Edge Bias   — กระจกมักอยู่ขอบซ้าย/ขวา
                         Box ที่ center_x อยู่ใน 15% ขอบซ้ายขวา
                         และมีขนาดเล็กกว่า 40% ของ Box ใหญ่สุด → ตัดทิ้ง

        3. Brightness  — เงาในกระจกมักมืดกว่าหรือ contrast ต่างกัน
                         (เฉพาะเมื่อส่ง frame_bgr มาด้วย)

    Parameters:
        boxes            : list ของ [x1,y1,x2,y2]
        ids              : list ของ Player ID
        frame_w          : ความกว้างของเฟรม (pixel)
        size_ratio_thresh: Box ที่เล็กกว่า ratio นี้ × Box ใหญ่สุด → ตัดทิ้ง
        brightness_thresh: ค่าต่างสว่าง (Mean BGR) ที่ถือว่า "ต่างกันมาก"
        frame_bgr        : numpy array ของ frame ต้นฉบับ (optional)

    Returns:
        (filtered_boxes, filtered_ids)
    """
    if len(boxes) <= 1:
        return boxes, ids  # คนเดียวไม่มีเงา

    areas    = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    max_area = max(areas)
    keep     = []

    for i, (box, area) in enumerate(zip(boxes, areas)):
        cx = (box[0] + box[2]) / 2  # Center X ของ Box

        # ── ชั้น 1: Size Ratio ──────────────────────────
        if max_area > 0 and (area / max_area) < size_ratio_thresh:
            # Box เล็กกว่า 55% ของ Box ใหญ่สุด → น่าจะเป็นเงา
            continue

        # ── ชั้น 2: Edge Bias ────────────────────────────
        edge_zone = frame_w * 0.15  # 15% ขอบซ้ายขวา
        is_edge   = (cx < edge_zone) or (cx > frame_w - edge_zone)
        is_small  = (area / max_area) < 0.40

        if is_edge and is_small:
            # อยู่ขอบและเล็กกว่า 40% → น่าจะเป็นเงาในกระจกข้างห้อง
            continue

        # ── ชั้น 3: Brightness (optional) ────────────────
        if frame_bgr is not None and brightness_thresh > 0:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)
            roi = frame_bgr[y1:y2, x1:x2]

            if roi.size > 0:
                # เปรียบสว่างเฉลี่ยกับ Box ที่ใหญ่ที่สุด
                big_idx = areas.index(max_area)
                bx1, by1, bx2, by2 = map(int, boxes[big_idx])
                bx1 = max(0, bx1);  by1 = max(0, by1)
                bx2 = min(frame_bgr.shape[1], bx2)
                by2 = min(frame_bgr.shape[0], by2)
                big_roi = frame_bgr[by1:by2, bx1:bx2]

                if big_roi.size > 0 and i != big_idx:
                    bright_diff = abs(float(roi.mean()) -
                                      float(big_roi.mean()))
                    if bright_diff > brightness_thresh and is_small:
                        continue  # สว่างต่างกันมากและเล็กกว่า → เงา

        keep.append(i)

    if len(keep) == 0:
        # ถ้ากรองหมดให้เก็บ Box ใหญ่สุดไว้
        keep = [areas.index(max_area)]

    return [boxes[i] for i in keep], [ids[i] for i in keep]


# ─────────────────────────────────────────────────────
#  PlayerStabilityFilter — กรอง ID กระพริบ
# ─────────────────────────────────────────────────────

class PlayerStabilityFilter:
    """
    กรอง Player ID ที่ปรากฏแล้วหายเร็ว (Noise)
    ID จริงปรากฏต่อเนื่อง ≥ MIN_FRAMES จึง Accept
    """

    MIN_FRAMES    = 8    # เฟรมขั้นต่ำก่อน Accept
    EXPIRE_FRAMES = 30   # ไม่เห็นกี่เฟรมถึงถือว่าออกไป

    def __init__(self):
        self._seen      : dict = {}
        self._last_seen : dict = {}
        self._active    : set  = set()
        self._frame     : int  = 0

    def update(self, current_ids: list) -> set:
        """
        อัปเดตด้วย ID ที่เห็นในเฟรมนี้
        Returns: set ของ ID ที่ "ยืนยันแล้ว" และอยู่ใน current_ids
        """
        self._frame += 1

        for pid in current_ids:
            self._seen[pid]      = self._seen.get(pid, 0) + 1
            self._last_seen[pid] = self._frame
            if self._seen[pid] >= self.MIN_FRAMES:
                self._active.add(pid)

        stale = [pid for pid, last in self._last_seen.items()
                 if self._frame - last > self.EXPIRE_FRAMES]
        for pid in stale:
            self._active.discard(pid)
            self._seen.pop(pid, None)
            self._last_seen.pop(pid, None)

        return self._active & set(current_ids)


# ─────────────────────────────────────────────────────
#  FoulLogger — บันทึก CSV
# ─────────────────────────────────────────────────────

class FoulLogger:
    """
    บันทึกฟาวล์ลง CSV พร้อม Cooldown ป้องกัน Log ซ้ำรัวๆ

    Usage:
        logger = FoulLogger("logs.csv", cooldown_sec=3.0)
        logger.log_foul(1, "TRAVELING")
    """

    CSV_HEADERS = ["Date_Time", "Player_ID", "Foul_Type"]

    def __init__(self, filename: str = "basketball_foul_logs.csv",
                 cooldown_sec: float = 3.0):
        self.filename     = filename
        self.cooldown_sec = cooldown_sec
        self._last_logged : dict = {}

        if not os.path.exists(self.filename):
            self._write_row(self.CSV_HEADERS, mode='w')

    def log_foul(self, player_id: int, foul_type: str) -> bool:
        """บันทึกถ้าพ้น Cooldown"""
        now = time.time()
        # ใช้ foul type ไม่เกิน 20 ตัวอักษรแรกเป็น key
        # กันกรณี velocity ต่างกันทุกเฟรมทำให้ key ต่างกันทุกเฟรม
        foul_key = foul_type[:20].strip()
        key = f"{player_id}_{foul_key}"

        if (key in self._last_logged and
                (now - self._last_logged[key]) < self.cooldown_sec):
            return False

        self._last_logged[key] = now
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        self._write_row([ts, f"Player_{player_id}", foul_type])
        print(f"📝 [LOG] {ts} | Player {player_id} | {foul_type}")
        return True

    def _write_row(self, row: list, mode: str = 'a'):
        with open(self.filename, mode=mode, newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)


# ─────────────────────────────────────────────────────
#  AccuracyTracker — รายงานความแม่นยำ
# ─────────────────────────────────────────────────────

class AccuracyTracker:
    """
    ติดตาม Activation Rate ของแต่ละ Rule
    พิมพ์รายงานสรุปเมื่อ Session จบ
    """

    def __init__(self):
        self._stats : dict = defaultdict(
            lambda: {"total": 0, "detected": 0,
                     "consecutive": 0, "max_consecutive": 0}
        )
        self._session_start = time.time()
        self._total_frames  = 0

    def record(self, foul_type: str, detected: bool):
        s = self._stats[foul_type]
        s["total"] += 1
        if detected:
            s["detected"]        += 1
            s["consecutive"]     += 1
            s["max_consecutive"]  = max(s["max_consecutive"],
                                        s["consecutive"])
        else:
            s["consecutive"] = 0

    def tick_frame(self):
        self._total_frames += 1

    def print_report(self):
        elapsed = time.time() - self._session_start
        fps     = self._total_frames / elapsed if elapsed > 0 else 0

        print("\n" + "═"*60)
        print("  📊  AI BASKETBALL REFEREE — SESSION REPORT")
        print("═"*60)
        print(f"  Duration : {elapsed:.1f}s  "
              f"({self._total_frames} frames @ {fps:.1f} FPS)")
        print("─"*60)
        print(f"  {'Rule':<22} {'Checks':>7} {'Hits':>7} "
              f"{'Rate':>7} {'MaxStreak':>10}")
        print("─"*60)

        for foul_type, s in sorted(self._stats.items()):
            total  = s["total"]
            det    = s["detected"]
            rate   = (det/total*100) if total > 0 else 0.0
            streak = s["max_consecutive"]
            print(f"  {foul_type:<22} {total:>7} {det:>7} "
                  f"{rate:>6.1f}%  {streak:>9}")

        print("═"*60 + "\n")