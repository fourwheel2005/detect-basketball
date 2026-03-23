"""
utils.py — ฟังก์ชันและคลาสช่วยเหลือสำหรับระบบ AI Basketball Referee
-----------------------------------------------------------------------
ประกอบด้วย:
  - get_dist()              : คำนวณระยะห่างระหว่าง 2 จุด
  - calculate_angle()       : คำนวณมุมจาก 3 จุด
  - compute_iou()           : คำนวณ Intersection over Union
  - filter_duplicate_boxes(): กรอง Box ซ้อนทับ (ลด False Person Detection)
  - PlayerStabilityFilter   : กรอง Player ID ที่กระพริบ (Noise)
  - FoulLogger              : บันทึกฟาวล์ลง CSV พร้อม Cooldown
  - AccuracyTracker         : ติดตามและรายงานความแม่นยำหลัง Run
"""

import numpy as np
import time
import csv
import os
from collections import defaultdict


# ─────────────────────────────────────────────────────
#  ฟังก์ชันคณิตศาสตร์พื้นฐาน
# ─────────────────────────────────────────────────────

def get_dist(p1, p2) -> float:
    """
    คำนวณระยะทาง Euclidean ระหว่างจุด 2 จุด (หน่วย: Pixel)

    Parameters:
        p1, p2: tuple หรือ array-like ที่มี (x, y)

    Returns:
        float: ระยะห่าง
    """
    return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))


def calculate_angle(a, b, c) -> float:
    """
    คำนวณมุมที่จุด b โดยมี a–b–c เป็น 3 จุดอ้างอิง

    Parameters:
        a, b, c: tuple (x, y) ของแต่ละข้อต่อ

    Returns:
        float: มุมในหน่วยองศา (0–180)
    """
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = (np.arctan2(c[1] - b[1], c[0] - b[0])
               - np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle = abs(np.degrees(radians))
    return 360.0 - angle if angle > 180.0 else angle


# ─────────────────────────────────────────────────────
#  IoU และ Box Deduplication — แก้ YOLO นับมือเป็นคน
# ─────────────────────────────────────────────────────

def compute_iou(boxA, boxB) -> float:
    """
    คำนวณ Intersection over Union (IoU) ของ 2 Bounding Box

    Parameters:
        boxA, boxB: array-like [x1, y1, x2, y2]

    Returns:
        float: ค่า IoU (0.0 – 1.0)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def filter_duplicate_boxes(boxes: list, ids: list,
                            iou_threshold: float = 0.4) -> tuple:
    """
    กรอง Bounding Box ที่ซ้อนทับกันเกิน iou_threshold ออก
    เก็บเฉพาะ Box ที่ใหญ่กว่า (= ตัวจริง ไม่ใช่มือ/แขนที่ลอยอยู่)

    Parameters:
        boxes         : list ของ [x1, y1, x2, y2]
        ids           : list ของ Player ID (ตำแหน่งตรงกับ boxes)
        iou_threshold : ค่า IoU ที่เกินแล้วถือว่า "ซ้อนกัน" (default 0.4)

    Returns:
        (filtered_boxes, filtered_ids)
    """
    if len(boxes) == 0:
        return boxes, ids

    # เรียงลำดับจาก Box ใหญ่ → เล็ก (ตาม Area)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    order = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)

    keep       = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        # Box เล็กกว่าที่ซ้อนทับเกิน Threshold → ตัดทิ้ง
        for j in order:
            if j == i or j in suppressed:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(j)

    return [boxes[i] for i in keep], [ids[i] for i in keep]


# ─────────────────────────────────────────────────────
#  PlayerStabilityFilter — กรอง ID กระพริบ
# ─────────────────────────────────────────────────────

class PlayerStabilityFilter:
    """
    กรอง Player ID ที่ "กระพริบ" (ปรากฏแล้วหายเร็วมาก)

    หลักการ:
        - ID จริง (คนจริง)   → ปรากฏต่อเนื่องหลายเฟรม
        - ID Noise (มือ/แขน) → ปรากฏแค่ 1–5 เฟรมแล้วหาย

    Attributes:
        MIN_FRAMES    : ต้องเห็น ID ต่อเนื่องกี่เฟรมถึง "ยืนยัน"
        EXPIRE_FRAMES : ไม่เห็น ID กี่เฟรมถึงถือว่าออกไปแล้ว
    """

    MIN_FRAMES    = 8   # เฟรมขั้นต่ำก่อน Accept
    EXPIRE_FRAMES = 30  # เฟรมที่ไม่เห็นแล้วถือว่าออกไป

    def __init__(self):
        self._seen      : dict = {}   # {id: จำนวนเฟรมที่เห็น}
        self._last_seen : dict = {}   # {id: เฟรมล่าสุดที่เห็น}
        self._active    : set  = set() # ID ที่ผ่านการยืนยัน
        self._frame     : int  = 0

    def update(self, current_ids: list) -> set:
        """
        อัปเดต Filter ด้วย ID ที่เห็นในเฟรมนี้

        Parameters:
            current_ids: list ของ ID ที่ YOLO ตรวจพบในเฟรมนี้

        Returns:
            set ของ ID ที่ "ผ่านการยืนยัน" และอยู่ใน current_ids ด้วย
        """
        self._frame += 1

        for pid in current_ids:
            self._seen[pid]      = self._seen.get(pid, 0) + 1
            self._last_seen[pid] = self._frame
            if self._seen[pid] >= self.MIN_FRAMES:
                self._active.add(pid)

        # ลบ ID ที่หายไปนานเกิน EXPIRE_FRAMES
        stale = [pid for pid, last in self._last_seen.items()
                 if self._frame - last > self.EXPIRE_FRAMES]
        for pid in stale:
            self._active.discard(pid)
            self._seen.pop(pid, None)
            self._last_seen.pop(pid, None)

        # Return เฉพาะ ID ที่ยืนยันแล้ว AND เห็นอยู่ในเฟรมนี้
        return self._active & set(current_ids)


# ─────────────────────────────────────────────────────
#  FoulLogger — บันทึกเหตุการณ์ฟาวล์ลง CSV
# ─────────────────────────────────────────────────────

class FoulLogger:
    """
    บันทึกการตรวจจับฟาวล์ลงไฟล์ CSV พร้อมระบบ Cooldown
    เพื่อป้องกันการบันทึกซ้ำในเฟรมที่ติดกัน

    Usage:
        logger = FoulLogger("logs.csv", cooldown_sec=3.0)
        logger.log_foul(player_id=1, foul_type="TRAVELING")
    """

    CSV_HEADERS = ["Date_Time", "Player_ID", "Foul_Type"]

    def __init__(self, filename: str = "basketball_foul_logs.csv",
                 cooldown_sec: float = 3.0):
        self.filename     = filename
        self.cooldown_sec = cooldown_sec
        self._last_logged : dict = {}  # key = "playerID_foulType"

        # สร้างไฟล์ใหม่พร้อม Header ถ้ายังไม่มี
        if not os.path.exists(self.filename):
            self._write_row(self.CSV_HEADERS, mode='w')

    def log_foul(self, player_id: int, foul_type: str) -> bool:
        """
        บันทึกฟาวล์ถ้าพ้น Cooldown แล้ว

        Returns:
            bool: True = บันทึกสำเร็จ, False = ยังอยู่ใน Cooldown
        """
        now = time.time()
        key = f"{player_id}_{foul_type}"

        if key in self._last_logged and (now - self._last_logged[key]) < self.cooldown_sec:
            return False

        self._last_logged[key] = now
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        self._write_row([timestamp, f"Player_{player_id}", foul_type])
        print(f"📝 [LOG] {timestamp} | Player {player_id} | {foul_type}")
        return True

    def _write_row(self, row: list, mode: str = 'a'):
        """เขียน 1 แถวลงไฟล์ CSV"""
        with open(self.filename, mode=mode, newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)


# ─────────────────────────────────────────────────────
#  AccuracyTracker — ติดตามและรายงานความแม่นยำ
# ─────────────────────────────────────────────────────

class AccuracyTracker:
    """
    ติดตามสถิติการตรวจจับของแต่ละประเภทฟาวล์
    และพิมพ์รายงานสรุปเมื่อจบ Session

    Usage:
        tracker = AccuracyTracker()
        tracker.record("TRAVELING", detected=True)
        tracker.tick_frame()
        tracker.print_report()
    """

    def __init__(self):
        self._stats : dict = defaultdict(
            lambda: {"total": 0, "detected": 0, "consecutive": 0, "max_consecutive": 0}
        )
        self._session_start = time.time()
        self._total_frames  = 0

    def record(self, foul_type: str, detected: bool):
        """
        บันทึกผลการตรวจ 1 เฟรมสำหรับ foul_type ที่ระบุ

        Parameters:
            foul_type : ชื่อฟาวล์ เช่น "TRAVELING"
            detected  : True = ตรวจพบในเฟรมนี้
        """
        s = self._stats[foul_type]
        s["total"] += 1
        if detected:
            s["detected"]        += 1
            s["consecutive"]     += 1
            s["max_consecutive"]  = max(s["max_consecutive"], s["consecutive"])
        else:
            s["consecutive"] = 0

    def tick_frame(self):
        """เรียกทุกเฟรมหลัก เพื่อนับ FPS และความยาว Session"""
        self._total_frames += 1

    def print_report(self):
        """พิมพ์รายงานสรุปสถิติเมื่อจบ Session"""
        elapsed = time.time() - self._session_start
        fps     = self._total_frames / elapsed if elapsed > 0 else 0

        print("\n" + "═" * 60)
        print("  📊  AI BASKETBALL REFEREE — SESSION REPORT")
        print("═" * 60)
        print(f"  Duration : {elapsed:.1f} sec  "
              f"({self._total_frames} frames @ {fps:.1f} FPS)")
        print(f"  Rules Monitored: {len(self._stats)}")
        print("─" * 60)
        print(f"  {'Rule':<22} {'Checks':>7} {'Detections':>11} "
              f"{'Rate':>7} {'MaxStreak':>10}")
        print("─" * 60)

        for foul_type, s in sorted(self._stats.items()):
            total  = s["total"]
            det    = s["detected"]
            rate   = (det / total * 100) if total > 0 else 0.0
            streak = s["max_consecutive"]
            print(f"  {foul_type:<22} {total:>7} {det:>11} "
                  f"{rate:>6.1f}%  {streak:>9}")

        print("═" * 60 + "\n")