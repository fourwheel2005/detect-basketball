# 🏀 AI Basketball Referee System

ระบบตรวจจับการทำฟาวล์ในกีฬาบาสเกตบอลด้วย Computer Vision โดยใช้กล้องเพียงตัวเดียว (Single Camera)  
ผสมผสาน **YOLOv8**, **MediaPipe Pose** และ **3D CNN** เพื่อตัดสินกฎกติกาแบบ Real-time

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-Active_Development-green)
![YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-red)

---

## เปรียบเทียบกับโปรเจกต์ต้นแบบ (vs ayushpai/AI-Basketball-Referee)

| จุด | ต้นแบบ (ayushpai) | ระบบนี้ |
|---|:---:|:---:|
| สถาปัตยกรรม | Monolithic (ไฟล์เดียว) | Modular (แยก class/ไฟล์) |
| รองรับหลายผู้เล่น | ❌ (คนแรกเท่านั้น) | ✅ (Track ID ต่อคน) |
| Pose Estimation | YOLOv8-pose (full frame) | MediaPipe บน ROI ต่อคน |
| Traveling Logic | นับต่อเฟรม (ขึ้น FPS) | Peak Detection (ไม่ขึ้น FPS) |
| Double Dribble | Normalized coords (bug) | Pixel coords ถูกต้อง |
| กรอง Noise มือบังหน้า | ❌ | ✅ (3 ด่าน) |
| Carrying / Goaltending | ❌ | ✅ |
| บันทึก Log CSV | ❌ | ✅ |
| Accuracy Report | ❌ | ✅ |
| 3D CNN Contact Foul | ❌ | ✅ (pipeline พร้อม) |

---

## ฟีเจอร์ทั้งหมด

### Rule Engine (กฎที่ใช้งานได้แล้ว)

| # | กฎ | หลักการตรวจจับ |
|---|---|---|
| 1 | **Double Dribble** | State Machine: IDLE → DRIBBLING → HOLDING → VIOLATION |
| 2 | **Traveling** | Peak Detection บน Y-axis ของ ankle + Rolling Average 5 เฟรม |
| 3 | **Carrying** | เปรียบ Y ของ Wrist vs Index Finger + Confirm Buffer 5 เฟรม |
| 4 | **Goaltending** | วิเคราะห์ Parabolic Trajectory + ตรวจ Peak ผ่านมาแล้ว |

### Noise Filtering (ลด False Positive)

| ด่าน | วิธีการ | แก้ปัญหา |
|---|---|---|
| 1 | IoU Deduplication | YOLO นับมือ/แขนเป็นคนแยก |
| 2 | PlayerStabilityFilter | ID กระพริบวูบๆ จาก Noise |
| 3 | Pose Validity Check | Pose ไม่ครบ / Aspect Ratio ผิดปกติ |

### ระบบสนับสนุน

- **FoulLogger** — บันทึกลง CSV พร้อม Cooldown 3 วินาที
- **AccuracyTracker** — รายงาน Activation Rate ทุก Rule หลัง Session จบ
- **Keyboard Controls** — `q` ออก, `r` Reset Accuracy

---

## โครงสร้างไฟล์

```
DETECT-BASKETBALL/
│
├── main.py                    # Loop หลัก: กล้อง → YOLO → MediaPipe → UI → Log
├── referee.py                 # Orchestrator: รับ Pose+Ball ส่งให้ Detector ทุกตัว
├── utils.py                   # Helper functions + Classes ทั้งหมด
│
├── fouls/                     # Rule Engine — แยกไฟล์ต่อกฎ
│   ├── __init__.py
│   ├── double_dribble.py      # State Machine: IDLE/DRIBBLING/HOLDING/VIOLATION
│   ├── traveling.py           # Peak Detection + Rolling Average
│   ├── carrying.py            # Wrist vs Index Finger Y comparison
│   ├── goaltending.py         # Parabolic Trajectory Analysis
│   │
│   ├── frame_buffer.py        # [3D CNN] เก็บ 16 เฟรมต่อเนื่องต่อผู้เล่น
│   ├── contact_3dcnn_model.py # [3D CNN] โหลดและรัน Model
│   └── contact_foul.py        # [3D CNN] Detector ที่เชื่อม Buffer + Model
│
└── trainmodel/
    ├── train_basketball.py    # [Custom YOLO] Train Basketball Detection Model
    ├── train_3dcnn.py         # [3D CNN] Train Contact Foul Model
    ├── basketball_model.pt    # Custom YOLO weights (หลัง Train)
    ├── contact_model.h5       # 3D CNN weights (หลัง Train)
    └── dataset/
        ├── normal/            # คลิปปกติ ไม่มีฟาวล์
        ├── arm_hit/           # คลิปตีแขน
        └── body_contact/      # คลิปชนตัว
```

---

## การติดตั้ง

### ความต้องการของระบบ

- Python **3.11** (แนะนำ — เสถียรที่สุดสำหรับ MediaPipe)
- Webcam หรือกล้อง USB
- GPU (ไม่บังคับ แต่ช่วยให้เร็วขึ้นมาก)

### ขั้นตอนติดตั้ง

```bash
# 1. สร้าง Virtual Environment
python3.11 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 2. ติดตั้ง Library
pip install numpy opencv-python mediapipe ultralytics
```

> หากจะใช้ **3D CNN** ให้ติดตั้งเพิ่ม:
> ```bash
> pip install tensorflow scikit-learn
> ```

---

## วิธีรัน

```bash
python main.py
```

| ปุ่ม | ฟังก์ชัน |
|---|---|
| `q` | ออกจากโปรแกรม (พิมพ์ Accuracy Report อัตโนมัติ) |
| `r` | Reset Accuracy Tracker |

---

## อธิบายการทำงานละเอียด

### 1. main.py — Loop หลัก

```
กล้อง → Frame
  ↓
YOLO track()           ← detect คน (class 0) + บอล (class 32)
  ↓
filter_duplicate_boxes ← ด่าน 1: ตัด Box ซ้อนทับ (IoU > 0.4)
  ↓
PlayerStabilityFilter  ← ด่าน 2: ตัด ID ที่ปรากฏ < 8 เฟรม
  ↓
[ต่อผู้เล่นแต่ละคน]
  crop ROI → MediaPipe Pose
  แปลง Normalized → Global Pixel coords
  ↓
referee.process()      ← ด่าน 3: ตรวจ Pose ครบ + ตัดสินกฎ
  ↓
FoulLogger.log_foul()  ← บันทึก CSV ถ้ามีฟาวล์
  ↓
draw UI + HUD
```

### 2. referee.py — Orchestrator

- สร้าง Detector แยกต่อ Player ID → State ไม่ปนกัน
- `_is_pose_valid()` — ตรวจ Landmark ครบ + Aspect Ratio ก่อนตัดสิน
- `_check_holding()` — ใช้ `.value` เพื่อ convert Enum → int key ให้ถูกต้อง

### 3. Traveling (fouls/traveling.py)

**ปัญหาเดิม (ayushpai):** นับทุกเฟรมที่เท้าขยับ → ผลต่างตาม FPS  
**วิธีใหม่:** Peak Detection

```
เฟรมที่ 1: เท้า Y = 400 (กำลังลง, direction = +1)
เฟรมที่ 2: เท้า Y = 420 (ยังลง)
เฟรมที่ 3: เท้า Y = 395 (กลับขึ้น, direction = -1)
             ↑ direction เปลี่ยนจาก +1 → -1 = นับ 1 ก้าว
```

Rolling Average 5 เฟรมก่อนเช็ค → กรอง Jitter จากมือสั่น

### 4. Double Dribble (fouls/double_dribble.py)

**บั๊กต้นแบบ:** ใช้ Normalized coords (0–1) เทียบกับ Pixel → ระยะห่างผิด  
**แก้ไข:** ใช้ `landmarks_px` ที่แปลงเป็น Pixel แล้วทั้งหมด

State Machine:
```
IDLE ──touch_one──→ DRIBBLING
DRIBBLING ──touch_two──→ HOLDING
HOLDING ──touch_one──→ VIOLATION ← Double Dribble!
any ──no touch (45 เฟรม)──→ IDLE  (Auto Reset)
```

### 5. Accuracy Report (หลังกด q)

```
════════════════════════════════════════════════════════════
  📊  AI BASKETBALL REFEREE — SESSION REPORT
════════════════════════════════════════════════════════════
  Duration : 45.2 sec  (1356 frames @ 30.0 FPS)
  Rules Monitored: 4
────────────────────────────────────────────────────────────
  Rule                   Checks  Detections    Rate  MaxStreak
────────────────────────────────────────────────────────────
  CARRYING                  450          12    2.7%          8
  DOUBLE_DRIBBLE            450           3    0.7%          1
  GOALTENDING               450           0    0.0%          0
  TRAVELING                 450          27    6.0%         15
════════════════════════════════════════════════════════════
```

> **Activation Rate** สูงผิดปกติ = ค่า Threshold ต้องจูนเพิ่ม

---

## 3D CNN — ตรวจจับฟาวล์ประเภทสัมผัส (Contact Foul)

Rule Engine ตรวจจับฟาวล์ที่วัดได้จากท่าทาง (Pose) ได้ดี แต่ฟาวล์ประเภท "ตีแขน" หรือ "ชนตัว" ต้องการการวิเคราะห์ **การเคลื่อนไหวต่อเนื่องในเวลา** ซึ่ง 3D CNN เหมาะกว่า

### หลักการ

```
[16 เฟรมต่อเนื่อง] → [3D CNN] → [normal / arm_hit / body_contact]
  shape: (1, 16, 64, 64, 3)     confidence score ต่อ class
```

### Pipeline ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `fouls/frame_buffer.py` | เก็บ ROI ย้อนหลัง 16 เฟรม, Resize เป็น 64×64 |
| `fouls/contact_3dcnn_model.py` | โหลด `.h5` และรัน predict (threshold 0.65) |
| `fouls/contact_foul.py` | เชื่อม Buffer + Model, ทำนายทุก 16 เฟรม |
| `trainmodel/train_3dcnn.py` | สอน Model จาก Dataset วิดีโอ |

### เตรียม Dataset

```
trainmodel/dataset/
    normal/          ← คลิปปกติ ไม่มีฟาวล์  (50+ คลิป)
    arm_hit/         ← คลิปตีแขน             (50+ คลิป)
    body_contact/    ← คลิปชนตัว             (50+ คลิป)
```

แต่ละคลิปควรยาว ~1 วินาที (16–30 เฟรม) ไฟล์ `.mp4` หรือ `.avi`

### Train 3D CNN

```bash
python trainmodel/train_3dcnn.py
# Model จะถูกบันทึกเป็น trainmodel/contact_model.h5
```

### เปิดใช้งานใน referee.py

```python
# เพิ่ม import
from fouls.contact_foul import ContactFoulDetector

# ใน _get_detectors() — เพิ่ม "cf"
self._players[player_id] = {
    "dd": DoubleDribbleDetector(),
    "tr": TravelingDetector(),
    "ca": CarryingDetector(),
    "gt": GoaltendingDetector(),
    "cf": ContactFoulDetector(),   # ← เพิ่มบรรทัดนี้
}

# ใน process() — เพิ่ม Rule ที่ 5
is_cf, msg_cf = detectors["cf"].check(roi_bgr)
self.accuracy.record("CONTACT_FOUL", is_cf)
if is_cf:
    violations.append(msg_cf)
```

### เพิ่ม roi_bgr ใน main.py

```python
# referee.process() — เพิ่ม parameter
violations, info_texts = referee.process(
    pid, landmarks_px, mp_pose,
    ball_box, frame_w, frame_h,
    roi_bgr=roi   # ← เพิ่ม
)
```

---

## Custom Basketball Detection Model

Train model ตรวจบอลบาสเองแทน YOLO class 32 เพื่อความแม่นยำสูงขึ้น

### เป้าหมาย mAP

| วิธี | Dataset | mAP50 (ประมาณ) |
|---|---|---|
| ayushpai ต้นแบบ | 3,000 ภาพ | ~70% |
| YOLOv8n + Augment | 10,000 ภาพ | ~82% |
| YOLOv8s + Augment | 10,000 ภาพ | ~87% |
| Ensemble 2 Model | 10,000×2 ภาพ | ~91% |

### แหล่ง Dataset (ฟรี)

- [Roboflow Universe — basketball-detection](https://universe.roboflow.com/roboflow-100/basketball-players-fy4c2)
- [Roboflow Universe — ball-detection](https://universe.roboflow.com/roboflow-100/ball-detection-gqjb3)

### Train

```bash
python trainmodel/train_basketball.py
# ใช้เวลา ~2–4 ชั่วโมง บน GPU
# Model จะถูกบันทึกเป็น trainmodel/basketball_v1/weights/best.pt
```

### ใช้ Model ที่ Train แล้วใน main.py

```python
# เปลี่ยนจาก model เดียว
person_model = YOLO("yolov8n.pt")
ball_model   = YOLO("trainmodel/basketball_v1/weights/best.pt")

person_results = person_model.track(frame, persist=True, classes=[0], ...)
ball_results   = ball_model(frame, conf=0.5, verbose=False)
```

---

## การปรับแต่ง Threshold

แก้ค่าในไฟล์แต่ละ Rule ตามสภาพแวดล้อมของกล้อง:

| ไฟล์ | ค่า | ความหมาย |
|---|---|---|
| `traveling.py` | `STEP_THRESH = 12` | ระยะ Y (px) ขั้นต่ำที่นับเป็นก้าว |
| `traveling.py` | `MAX_STEPS = 2` | ก้าวสูงสุดที่อนุญาต |
| `carrying.py` | `Y_BUFFER = 20` | ระยะ Y ที่ข้อมือต้องต่ำกว่านิ้ว |
| `carrying.py` | `CONFIRM_FRAMES = 5` | ต้องเกิดต่อเนื่องกี่เฟรม |
| `double_dribble.py` | `HOLD_THRESHOLD = 110` | ระยะ (px) ที่ถือว่าจับบอล |
| `double_dribble.py` | `TIMEOUT_FRAMES = 45` | เฟรมที่ไม่แตะบอล → Reset |
| `referee.py` | `HOLDING_DIST = 120` | ระยะ (px) ที่ถือว่าถือบอล |
| `utils.py` | `MIN_FRAMES = 8` | เฟรมขั้นต่ำก่อน Accept Player ID |
| `main.py` | `YOLO_CONF = 0.35` | Confidence ขั้นต่ำของ YOLO |
| `main.py` | `LOG_COOLDOWN = 3.0` | วินาที — ป้องกัน Log ซ้ำรัวๆ |

---

## ข้อจำกัดที่ทราบ

- **2D only** — ไม่มีข้อมูล Depth ระยะมือ-บอลอาจคลาดเคลื่อนหากอยู่คนละระนาบ
- **Occlusion** — บอลหรือมือที่ถูกบังชั่วคราวอาจทำให้ State Machine Reset
- **Goaltending** — เป็น Simulation โดยใช้ครึ่งจอเป็น "ระดับห่วง" ต้อง Calibrate ตามมุมกล้องจริง
- **3D CNN** — ต้องการ Dataset วิดีโอก่อนจึงจะใช้งานได้ ไม่มี pretrained ให้ใช้ทันที
- **Custom Ball Model** — ต้องการ Dataset ภาพและเวลา Train ก่อนใช้งาน

---

## Roadmap

- [x] Rule Engine: Traveling, Double Dribble, Carrying, Goaltending
- [x] Multi-player support ด้วย Track ID
- [x] Noise Filtering 3 ด่าน
- [x] FoulLogger CSV + AccuracyTracker
- [x] 3D CNN Pipeline (frame_buffer, model wrapper, detector)
- [ ] Train Custom Basketball Detection Model
- [ ] เก็บ Dataset วิดีโอและ Train 3D CNN
- [ ] Ensemble Ball Detection (Custom + YOLOv8 class 32)
- [ ] Web Dashboard แสดง Foul Log แบบ Real-time