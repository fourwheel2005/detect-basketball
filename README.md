# 🏀 AI Basketball Referee System (ระบบกรรมการบาสเกตบอล AI)

ระบบตรวจจับการทำฟาวล์ในกีฬาบาสเกตบอลด้วย Computer Vision โดยใช้กล้องเพียงตัวเดียว (Single Camera) ระบบผสมผสานระหว่าง **YOLOv8** (สำหรับการตรวจจับคนและลูกบอล) และ **MediaPipe** (สำหรับการวิเคราะห์ท่าทางกระดูก) เพื่อตัดสินกฎกติกาต่างๆ

![Project Status](https://img.shields.io/badge/Status-Educational_Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

## 🆕 อัปเดตล่าสุด (Recent Updates & Fixes)
* **[Feature] ระบบบันทึกสถิติ (Foul Logger):** เพิ่มการบันทึกประวัติการทำฟาวล์ลงไฟล์ `basketball_foul_logs.csv` อัตโนมัติ พร้อมระบบ Cooldown 3 วินาทีเพื่อป้องกันการบันทึกซ้ำซ้อน
* **[Fix] รองรับ Mac Apple Silicon (M1/M2/M3/M4):** แก้ไขบั๊ก `mediapipe has no attribute 'solutions'` โดยปรับสภาพแวดล้อมให้ใช้ **Python 3.11** ซึ่งเสถียรที่สุดสำหรับ MediaPipe บน Mac
* **[Fix] Data Structure & Coordinate Logic:** แก้ไขบั๊กการดึงพิกัด `(x, y)` จาก Tuple ในกฎ Traveling และ Carrying (`tuple object has no attribute y`) เพื่อให้รองรับพิกัด Pixel จริง (Global Coordinates) ที่แม่นยำขึ้น
* **[Improvement] ROI to Global Transformation:** ปรับจูนระบบแปลงพิกัดกระดูกจากภาพตัด (ROI) กลับเป็นภาพรวมหน้าจอ (Global) ทำให้การคำนวณระยะห่างระหว่างมือกับลูกบาสแม่นยำ 100%

--

## ✨ ฟีเจอร์หลัก (Features)
ระบบสามารถตรวจจับเหตุการณ์ดังนี้:
1.  **Double Dribble (เบิ้ลบอล):** ตรวจจับเมื่อผู้เล่นหยุดเลี้ยงบอล (จับสองมือ) แล้วกลับมาเลี้ยงใหม่
2.  **Traveling (พาบอล):** นับก้าวขาขณะถือบอล หากเกิน 2 ก้าวจะแจ้งเตือน
3.  **Carrying (หงายมือเลี้ยง):** ตรวจจับการหงายมือรองใต้ลูกบาสขณะเลี้ยง
4.  **Goaltending (รบกวนห่วง):** ตรวจจับการสัมผัสลูกบาสขณะลูกกำลังตกลงสู่ห่วง (Simulation)

---

## 🛠️ การติดตั้ง (Installation)

### 1. เตรียมสภาพแวดล้อม (Prerequisites)
ต้องติดตั้ง **Python 3.11** แนะนำให้สร้าง Virtual Environment เพื่อความสะอาดของโปรเจกต์ (และป้องกันปัญหาไลบรารีตีกัน)

```bash
# สำหรับ macOS/Linux (แนะนำให้เจาะจงใช้ Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# สำหรับ Windows
python -m venv venv
venv\Scripts\activate
2. ติดตั้ง Library ที่จำเป็น

รันคำสั่งต่อไปนี้ใน Terminal:

Bash
pip install numpy opencv-python mediapipe ultralytics
numpy: ใช้คำนวณคณิตศาสตร์ (ระยะทาง, มุม)

opencv-python: ใช้จัดการภาพและกล้อง (cv2)

mediapipe: ใช้จับจุดกระดูกร่างกาย (Pose Estimation)

ultralytics: ใช้โหลดโมเดล YOLOv8 เพื่อจับลูกบาสและคน

🚀 วิธีการรันโปรแกรม (How to Run)
ตรวจสอบว่าเสียบกล้อง Webcam แล้ว

รันคำสั่ง:

Bash
python main.py
(กดปุ่ม q บนคีย์บอร์ดเพื่อออกจากโปรแกรม)

📂 โครงสร้างไฟล์ (Project Structure)
Plaintext
DETECT-BASKETBALL/
│
├── main.py                # 🎬 ไฟล์หลัก: เปิดกล้อง, เรียก AI, วาดกราฟิก, สั่ง Log
├── referee.py             # 🧠 ผู้คุมกฎ: รับข้อมูลจาก AI ส่งต่อให้กฎต่างๆ
├── utils.py               # 📐 เครื่องมือ: สูตรหา distance, angle และคลาส FoulLogger
│
└── fouls/                 # 📚 กฎกติกา (แยกเป็นโมดูล)
    ├── __init__.py
    ├── double_dribble.py  # กฎเบิ้ลบอล
    ├── traveling.py       # กฎถือบอลเเล้วก้าวเกิน 2 ก้าว
    ├── carrying.py        # กฎหงายมือพาบอล
    └── goaltending.py     # กฎ Goaltending
📖 อธิบายการทำงานอย่างละเอียด (Code Explanation)
1. main.py (หัวใจหลัก)

ทำหน้าที่เป็น Loop หลักของโปรแกรม:

Object Detection: ใช้ YOLOv8 ค้นหา "คน" (Class 0) และ "ลูกบาส" (Class 32) ในภาพ

ROI Cropping: เมื่อเจอคน ระบบจะตัดภาพเฉพาะส่วนคนนั้น (Region of Interest) เพื่อส่งไปเข้า MediaPipe (ช่วยให้ AI ทำงานเร็วขึ้นและแม่นยำขึ้นกว่าส่งทั้งภาพ)

Coordinate Transformation (สำคัญ): พิกัดที่ได้จาก MediaPipe จะเป็นพิกัดในภาพเล็ก (ROI) เราต้องแปลงกลับเป็นพิกัดภาพใหญ่ (Global) เป็นพิกัด Pixel จริงๆ ก่อนส่งให้ Referee ตัดสิน เพื่อให้เทียบตำแหน่งมือกับลูกบาสได้ถูกต้อง

Visualization & Logging: วาดกรอบสี่เหลี่ยม ข้อความแจ้งเตือน และเรียกใช้ FoulLogger เพื่อบันทึก CSV

2. referee.py (ผู้จัดการ)

ทำหน้าที่จัดการผู้เล่นแต่ละคน (Player ID):

สร้าง Instance ของกฎต่างๆ (dd, tr, ca, gt) แยกตามตัวบุคคล

ตรวจสอบเบื้องต้นว่า "ใครถือบอลอยู่" (is_holding) โดยเช็คระยะห่างระหว่างพิกัดมือกับลูกบาส

เรียกใช้ฟังก์ชัน check() ของกฎแต่ละข้อ โดยส่งผ่านพิกัดแบบ Global Pixel

3. Logic การตรวจจับฟาวล์ (Foul Logic)

A. Double Dribble (fouls/double_dribble.py)
ใช้หลักการ State Machine (สถานะ):

IDLE: ไม่ได้ทำอะไร

DRIBBLING: มือข้างเดียวสัมผัสบอล (เลี้ยง)

HOLDING: สองมือสัมผัสบอลพร้อมกัน (หยุดเลี้ยง)

เงื่อนไขฟาวล์: หากสถานะเปลี่ยนจาก HOLDING -> DRIBBLING (หยุดแล้วกลับมาเลี้ยงใหม่) = Double Dribble

B. Traveling (fouls/traveling.py)
ใช้หลักการ Step Counter (นับก้าว):

ทำงานเฉพาะตอนที่ is_holding = True

ดูการเปลี่ยนแปลงตำแหน่งแกน Y ของข้อเท้า (Ankle)

หากมีการขยับขึ้นลงของข้อเท้าเกินค่า Threshold ที่กำหนด ให้นับเป็น 1 ก้าว

เงื่อนไขฟาวล์: หากก้าวสะสมเกิน 2 ก้าว = Traveling

C. Carrying (fouls/carrying.py)
ใช้หลักการ Relative Height (ความสูงสัมพัทธ์):

เปรียบเทียบพิกัดแกน Y ระหว่าง "ข้อมือ" (Wrist) และ "ปลายนิ้ว" (Index Finger)

เงื่อนไขฟาวล์: หากข้อมืออยู่ต่ำกว่าปลายนิ้ว (ค่า Y มากกว่า) เกินระยะที่กำหนด แสดงว่ากำลังหงายมือรองบอล = Carrying

D. Goaltending (fouls/goaltending.py)
ใช้หลักการ Trajectory History (ประวัติการเคลื่อนที่):

เก็บตำแหน่งลูกบาสย้อนหลัง 15 เฟรม

คำนวณว่าลูกบาสกำลัง "ตกลง" หรือไม่ (ค่า Y เพิ่มขึ้น)

ตรวจสอบความสูงลูกบาสว่าอยู่เหนือระดับห่วงหรือไม่ (Simulate ว่าครึ่งจอคือระดับห่วง)

เงื่อนไขฟาวล์: บอลกำลังตก + อยู่สูง + มีคนสัมผัส = Goaltending Warning

⚠️ ข้อจำกัด (Limitations)
2D Perspective: ระบบใช้กล้องตัวเดียว ไม่มีความลึก (Depth) การวัดระยะห่างระหว่างมือกับบอลอาจคลาดเคลื่อนหากมืออยู่คนละระนาบความลึกกับบอล

Occlusion: หากลูกบาสบังมือ หรือตัวคนบังลูกบาส AI อาจจับตำแหน่งไม่ได้ชั่วคราว

Lighting: แสงสว่างมีผลต่อความแม่นยำของ MediaPipe และ YOLO

🔧 การปรับแต่ง (Configuration)
คุณสามารถปรับค่าความไว (Sensitivity) ได้ในไฟล์แต่ละกฎ เช่น:

threshold ใน double_dribble.py: ปรับระยะการสัมผัสบอล (คำนวณเป็นระยะพิกัด Pixel)

threshold ใน traveling.py: ปรับความไวในการนับก้าว

cooldown_sec ใน main.py (ส่วนของการเรียก logger): ปรับระยะเวลาหน่วงก่อนจะบันทึกฟาวล์เดิมซ้ำลงไฟล์ CSV