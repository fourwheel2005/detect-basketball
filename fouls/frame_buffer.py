from collections import deque
import numpy as np
import cv2

class FrameBuffer:
    """
    เก็บ ROI ของผู้เล่นย้อนหลัง N เฟรม
    เพื่อส่งเป็น Clip ให้ 3D CNN วิเคราะห์
    """
    CLIP_LEN   = 16   # จำนวนเฟรมต่อ Clip
    FRAME_SIZE = (64, 64)  # ขนาด Resize

    def __init__(self):
        self._buffer: deque = deque(maxlen=self.CLIP_LEN)

    def push(self, roi_bgr):
        """รับ ROI (BGR numpy array) แล้วเพิ่มเข้า Buffer"""
        frame = cv2.resize(roi_bgr, self.FRAME_SIZE)
        frame = frame.astype(np.float32) / 255.0
        self._buffer.append(frame)

    def is_ready(self) -> bool:
        """มีเฟรมครบ CLIP_LEN แล้วหรือยัง"""
        return len(self._buffer) == self.CLIP_LEN

    def get_clip(self) -> np.ndarray:
        """
        Return shape: (1, CLIP_LEN, H, W, 3)
        รูปแบบที่ TensorFlow/Keras รับได้
        """
        clip = np.stack(list(self._buffer), axis=0)  # (T, H, W, 3)
        return np.expand_dims(clip, axis=0)           # (1, T, H, W, 3)