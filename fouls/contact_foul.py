from fouls.frame_buffer          import FrameBuffer
from fouls.contact_3dcnn_model   import ContactFoulModel

class ContactFoulDetector:
    """
    ตรวจฟาวล์ประเภท "สัมผัส" ด้วย 3D CNN
    ทำงานทุก CLIP_LEN เฟรม (ไม่ใช่ทุกเฟรม)
    """
    def __init__(self, model_path: str = "trainmodel/contact_model.h5"):
        self.buffer = FrameBuffer()
        self.model  = ContactFoulModel(model_path)
        self._last_result = (False, "")  # Cache ผลล่าสุด

    def check(self, roi_bgr) -> tuple:
        """
        Parameters:
            roi_bgr: numpy array BGR ของผู้เล่น (ตัดมาแล้ว)
        Returns:
            (is_violation: bool, message: str)
        """
        self.buffer.push(roi_bgr)

        # ทำนายเฉพาะเมื่อ Buffer เต็ม (ทุก 16 เฟรม)
        if self.buffer.is_ready():
            clip        = self.buffer.get_clip()
            label, conf = self.model.predict(clip)

            if label:
                msg = f"{label.upper().replace('_',' ')} ({conf:.0%})"
                self._last_result = (True, msg)
            else:
                self._last_result = (False, "")

        return self._last_result