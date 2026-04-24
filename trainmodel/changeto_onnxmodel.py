import tensorflow as tf
import tf2onnx
import onnx

# 1. โหลดโมเดล TensorFlow ที่เทรนเสร็จแล้ว
model_path = "trainmodel/contact_model.h5"
model = tf.keras.models.load_model(model_path)

# 2. แปลงเป็น ONNX
# สมมติว่า Input ของ 3D CNN คือ (Batch, 16 Frames, 112, 112, 3)
input_signature = [tf.TensorSpec([None, 16, 112, 112, 3], tf.float32, name='input_frames')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# 3. บันทึกไฟล์
onnx.save(onnx_model, "trainmodel/contact_model.onnx")
print("✅ แปลงไฟล์เป็น ONNX สำเร็จ!")