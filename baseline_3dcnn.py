import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization, Input

def build_baseline_3d_cnn(input_shape=(16, 112, 112, 3), num_classes=3):
    """
    สร้างโครงสร้าง 3D CNN สำหรับแยกแยะพฤติกรรมการฟาวล์
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Block 1: ดึงฟีเจอร์พื้นฐาน
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),
        
        # Block 2: ดึงฟีเจอร์ที่ซับซ้อนขึ้น
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),
        
        # Block 3: เน้นความสัมพันธ์เชิงเวลา (Temporal Context)
        Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=(2, 2, 2)),
        
        # Classifier
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5), # ป้องกัน Overfitting เพราะ Dataset ช่วงแรกอาจจะน้อย
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# สร้างโมเดล
model = build_baseline_3d_cnn()
model.summary()