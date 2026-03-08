from ultralytics import YOLO  

model = YOLO('yolov8n.pt') 

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="trainmodel/Basketball_Dataset/data.yaml",  # Path to dataset configuration file
    epochs=30,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cpu",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

metrics = model.val()