from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n-pose.pt')

# Run inference on 'bus.jpg' with arguments
model.predict('1.jpg', save=True, imgsz=320, conf=0.5,show = True,save_txt = True)