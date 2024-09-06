# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

model=YOLO('yolov8n.pt')
model.train(data='data.yaml',imgsz=(720,1280),workers=2,batch=-1,epochs=60)