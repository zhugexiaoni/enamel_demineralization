from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model=YOLO('yolov8n.pt')
model.train(data='data.yaml',imgsz=(6720,4480),workers=0,batch=-1,epochs=60,device=0)