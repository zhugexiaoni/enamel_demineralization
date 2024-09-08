import os  
from ultralytics import YOLO
  
# 加载模型  
model = YOLO('/home/aistudio/work/enamel_demineralization/ultralytics-main/utils/runs/detect/train2/weights/best.pt')
model.val()
# # 假设的输入图片文件夹路径  
# img_folder = '/home/aistudio/work/yolo_data/test/images'   
  
# # 遍历文件夹中的所有图片  
# for filename in os.listdir(img_folder):  
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  
#         img_path = os.path.join(img_folder, filename)  
#         model.predict(source=img_path,save=True)  
#         print(f'Processed {filename}')