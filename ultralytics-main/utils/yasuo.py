from PIL import Image  
import os  
  
def compress_images(folder_path, output_folder, size=(1080, 720)):
    """  
    批量压缩指定文件夹中的所有图片到指定大小，并保存到另一个文件夹中。  
  
    :param folder_path: 原始图片文件夹的路径  
    :param output_folder: 压缩后图片保存的文件夹路径  
    :param size: 压缩后的图片大小，格式为(宽度, 高度)  
    """  
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    # 遍历指定文件夹中的所有文件  
    for filename in os.listdir(folder_path):  
        # 构建文件的完整路径  
        img_path = os.path.join(folder_path, filename)  
          
        # 检查文件是否是图片（这里简单判断文件扩展名，可以根据需要增加更多判断）  
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
            # 打开图片  
            with Image.open(img_path) as img:  
                # 设定压缩后的大小  
                img.thumbnail(size)  
                  
                # 构建输出文件的路径  
                output_path = os.path.join(output_folder, filename)  
                  
                # 保存压缩后的图片  
                img.save(output_path, optimize=True, quality=85)  # quality参数可以根据需要调整
            print(output_path,"保存成功") 
  
# 示例用法  
compress_images('/home/aistudio/data/data293184/yolo_data/test/images', '/home/aistudio/work/yolo_data/test/images')