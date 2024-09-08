import json
import os
import base64
from io import BytesIO
from PIL import Image
import shutil
import random

# 生成标签
def convert_labelme_to_yolo(json_folder, output_folder):
    """
    将labelme的JSON标注转换为YOLO格式的txt文件。

    :param json_folder: 包含JSON文件的文件夹路径。
    :param output_folder: 存储转换后txt文件的文件夹路径。
    :param image_width: 图像的宽度（用于归一化）。
    :param image_height: 图像的高度（用于归一化）。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_path = data.get('imagePath', None)  # 可能需要根据实际情况调整
                image_width = data.get('imageWidth',None)
                image_height = data.get('imageHeight',None)
                if image_path is None:
                    # 如果没有imagePath，可能需要从文件名或其他地方获取图像尺寸
                    # 这里我们假设所有图像都有相同的尺寸，并已通过参数传入
                    pass
                image_name = os.path.splitext(os.path.basename(filename))[0]

                with open(os.path.join(output_folder, f"{image_name}.txt"), 'w') as out:
                    for shape in data['shapes']:
                        label = 1
                        if shape['label'] != 'ED':
                            label = 0
                        points = shape['points']

                        # 假设points是一个包含四个点（左上角、右上角、右下角、左下角）的列表
                        # 但通常labelme只提供多边形的顶点，这里我们取前两个和后两个点作为边界框
                        x_min, y_min = min(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1]
                        x_max, y_max = max(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1]

                        # 归一化坐标
                        center_x = (x_min + x_max) / (2 * image_width)
                        center_y = (y_min + y_max) / (2 * image_height)
                        width = (x_max - x_min) / image_width
                        height = (y_max - y_min) / image_height
                        # 为适应图片压缩，坐标同比缩放
                        center_x *= 720/4480
                        center_y *= 720/4480
                        width *= 720/4480
                        height *= 720/4480

                        # YOLO格式：类别 中心点x 中心点y 宽度 高度
                        line = f"{label} {center_x} {center_y} {width} {height}\n"
                        out.write(line)
            print(image_name+'.txt',"保存成功")

json_folder = "/home/aistudio/data/data293184/json"
output_folder = "/home/aistudio/data/data293184/labels"

convert_labelme_to_yolo(json_folder, output_folder)

# 从json转为图片
def get_image(json_folder, output_folder,size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(json_folder, filename)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_path = data.get('imagePath', None)  # 可能需要根据实际情况调整
                if image_path is None:
                    # 如果没有imagePath，可能需要从文件名或其他地方获取图像尺寸
                    # 这里我们假设所有图像都有相同的尺寸，并已通过参数传入
                    pass
                image_name = os.path.splitext(os.path.basename(filename))[0]

                image_data_encoded = data.get('imageData', None)  # 或者 data.get('imagesData', None)
                if not image_data_encoded:
                    print("没有找到 imageData 或 imagesData 字段")
                    return

                # 解码 base64 编码的数据
                image_data = base64.b64decode(image_data_encoded)

                image = Image.open(BytesIO(image_data))

                # 压缩图片至指定的大小  
                image.thumbnail(size)  
                  
                # 保存压缩后的图片 
                output_path = os.path.join(output_folder, f"{image_name}.jpg") 
                image.save(output_path, optimize=True, quality=85)  # quality参数可以根据需要调整
                
                print(f"Image saved to {output_path}")

json_folder = "/home/aistudio/data/data293184/json"
output_folder = "/home/aistudio/data/data293184/images"

get_image(json_folder, output_folder, size=(1080, 720))

# 分割数据集
random.seed(0)
def split_data(file_path, xml_path, new_file_path, train_rate, val_rate, test_rate):
    each_class_image = []
    each_class_label = []
    for image in os.listdir(file_path):
        each_class_image.append(image)
    for label in os.listdir(xml_path):
        each_class_label.append(label)
    each_class_image.sort()
    each_class_label.sort()
    print(each_class_image)
    print(each_class_label)
    data = list(zip(each_class_image, each_class_label))
    total = len(each_class_image)
    random.shuffle(data)
    each_class_image, each_class_label = zip(*data)
    # print(each_class_image)
    # print(each_class_label)
    train_images = each_class_image[0:int(train_rate * total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_images = each_class_image[int((train_rate + val_rate) * total):]
    train_labels = each_class_label[0:int(train_rate * total)]
    val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]
    test_labels = each_class_label[int((train_rate + val_rate) * total):]

    for image in train_images:
        print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in train_labels:
        print(label)
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)

    for image in val_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in val_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)

    for image in test_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    for label in test_labels:
        old_path = xml_path + '/' + label
        new_path1 = new_file_path + '/' + 'test' + '/' + 'labels'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + label
        shutil.copy(old_path, new_path)


file_path = r"/home/aistudio/data/data293184/images"
txt_path = r"/home/aistudio/data/data293184/labels"
new_file_path = r"/home/aistudio/work/yolo_data"
split_data(file_path, txt_path, new_file_path, train_rate=0.7, val_rate=0.2, test_rate=0.1)