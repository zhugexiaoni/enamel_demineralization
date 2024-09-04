import json
import os


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
                image_name = os.path.splitext(os.path.basename(image_path))[0]

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

                        # YOLO格式：类别 中心点x 中心点y 宽度 高度
                        line = f"{label} {center_x} {center_y} {width} {height}\n"
                        out.write(line)
            print(image_path,"保存成功")

json_folder = "F:\AI_tooth\data\json"
output_folder = "F:\AI_tooth\data\labels"

convert_labelme_to_yolo(json_folder, output_folder)