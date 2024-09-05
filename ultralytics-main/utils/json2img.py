import base64
import json
import os
from io import BytesIO
from PIL import Image


def get_image(json_folder, output_folder):
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
                image_name = os.path.splitext(os.path.basename(image_path))[0]

                image_data_encoded = data.get('imageData', None)  # 或者 data.get('imagesData', None)
                if not image_data_encoded:
                    print("没有找到 imageData 或 imagesData 字段")
                    return

                # 解码 base64 编码的数据
                image_data = base64.b64decode(image_data_encoded)

                image = Image.open(BytesIO(image_data))
                output_file_path = os.path.join(output_folder, f"{image_name}.jpg")

                # 保存图像到指定目录
                image.save(output_file_path)
                print(f"Image saved to {output_file_path}")

json_folder = "F:\AI_tooth\data\json"
output_folder = "F:\AI_tooth\data\images"

get_image(json_folder, output_folder)
