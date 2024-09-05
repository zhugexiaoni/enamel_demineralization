import base64
import json
from PIL import Image
from io import BytesIO

def display_image_from_labelsme_json(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # 假设 imageData 字段存在且包含图片的 base64 编码
    # 注意：实际字段名可能是 'imageData' 或 'imagesData'，取决于你的数据
    image_data_encoded = data.get('imageData', None)  # 或者 data.get('imagesData', None)
    if not image_data_encoded:
        print("没有找到 imageData 或 imagesData 字段")
        return

    # 去掉 base64 编码字符串的数据前缀（如果有的话）
    # image_data_encoded = image_data_encoded.split(',')[1]

    # 解码 base64 编码的数据
    image_data = base64.b64decode(image_data_encoded)
    
    # 使用 PIL 加载图像
    image = Image.open(BytesIO(image_data))

    # 显示图像
    image.show()

# 替换为你的 JSON 文件路径
json_file_path = r"F:\AI_tooth\data\labels\1-1.json"
display_image_from_labelsme_json(json_file_path)