import os
import shutil


def copy_files(src_folder, dest_folder):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

        # 遍历源文件夹中的所有文件
    for root, dirs, files in os.walk(src_folder):
        for file_name in files:
            # 获取文件名（不包括后缀）
            base_name = os.path.splitext(file_name)[0]

            # 检查文件名是否以1、2或3结尾
            if base_name.endswith('1') or base_name.endswith('2') or base_name.endswith('3'):
                # 构建文件的完整路径
                src_file_path = os.path.join(root, file_name)
                dest_file_path = os.path.join(dest_folder, file_name)

                # 复制文件到目标文件夹
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied: {src_file_path} to {dest_file_path}")


source_folder = r'F:\AI_tooth\data\images'
destination_folder = r'F:\AI_tooth\data\segment\img'

copy_files(source_folder, destination_folder)