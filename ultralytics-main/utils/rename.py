import os


def rename_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 构建文件的完整路径
        full_path = os.path.join(directory, filename)

        # 检查是否是文件（而不是文件夹）
        if os.path.isfile(full_path):
            # 查找文件名中第一个 '-' 符号的位置
            index = filename.find('-')

            # 如果找到了 '-' 符号
            if index != -1:
                # 提取从 '-' 符号之后的部分作为新文件名
                new_filename = filename[index + 1:]

                # 构建新文件的完整路径
                new_full_path = os.path.join(directory, new_filename)

                # 重命名文件
                os.rename(full_path, new_full_path)
                print(f'Renamed: {filename} -> {new_filename}')
            else:
                # 如果没有找到 '-' 符号，则打印提示信息（可选）
                print(f'No "-" found in filename: {filename}')


directory_path = r"F:\AI_tooth\data\segment\labels\project1\labels"  # 替换为你的文件夹路径
rename_files(directory_path)