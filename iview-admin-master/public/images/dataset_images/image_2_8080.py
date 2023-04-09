from PIL import Image
import os

# 设置图片文件夹路径
img_folder = './'

# 遍历文件夹下的所有文件
for filename in os.listdir(img_folder):
    # 检查文件是否是图片格式
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # 打开图片文件
        img = Image.open(os.path.join(img_folder, filename))
        # 将图片转换为80x80大小
        img = img.resize((80, 80))
        # 保存处理后的图片
        img.save(os.path.join(img_folder, filename))
