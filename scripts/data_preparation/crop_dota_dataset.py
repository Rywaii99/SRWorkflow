import os
import cv2
import numpy as np
from tqdm import tqdm

# 输入和输出文件夹路径
input_folder = "../../datasets/processed/DOTA/test"  # 输入的图像文件夹
output_folder = "../../datasets/processed/DOTA/test_crop"  # 输出的图像文件夹
os.makedirs(output_folder, exist_ok=True)

# 切割的图像尺寸
crop_size = 792

# 黑色区域的阈值，低于该值的像素认为是黑色区域
black_threshold_ratio = 0.01  # 如果黑色区域占比超过1%，则舍弃

# 获取文件夹中的所有图像文件
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]


def is_black_region(crop, black_threshold_ratio):
    """
    判断裁剪区域中纯黑色（RGB(0, 0, 0)）部分的占比是否超过阈值，超过则返回True，否则返回False。
    """
    # 计算图像中黑色区域的像素数
    black_pixel_count = np.sum(np.all(crop == [0, 0, 0], axis=-1))  # 检查RGB(0,0,0)的位置
    total_pixel_count = crop.size // 3  # 每个裁剪块的总像素数量（每个像素有3个通道）

    # 计算黑色区域的比例
    black_ratio = black_pixel_count / total_pixel_count
    return black_ratio > black_threshold_ratio  # 如果黑色区域超过设定比例，则认为是黑色区域


def crop_image(image, crop_size, black_threshold_ratio):
    """
    将一张图像切割成多个不重叠的小图，大小为 crop_size x crop_size
    并判断切割块是否包含过多黑色区域（如果是，舍弃该切割块）
    """
    h, w, c = image.shape
    crops = []

    # 遍历图像，按指定尺寸切割
    for y in range(0, h, crop_size):
        for x in range(0, w, crop_size):
            # 获取每个切割区域
            crop = image[y:y + crop_size, x:x + crop_size]

            # 确保每个切割区域是完整的crop_size大小
            if crop.shape[0] == crop_size and crop.shape[1] == crop_size:
                # 如果这个裁剪块包含过多的黑色区域，则丢弃
                if not is_black_region(crop, black_threshold_ratio):
                    crops.append(crop)

    return crops


# 使用 tqdm 显示整体进度条，遍历所有图像
for img_file in tqdm(image_files, desc="Processing images", ncols=100, unit="file"):
    img_path = os.path.join(input_folder, img_file)
    image = cv2.imread(img_path)

    if image is not None:
        # 获取切割图像
        crops = crop_image(image, crop_size, black_threshold_ratio)

        # 保存每一个有效的小图
        for i, crop in enumerate(crops):
            # 保存文件名
            output_file = os.path.join(output_folder, f"{os.path.splitext(img_file)[0]}_crop_{i + 1}.png")
            cv2.imwrite(output_file, crop)
            # print(f"Saved {output_file}")
    else:
        print(f"Failed to load image {img_path}")
