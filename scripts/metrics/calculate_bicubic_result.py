import os
import cv2
import numpy as np
from tqdm import tqdm
from metrics.metric_util import reorder_image, to_y_channel
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

# 动态配置参数
dataset_name = 'DIOR'  # 数据集名称
scale = 3  # 超分倍数
sr_dir = f'../../datasets/processed/{dataset_name}/{dataset_name}_test_HR_bicubic/X{scale}'  # SR图像目录
hr_dir = f'../../datasets/processed/{dataset_name}/{dataset_name}_test_HR'  # HR图像目录
log_path = f'./{dataset_name}_X{scale}_bicubic_metrics.log'  # 日志文件路径
crop_border = 3  # 裁剪边缘像素
test_y_channel = False  # 是否使用Y通道计算
input_order = 'HWC'  # 输入图像的通道顺序

# 初始化存储结果
psnr_list = []
ssim_list = []

# 获取SR和HR图像文件名列表，确保一一对应
sr_files = sorted(os.listdir(sr_dir))
hr_files = sorted(os.listdir(hr_dir))
assert len(sr_files) == len(hr_files), "SR和HR图像数量不一致"

# 打印当前处理的数据集和超分倍数
print(f"正在处理数据集: {dataset_name}, 超分倍数: X{scale}")
print(f"SR图像目录: {sr_dir}")
print(f"HR图像目录: {hr_dir}")

# 处理图像的函数
def process_image(sr_name, hr_name):
    sr_path = os.path.join(sr_dir, sr_name)
    hr_path = os.path.join(hr_dir, hr_name)

    # 读取图像（OpenCV读取为BGR格式）
    sr_img = cv2.imread(sr_path)
    hr_img = cv2.imread(hr_path)
    assert sr_img is not None and hr_img is not None, f"图像读取失败: {sr_path} 或 {hr_path}"

    # 转换颜色通道为RGB
    sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
    hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

    # 调整图像顺序（如果需要）
    sr_img = reorder_image(sr_img, input_order=input_order)
    hr_img = reorder_image(hr_img, input_order=input_order)

    # 裁剪边缘
    if crop_border != 0:
        sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, ...]
        hr_img = hr_img[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 转换为Y通道（如果启用）
    if test_y_channel:
        sr_img = to_y_channel(sr_img)
        hr_img = to_y_channel(hr_img)

    # 计算PSNR
    psnr = calculate_psnr(sr_img, hr_img, crop_border=0, input_order=input_order, test_y_channel=False)

    # 计算SSIM
    ssim = calculate_ssim(sr_img, hr_img, crop_border=0, input_order=input_order, test_y_channel=False)

    return psnr, ssim

# 使用线程池并行处理图像
with ThreadPoolExecutor() as executor:
    futures = []
    for sr_name, hr_name in zip(sr_files, hr_files):
        futures.append(executor.submit(process_image, sr_name, hr_name))

    # 获取每个任务的结果
    for future in tqdm(as_completed(futures), total=len(futures), desc="计算指标"):
        psnr, ssim = future.result()
        psnr_list.append(psnr)
        ssim_list.append(ssim)

# 计算平均值
avg_psnr = np.mean(psnr_list)
avg_ssim = np.mean(ssim_list)

print(f"平均 PSNR: {avg_psnr:.4f}")
print(f"平均 SSIM: {avg_ssim:.4f}")

# 将结果写入日志文件
with open(log_path, 'w') as log_file:
    log_file.write(f"数据集: {dataset_name}\n")
    log_file.write(f"超分倍数: X{scale}\n")
    log_file.write(f"SR图像目录: {sr_dir}\n")
    log_file.write(f"HR图像目录: {hr_dir}\n")
    log_file.write(f"平均 PSNR: {avg_psnr:.4f}\n")
    log_file.write(f"平均 SSIM: {avg_ssim:.4f}\n")

print(f"测试结果已保存到: {log_path}")
