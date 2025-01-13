import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from utils import scandir  # 自定义工具函数，用于扫描文件夹中的文件

PRE_DIR = '../../'

def main():
    """
    多线程工具，用于将大图像裁剪成子图像以加快 IO 速度。

    Args:
        opt (dict): 配置字典，包含以下配置项：
        - n_thread (int): 线程数量。
        - compression_level (int): PNG 图像的压缩等级（范围：0 到 9）。
            值越高表示压缩时间更长，但生成的文件越小。默认值为 3，与 cv2 一致。
        - dataset (str): 要处理的数据集名称。
        - input_folder (str): 输入文件夹路径。
        - save_folder (str): 保存文件夹路径。
        - crop_size (int): 裁剪大小。
        - step (int): 滑动窗口的步长。
        - thresh_size (int): 阈值大小，小于此阈值的图像将被丢弃。

    Usage:
        对于数据集中的每个文件夹运行本脚本，
        通常需要处理以下四个文件夹：
        - xxx_train_HR
        - xxx_train_LR_bicubic/X2
        - xxx_train_LR_bicubic/X3
        - xxx_train_LR_bicubic/X4

        修改 opt 配置以匹配数据集设置。
    """
    opt = {}
    opt['n_thread'] = 20  # 设置线程数量为 20
    opt['compression_level'] = 3  # 设置 PNG 压缩等级为 3
    opt['dataset'] = 'AID'

    # 高分辨率 (HR) 图像处理
    opt['input_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_HR'  # 输入文件夹路径
    opt['save_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_HR_sub'  # 保存文件夹路径
    opt['crop_size'] = 480  # 子图像大小
    opt['step'] = 240  # 滑动窗口步长
    opt['thresh_size'] = 0  # 忽略阈值
    extract_subimages(opt)

    # 低分辨率 (LR) x2 图像处理
    opt['input_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_LR_bicubic/X2'
    opt['save_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_LR_bicubic/X2_sub'
    opt['crop_size'] = 240
    opt['step'] = 120
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # 低分辨率 (LR) x3 图像处理
    opt['input_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_LR_bicubic/X3'
    opt['save_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + 'train_LR_bicubic/X3_sub'
    opt['crop_size'] = 160
    opt['step'] = 80
    opt['thresh_size'] = 0
    extract_subimages(opt)

    # 低分辨率 (LR) x4 图像处理
    opt['input_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_LR_bicubic/X4'
    opt['save_folder'] = PRE_DIR + 'datasets/processed/' + opt['dataset'] + '/' + opt['dataset'] + '_train_LR_bicubic/X4_sub'
    opt['crop_size'] = 120
    opt['step'] = 60
    opt['thresh_size'] = 0
    extract_subimages(opt)


def extract_subimages(opt):
    """
    将图像裁剪为子图像。

    Args:
        opt (dict): 配置字典，包含以下内容：
        - input_folder (str): 输入文件夹路径。
        - save_folder (str): 保存文件夹路径。
        - n_thread (int): 线程数量。
    """
    input_folder = opt['input_folder']  # 输入文件夹
    save_folder = opt['save_folder']  # 保存文件夹
    if not osp.exists(save_folder):  # 如果保存文件夹不存在
        os.makedirs(save_folder)  # 创建保存文件夹
        print(f'mkdir {save_folder} ...')  # 打印文件夹创建信息
    else:
        print(f'Folder {save_folder} already exists. Exit.')  # 如果文件夹已存在，退出程序
        sys.exit(1)

    img_list = list(scandir(input_folder, full_path=True))  # 获取输入文件夹中的图像路径列表

    # 进度条显示处理进度
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])  # 创建线程池
    for path in img_list:
        # 异步处理每张图像
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()  # 等待所有线程完成
    pbar.close()
    print('All processes done.')  # 输出处理完成信息


def worker(path, opt):
    """
    每个线程的工作函数，用于裁剪单张图像。

    Args:
        path (str): 图像路径。
        opt (dict): 配置字典，包含以下内容：
        - crop_size (int): 裁剪大小。
        - step (int): 滑动窗口步长。
        - thresh_size (int): 阈值大小。
        - save_folder (str): 保存文件夹路径。
        - compression_level (int): PNG 压缩等级。

    返回：
        process_info (str): 进度条中显示的处理信息。
    """
    crop_size = opt['crop_size']  # 裁剪大小
    step = opt['step']  # 滑动窗口步长
    thresh_size = opt['thresh_size']  # 忽略阈值
    img_name, extension = osp.splitext(osp.basename(path))  # 提取图像文件名和扩展名

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # 读取图像（保留原始数据）

    h, w = img.shape[0:2]  # 获取图像高度和宽度
    h_space = np.arange(0, h - crop_size + 1, step)  # 高度滑动窗口位置
    if h - (h_space[-1] + crop_size) > thresh_size:  # 如果最后一个子图像不足阈值大小
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)  # 宽度滑动窗口位置
    if w - (w_space[-1] + crop_size) > thresh_size:  # 如果最后一个子图像不足阈值大小
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            # 裁剪子图像
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)  # 转换为连续数组
            # 保存子图像
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main()  # 调用主函数
