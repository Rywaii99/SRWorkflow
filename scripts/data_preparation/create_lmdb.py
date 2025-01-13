import argparse
from os import path as osp

from utils import scandir
from utils.lmdb_util import make_lmdb_from_imgs


PRE_DIR = '../../'

def create_lmdb_for_aid():
    """
    为 AID 数据集创建 LMDB 文件。

    Usage:
        通常，AID 数据集中有以下四个需要处理的文件夹：

        * AID_train_HR：高分辨率训练图像的子图
        * AID_train_LR_bicubic/X2：双三次插值下采样的 X2 低分辨率训练图像子图
        * AID_train_LR_bicubic/X3：双三次插值下采样的 X3 低分辨率训练图像子图
        * AID_train_LR_bicubic/X4：双三次插值下采样的 X4 低分辨率训练图像子图

        请根据自己的设置修改 opt 配置。
    """
    # HR（高分辨率）图像
    folder_path = PRE_DIR + 'datasets/processed/AID/AID_train_HR'  # 输入高分辨率图像的文件夹路径
    lmdb_path = PRE_DIR + 'datasets/processed/AID/AID_train_HR.lmdb'  # 输出 LMDB 文件路径
    img_path_list, keys = prepare_keys_aid(folder_path)  # 准备图像路径列表和键列表
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)  # 调用工具函数生成 LMDB 文件

    # LRx2 图像
    folder_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic/X2'  # 输入 X2 低分辨率图像文件夹路径
    lmdb_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic_X2.lmdb'  # 输出 LMDB 文件路径
    img_path_list, keys = prepare_keys_aid(folder_path)  # 准备图像路径列表和键列表
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx3 图像
    folder_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic/X3'  # 输入 X3 低分辨率图像文件夹路径
    lmdb_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic_X3.lmdb'  # 输出 LMDB 文件路径
    img_path_list, keys = prepare_keys_aid(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # LRx4 图像
    folder_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic/X4'  # 输入 X4 低分辨率图像文件夹路径
    lmdb_path = PRE_DIR + 'datasets/processed/AID/AID_train_LR_bicubic_X4.lmdb'  # 输出 LMDB 文件路径
    img_path_list, keys = prepare_keys_aid(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def prepare_keys_aid(folder_path):
    """
    为 AID 数据集准备图像路径列表和键列表。

    Args:
        folder_path (str): 文件夹路径。

    Returns:
        list[str]: 图像路径列表。
        list[str]: 键列表。
    """
    print('Reading image path list ...')  # 打印提示信息，表明正在读取图像路径列表
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))  # 获取文件夹中所有 PNG 格式的图像路径
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]  # 从路径中提取键（去掉后缀）

    return img_path_list, keys


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     '--dataset',
    #     type=str,
    #     help=("Options: 'AID', 'DOTA', 'DIOR' You may need to modify the corresponding configurations in codes."))
    # args = parser.parse_args()
    # dataset = args.dataset.lower()
    # if dataset == 'aid':
    #     create_lmdb_for_aid()
    # # elif dataset == 'dota':
    # #     create_lmdb_for_dota()
    # # elif dataset == 'dior':
    # #     create_lmdb_for_dior()
    # else:
    #     raise ValueError('Wrong dataset.')

    create_lmdb_for_aid()
