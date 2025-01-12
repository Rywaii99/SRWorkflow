import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from data.transforms import mod_crop
from utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False):
    """从给定的文件夹路径读取图像序列。

    Args:
        path (list[str] | str): 图像路径列表或图像文件夹路径。
        require_mod_crop (bool): 是否需要对每张图像进行 mod_crop（修改裁剪）。默认：False。
        scale (int): mod_crop 的缩放因子。默认：1。
        return_imgname (bool): 是否返回图像的名称。默认：False。

    Returns:
        Tensor: 图像序列，形状为 (t, c, h, w)，RGB，值范围 [0, 1]。
        list[str]: 图像名称列表（如果 `return_imgname=True`）。
    """
    if isinstance(path, list):
        img_paths = path  # 如果路径是列表，直接使用
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))  # 否则扫描文件夹获取图像路径，并排序
    # 读取图像，并将其转换为浮动32位，并归一化到 [0, 1] 范围
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    # 如果需要进行 mod_crop，则对每张图像进行裁剪
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]

    # 将图像从 BGR 转为 RGB，并转换为张量
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)  # 将图像堆叠为一个张量

    # 如果需要返回图像名称，提取文件名并返回
    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def paired_paths_from_lmdb(folders, keys):
    """从 LMDB 文件生成配对路径。

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    data.mdb 和 lock.mdb 是标准的 lmdb 文件，详情可参考
    https://lmdb.readthedocs.io/en/release/ 了解更多详情。

    meta_info.txt 是一个指定的 txt 文件，用于记录数据集的元信息。
    数据集的元信息。它将在
    数据集工具会自动创建该文件。
    txt 文件中的每一行都会记录
    1）图像名称（带扩展名）、
    2）图像形状、
    3）压缩级别，中间用空格隔开。
    例如：`baboon.png (120,125,3) 1`...

    我们使用不带扩展名的图像名称作为 lmdb 密钥。
    请注意，我们对相应的 lq 和 gt 图像使用相同的密钥。

    Args:
        folders (list[str]): 文件夹路径列表，顺序应为 [输入文件夹, GT 文件夹]。
        keys (list[str]): 标识文件夹的键，顺序应与 folders 一致，例如 ['lq', 'gt']。

    Returns:
        list[str]: 返回路径列表。
    """
    assert len(folders) == 2, ('folders 的长度应该是 2，分别是输入文件夹和 GT 文件夹。')
    assert len(keys) == 2, 'keys 的长度应该是 2。'

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} 文件夹和 {gt_key} 文件夹应该都是 LMDB 格式。')

    # 确保两个 meta_info 文件的键相同
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]

    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'{input_key} 文件夹和 {gt_key} 文件夹中的键不匹配。')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """从 meta 信息文件生成配对路径。

    元信息文件中的每一行都包含图像名称和
    图像形状（通常为 gt），中间用空格隔开。

    元信息文件示例：
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): 文件夹路径列表，顺序应为 [输入文件夹, GT 文件夹]。
        keys (list[str]): 标识文件夹的键，顺序应与 folders 一致。
        meta_info_file (str): meta 信息文件的路径。
        filename_tmpl (str): 文件名模板，通常用于生成输入文件夹的文件名。

    Returns:
        list[str]: 返回路径列表。
    """
    assert len(folders) == 2, ('folders 的长度应该是 2，分别是输入文件夹和 GT 文件夹。')
    assert len(keys) == 2, 'keys 的长度应该是 2。'

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.strip().split(' ')[0] for line in fin]  # 读取 GT 图像名称

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))  # 获取文件名和扩展名
        input_name = f'{filename_tmpl.format(basename)}{ext}'  # 根据模板生成输入文件名
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """从文件夹中生成配对路径。

    Args:
        folders (list[str]): 文件夹路径列表，顺序应为 [输入文件夹, GT 文件夹]。
        keys (list[str]): 标识文件夹的键，顺序应与 folders 一致。
        filename_tmpl (str): 文件名模板，通常用于生成输入文件夹的文件名。

    Returns:
        list[str]: 返回路径列表。
    """
    assert len(folders) == 2, ('folders 的长度应该是 2，分别是输入文件夹和 GT 文件夹。')
    assert len(keys) == 2, 'keys 的长度应该是 2。'

    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))  # 获取输入文件夹中的所有文件路径
    gt_paths = list(scandir(gt_folder))  # 获取 GT 文件夹中的所有文件路径

    assert len(input_paths) == len(gt_paths), (f'{input_key} 和 {gt_key} 数据集中的图像数量不同：'
                                               f'{len(input_paths)}, {len(gt_paths)}。')

    paths = []
    for gt_path in gt_paths:
        basename, ext = osp.splitext(osp.basename(gt_path))  # 获取 GT 文件名和扩展名
        input_name = f'{filename_tmpl.format(basename)}{ext}'  # 根据模板生成输入文件名
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """从文件夹中生成路径。

    Args:
        folder (str): 文件夹路径。

    Returns:
        list[str]: 返回路径列表。
    """
    paths = list(scandir(folder))  # 获取文件夹中所有文件路径
    paths = [osp.join(folder, path) for path in paths]  # 生成完整路径
    return paths


def paths_from_lmdb(folder):
    """从 LMDB 文件夹中生成路径。

    Args:
        folder (str): LMDB 文件夹路径。

    Returns:
        list[str]: 返回路径列表。
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'文件夹 {folder} 必须是 LMDB 格式。')

    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]  # 从 meta_info.txt 文件中读取所有键并去掉扩展名
    return paths

