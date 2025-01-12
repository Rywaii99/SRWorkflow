import numpy as np
import os
import random
import time
import torch
from os import path as osp

from .dist_util import master_only


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """扫描目录，查找符合条件的文件。

    参数:
        dir_path (str): 要扫描的目录路径。
        suffix (str | tuple(str), optional): 文件后缀，用于筛选目标文件。默认为 None。
        recursive (bool, optional): 是否递归扫描子目录。默认为 False。
        full_path (bool, optional): 是否返回完整的文件路径。默认为 False。

    返回:
        生成器：返回符合条件的文件路径（相对路径或完整路径）。
    """

    # 校验suffix参数的类型
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix"必须是字符串或字符串元组')

    root = dir_path  # 保存根目录路径，用于生成相对路径

    def _scandir(dir_path, suffix, recursive):
        """内部递归扫描文件函数"""
        for entry in os.scandir(dir_path):  # 遍历目录下的每个条目
            # 跳过隐藏文件，确保是文件而非目录
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path  # 返回完整路径
                else:
                    return_path = osp.relpath(entry.path, root)  # 返回相对路径

                # 根据后缀过滤文件
                if suffix is None:
                    yield return_path  # 如果没有指定后缀，返回所有文件
                elif return_path.endswith(suffix):
                    yield return_path  # 返回符合后缀要求的文件路径
            else:
                # 如果是目录且需要递归，则递归扫描
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)  # 返回生成器


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_network paths.

    该函数检查恢复的状态以及预训练网络的路径配置。

    Args:
        opt (dict): Options. 一个包含配置选项的字典，通常包括网络配置、路径配置等信息。
        resume_iter (int): Resume iteration. 恢复的迭代次数，用来指定恢复时使用的模型参数文件。
    """

    # 如果配置中包含恢复状态的路径，意味着我们需要从先前的训练状态恢复
    if opt['path']['resume_state']:
        # 获取所有的网络配置，网络配置的键名是以 'network_' 开头的
        networks = [key for key in opt.keys() if key.startswith('network_')]
        flag_pretrain = False  # 用于标记是否存在预训练模型路径

        # 检查是否有任何网络配置对应的预训练模型路径
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True

        # 如果存在预训练模型路径，则打印提示信息，表明恢复时预训练路径会被忽略
        if flag_pretrain:
            print('pretrain_network path will be ignored during resuming.')

        # 设置预训练模型路径
        for network in networks:
            name = f'pretrain_{network}'  # 构造预训练模型的路径键名
            basename = network.replace('network_', '')  # 去除 'network_' 前缀，得到基础网络名称
            # 如果配置中没有明确指定忽略恢复的网络，则设置预训练模型路径
            if opt['path'].get('ignore_resume_networks') is None or (network
                                                                     not in opt['path']['ignore_resume_networks']):
                # 设置恢复模型的路径，并打印设置的路径
                opt['path'][name] = osp.join(opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                print(f"Set {name} to {opt['path'][name]}")

        # 更新恢复参数时的参数键名
        param_keys = [key for key in opt['path'].keys() if key.startswith('param_key')]
        # 遍历所有的参数键名，如果当前参数键名是 'params_ema'，则将其改为 'params'
        for param_key in param_keys:
            if opt['path'][param_key] == 'params_ema':
                opt['path'][param_key] = 'params'
                # 打印更改的参数键名
                print(f'Set {param_key} to params')


@master_only
def make_exp_dirs(opt):
    """Make dirs for experiments.

    为实验创建所需的目录。根据是否处于训练模式，选择不同的根目录进行目录创建。

    Args:
        opt (dict): Options. 包含实验配置的字典。
    """

    # 复制配置字典中的 'path' 配置，避免修改原始字典
    path_opt = opt['path'].copy()

    # 如果是训练模式
    if opt['is_train']:
        # 删除 'experiments_root' 配置并调用 mkdir_and_rename 函数来创建实验根目录
        mkdir_and_rename(path_opt.pop('experiments_root'))
    else:
        # 如果不是训练模式，删除 'results_root' 配置并创建结果根目录
        mkdir_and_rename(path_opt.pop('results_root'))

    # 遍历路径配置字典中的其他项
    for key, path in path_opt.items():
        # 如果路径配置项与严格加载、预训练网络、恢复状态或参数键有关，跳过
        if ('strict_load' in key) or ('pretrain_network' in key) or ('resume' in key) or ('param_key' in key):
            continue
        else:
            # 否则，为路径创建目录（如果目录不存在的话）
            os.makedirs(path, exist_ok=True)


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    创建目录。如果路径已存在，则将其重命名（带时间戳），并创建一个新的目录。

    Args:
        path (str): Folder path. 要创建或重命名的目录路径。
    """

    # 检查指定路径是否已经存在
    if osp.exists(path):
        # 如果路径存在，则生成一个新名称，新名称包括原路径和时间戳
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        # 使用 os.rename 将原目录重命名为新名称（带时间戳）
        os.rename(path, new_name)

    # 创建新的目录，如果目录已经存在则不报错（exist_ok=True）
    os.makedirs(path, exist_ok=True)
