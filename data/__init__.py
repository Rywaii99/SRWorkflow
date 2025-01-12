import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from data.prefetch_dataloader import PrefetchDataLoader
from utils import get_root_logger, scandir
from utils.dist_util import get_dist_info
from utils.registry import DATASET_REGISTRY

__all__ = ['build_dataset', 'build_dataloader']

# 自动扫描并导入数据集模块
# 扫描 'data' 文件夹下所有以 '_dataset.py' 结尾的文件
data_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录的绝对路径
dataset_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(data_folder) if v.endswith('_dataset.py')]  # 获取所有以 '_dataset.py' 结尾的文件名（不包括扩展名）
# 导入所有数据集模块
_dataset_modules = [importlib.import_module(f'data.{file_name}') for file_name in dataset_filenames]  # 动态导入每个数据集模块


def build_dataset(dataset_opt):
    """根据配置文件选项构建数据集。

    参数：
        dataset_opt (dict): 数据集配置字典，必须包含以下内容：
            name (str): 数据集名称。
            type (str): 数据集类型。

    返回：
        dataset (torch.utils.data.Dataset): 构建好的数据集。
    """
    dataset_opt = deepcopy(dataset_opt)  # 深拷贝配置字典，避免直接修改原始字典
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt)  # 根据数据集类型从注册器中获取并实例化数据集
    logger = get_root_logger()  # 获取日志记录器
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.')  # 输出数据集创建日志
    return dataset  # 返回构建好的数据集


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """根据配置文件选项构建数据加载器。

    参数：
        dataset (torch.utils.data.Dataset): 数据集。
        dataset_opt (dict): 数据集配置字典，必须包含以下内容：
            phase (str): 'train' 或 'val'。
            num_worker_per_gpu (int): 每个 GPU 的工作进程数。
            batch_size_per_gpu (int): 每个 GPU 的批量大小。
        num_gpu (int): GPU 数量，仅在训练阶段使用。默认值：1。
        dist (bool): 是否进行分布式训练，仅在训练阶段使用。默认值：False。
        sampler (torch.utils.data.sampler): 数据采样器，默认为 None。
        seed (int | None): 随机种子，默认为 None。

    返回：
        dataloader (torch.utils.data.DataLoader): 构建好的数据加载器。
    """
    phase = dataset_opt['phase']  # 获取数据集的训练阶段（训练、验证或测试）
    rank, _ = get_dist_info()  # 获取分布式训练中的 rank 和 world_size
    if phase == 'train':  # 如果是训练阶段
        if dist:  # 如果是分布式训练
            batch_size = dataset_opt['batch_size_per_gpu']  # 使用每个 GPU 的批量大小
            num_workers = dataset_opt['num_worker_per_gpu']  # 使用每个 GPU 的工作进程数
        else:  # 非分布式训练
            multiplier = 1 if num_gpu == 0 else num_gpu  # 根据 GPU 数量设置乘数
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier  # 设置总的批量大小
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier  # 设置总的工作进程数
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # 默认不打乱顺序
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)  # 丢弃最后一个不完整的批次
        if sampler is None:
            dataloader_args['shuffle'] = True  # 如果没有指定采样器，则启用数据打乱
        # 设置 worker 初始化函数
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # 如果是验证或测试阶段
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)  # 只使用一个工作进程
    else:
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")  # 错误的阶段配置

    # 获取额外的配置项
    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')  # 获取预取模式
    if prefetch_mode == 'cpu':  # 使用 CPU 预取
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)  # 获取预取队列数量
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)  # 使用 PrefetchDataLoader
    else:
        # prefetch_mode=None: 普通数据加载器
        # prefetch_mode='cuda': 使用 CUDAPrefetcher 数据加载器
        return torch.utils.data.DataLoader(**dataloader_args)  # 返回标准的 PyTorch DataLoader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """为每个 worker 设置不同的随机种子。

    参数：
        worker_id (int): 当前 worker 的 ID。
        num_workers (int): 总的 worker 数量。
        rank (int): 当前进程的 rank。
        seed (int): 随机种子。

    设置的随机种子用于确保每个 worker 在数据加载时产生一致的随机性。
    """
    worker_seed = num_workers * rank + worker_id + seed  # 根据 rank 和 worker_id 设置种子
    np.random.seed(worker_seed)  # 设置 NumPy 随机种子
    random.seed(worker_seed)  # 设置 Python 随机模块的种子
