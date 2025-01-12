"""
这段代码是 torch.utils.data.Sampler 的一个扩展，旨在控制数据集的采样，
特别适用于在分布式训练中进行数据分配，同时支持对数据集进行“扩展”，
使得每个进程（或设备）在每轮训练时可以加载不同的数据子集。
"""


import math
import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """
    限制数据加载为数据集的子集的采样器（Sampler）。

    基于 `torch.utils.data.distributed.DistributedSampler` 进行修改，
    支持通过扩展数据集的大小来加速迭代式训练，避免每轮训练后重新启动数据加载器。

    Arg：
        dataset (torch.utils.data.Dataset): 用于采样的数据集，通常是 `Dataset` 类型。
        num_replicas (int | None): 参与训练的进程数，通常是分布式训练中的 `world_size`。
        rank (int | None): 当前进程在所有进程中的编号。每个进程处理不同的数据子集。
        ratio (int): 数据集扩展比例。默认为 1，表示不扩展数据集；如果为 2，则表示扩展为原数据集的 2 倍。
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        """
        初始化 `EnlargedSampler` 对象。

        Arg：
            dataset (torch.utils.data.Dataset): 用于采样的数据集。
            num_replicas (int): 参与训练的进程数（即 `world_size`）。
            rank (int): 当前进程的编号。
            ratio (int): 数据集扩展比例，默认为 1。
        """
        self.dataset = dataset  # 数据集
        self.num_replicas = num_replicas  # 进程数，通常为 `world_size`
        self.rank = rank  # 当前进程的编号
        self.epoch = 0  # 当前 epoch 编号，用于控制数据的随机顺序
        # 计算每个进程需要采样的数据量
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        # 扩展后的数据集总大小（所有进程需要处理的数据总量）
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """
        实现迭代方法，返回采样的索引。

        该方法会根据 `epoch` 值来进行确定性的随机打乱，以保证每个 epoch 加载不同的数据。
        """
        # 创建一个生成器，以确保每个 epoch 的数据顺序不同
        g = torch.Generator()
        g.manual_seed(self.epoch)  # 使用 epoch 作为随机种子，保证每个 epoch 顺序不同
        # 随机打乱数据集中的索引
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)  # 获取数据集大小
        # 对每个随机生成的索引进行取模操作，确保它们在数据集的大小范围内
        indices = [v % dataset_size for v in indices]

        # 根据进程的 `rank` 选择当前进程应该处理的子集
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples  # 确保每个进程采样的数量是正确的

        # 返回迭代器，允许外部遍历这些索引
        return iter(indices)

    def __len__(self):
        """
        返回当前进程需要采样的样本数量。

        Return：
            int: 当前进程需要处理的数据量
        """
        return self.num_samples

    def set_epoch(self, epoch):
        """
        设置当前的 `epoch`，用于控制随机数据的顺序。

        Return：
            epoch (int): 当前的训练轮次（epoch）编号。
        """
        self.epoch = epoch  # 设置当前的 epoch
