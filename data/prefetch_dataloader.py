"""
这段代码实现了数据加载器（DataLoader）的预取机制，旨在提高数据加载效率。
代码使用了多线程和GPU预取来加速数据加载过程。
数据加载是深度学习训练中的一个瓶颈，通过异步预取，可以在模型计算时提前加载数据，从而提高训练效率。
"""
import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader


class PrefetchGenerator(threading.Thread):
    """通用预取生成器。

    参数:
        generator: Python生成器。
        num_prefetch_queue (int): 预取队列的数量。
    """
    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)  # 创建一个队列来存储预取的数据
        self.generator = generator  # 保存数据生成器
        self.daemon = True  # 使线程为守护线程
        self.start()  # 启动线程

    def run(self):
        # 运行线程并从生成器获取数据
        for item in self.generator:
            self.queue.put(item)  # 将数据放入队列中
        self.queue.put(None)  # 当数据生成完成，放入None表示结束

    def __next__(self):
        # 从队列获取下一条数据
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration  # 如果队列中没有数据，抛出停止迭代异常
        return next_item

    def __iter__(self):
        return self  # 返回迭代器本身


class PrefetchDataLoader(DataLoader):
    """预取版本的DataLoader。

    参数:
        num_prefetch_queue (int): 预取队列的数量。
        kwargs (dict): 传递给DataLoader的其他参数。
    """
    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue  # 设置队列大小
        super(PrefetchDataLoader, self).__init__(**kwargs)  # 调用父类构造函数

    def __iter__(self):
        # 返回一个PrefetchGenerator实例
        # PrefetchDataLoader 在原有的 DataLoader 基础上，重载了 __iter__ 方法，使用 PrefetchGenerator 来实现数据的异步预取。
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """CPU预取器。

    参数:
        loader: DataLoader。
    """

    def __init__(self, loader):
        self.ori_loader = loader  # 保存原始数据加载器
        self.loader = iter(loader)  # 将原始加载器转为迭代器

    def next(self):
        try:
            return next(self.loader)  # 获取下一批数据
        except StopIteration:
            return None  # 如果没有更多数据，返回None

    def reset(self):
        self.loader = iter(self.ori_loader)  # 重置迭代器


class CUDAPrefetcher():
    """CUDA预取器。

    参数:
        loader: DataLoader。
        opt (dict): 配置选项。
    """

    def __init__(self, loader, opt):
        self.ori_loader = loader  # 保存原始数据加载器
        self.loader = iter(loader)  # 转为迭代器
        self.opt = opt
        self.stream = torch.cuda.Stream()  # 创建一个CUDA流
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')  # 设置设备
        self.preload()  # 预加载数据

    def preload(self):
        """预加载数据到GPU"""
        try:
            self.batch = next(self.loader)  # 从迭代器中获取数据
        except StopIteration:
            self.batch = None  # 如果没有数据，设为None
            return None

        # 将数据加载到GPU
        with torch.cuda.stream(self.stream):  # 使用指定的CUDA流
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        """获取下一批数据"""
        torch.cuda.current_stream().wait_stream(self.stream)  # 等待当前CUDA流完成
        batch = self.batch
        self.preload()  # 预加载下一批数据
        return batch

    def reset(self):
        """重置数据加载器"""
        self.loader = iter(self.ori_loader)  # 重置迭代器
        self.preload()  # 重新预加载数据


