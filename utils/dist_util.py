import functools
import os
import subprocess
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def init_dist(launcher, backend='nccl', **kwargs):
    """
        初始化分布式训练环境。

        根据选择的启动方式（'pytorch' 或 'slurm'）来初始化分布式训练。
        该函数会设置多进程启动方式，并调用对应的初始化函数来设置分布式训练环境。

        Args:
            launcher (str): 启动方式，支持 'pytorch' 或 'slurm'。
                - 'pytorch': 使用 PyTorch 自带的分布式训练。
                - 'slurm': 使用 SLURM 作业调度系统启动分布式训练。
            backend (str, optional): 分布式训练使用的后端，默认为 'nccl'。
                - 'nccl'：NVIDIA Collective Communications Library，用于多 GPU 训练。
                - 'gloo'：支持 CPU 和 GPU 的分布式训练。
            **kwargs: 其他参数，传递给初始化分布式训练时的设置。

        Raises:
            ValueError: 如果 `launcher` 参数不为 'pytorch' 或 'slurm' 时抛出异常。
        """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    """
        初始化基于 PyTorch 的分布式训练环境。

        设置分布式训练进程的 `rank`，并根据 `rank` 设置当前进程使用的 GPU。
        然后通过 `torch.distributed.init_process_group` 初始化分布式训练。

        Args:
            backend (str): 分布式训练使用的后端，默认为 'nccl'。
                - 'nccl'：适用于多 GPU 环境。
                - 'gloo'：支持 CPU 和 GPU。
            **kwargs: 其他参数，传递给 `dist.init_process_group`。
        """
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """
        初始化基于 SLURM 作业调度系统的分布式训练环境。

        如果传入的端口 `port` 为 None，则会使用环境变量 `MASTER_PORT` 中的端口，如果没有该环境变量，则使用默认的 29500 端口。
        获取 `SLURM_PROCID` 和 `SLURM_NTASKS` 来确定进程编号和总进程数，并通过 `scontrol` 命令获取主节点的地址。

        Args:
            backend (str): 分布式训练使用的后端，默认为 'nccl'。
            port (int, optional): 指定主节点的端口，默认使用 `MASTER_PORT` 环境变量或 29500。
        """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        os.environ['MASTER_PORT'] = '29500'
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    """
    获取分布式训练的进程信息。

    如果分布式环境已经初始化，则返回当前进程的 rank 和总进程数。
    如果没有初始化分布式环境，则返回 rank 为 0 和 total_rank 为 1。

    Returns:
        tuple: 包含当前进程的 rank 和总进程数 world_size。
            - rank (int): 当前进程的编号。
            - world_size (int): 总的进程数。
    """
    if dist.is_available():
        initialized = dist.is_initialized()  # 检查分布式是否可用并初始化
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()  # 获取当前进程的 rank
        world_size = dist.get_world_size()  # 获取总进程数
    else:
        rank = 0  # 如果没有初始化，默认 rank 为 0
        world_size = 1  # 默认总进程数为 1

    return rank, world_size


def master_only(func):
    """
    装饰器，用于确保某个函数只在主进程中执行。

    该装饰器会获取当前进程的 rank，仅当 rank 为 0（主进程）时执行被装饰的函数。
    其他进程会跳过该函数的执行。

    Args:
        func (callable): 要装饰的函数。

    Returns:
        callable: 返回装饰后的函数，只有在主进程（rank == 0）时执行。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()  # 获取当前进程的 rank 和 world_size
        if rank == 0:
            return func(*args, **kwargs)  # 仅主进程执行该函数

    return wrapper
