import datetime
import logging
import time

from .dist_util import get_dist_info, master_only

initialized_logger = {}


class AvgTimer():
    def __init__(self, window=200):
        """
        初始化AvgTimer类，设置窗口大小，并初始化时间记录相关的参数。

        Args:
            window (int): 用于计算平均时间的窗口大小，默认是200。
        """
        self.window = window  # 平均时间的窗口大小
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        """
        启动定时器，记录当前时间戳作为开始时间。
        """
        self.start_time = self.tic = time.time()

    def record(self):
        """
        记录当前的时间，并计算平均时间。如果计数超过窗口大小，则重置累计时间。
        """
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        # 计算平均时间
        self.avg_time = self.total_time / self.count

        # 如果计数超过窗口大小，重置
        if self.count > self.window:
            self.count = 0
            self.total_time = 0

        self.tic = time.time()

    def get_current_time(self):
        """
        获取当前迭代的时间。

        Returns:
            float: 当前迭代的时间。
        """
        return self.current_time

    def get_avg_time(self):
        """
        获取平均时间。

        Returns:
            float: 平均时间。
        """
        return self.avg_time


class MessageLogger():
    def __init__(self, opt, start_iter=1, tb_logger=None):
        """
        初始化MessageLogger类，配置日志记录的相关参数。

        Args:
            opt (dict): 配置字典，包括实验名称、日志记录频率、训练参数等。
            start_iter (int): 开始的迭代次数，默认值为1。
            tb_logger (obj): TensorBoard日志记录对象，默认值为None。
        """
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = opt['train']['total_iter']
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    def reset_start_time(self):
        """
        重置开始时间，通常在每个epoch开始时调用。
        """
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars):
        """
        格式化并打印日志消息。

        Args:
            log_vars (dict): 包含训练过程中的变量，通常包括epoch、iter、学习率、时间、损失等。
        """
        # 获取epoch，iter，学习率
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')

        # 格式化日志信息
        message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:(')
        for v in lrs:
            message += f'{v:.6f},'
        message += ')] '

        # 计算并显示时间和预计剩余时间
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')

            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

        # 记录损失等其他变量
        for k, v in log_vars.items():
            message += f'{k}: {v:.7f} '
            # 使用TensorBoard记录损失等信息
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    """
    初始化 TensorBoard 记录器。

    Args:
        log_dir (str): TensorBoard 日志保存的目录路径。

    Returns:
        tb_logger (obj): 初始化的 TensorBoard 记录器。
    """
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """
    初始化 wandb 记录器，用于同步 TensorBoard 日志。

    Args:
        opt (dict): 配置字典，包含wandb的项目名称和其他设置。

    """
    import wandb
    logger = get_root_logger()

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(id=wandb_id, resume=resume, name=opt['name'], config=opt, project=project, sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='srworkflow', log_level=logging.INFO, log_file=None):
    """
    获取根日志记录器，初始化日志处理器并设置日志级别。

    Args:
        logger_name (str): 根日志记录器的名称，默认是 'srworkflow'。
        log_file (str | None): 日志文件名，如果指定，则将日志写入文件。
        log_level (int): 日志级别，默认是 INFO。

    Returns:
        logging.Logger: 根日志记录器。
    """
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger


def get_env_info():
    """
    获取当前环境的信息，包括软件版本。

    Returns:
        str: 环境信息的字符串，包括 BasicSR、PyTorch 和 TorchVision 的版本。
    """
    import torch
    import torchvision

    msg = r"""
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}'
            f'\n\tCUDA Version: {torch.version.cuda}'
            f'\n\tcuDNN Version: {torch.backends.cudnn.version()}')

    return msg
