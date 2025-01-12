import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir
from utils.registry import LOSS_REGISTRY
from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

__all__ = ['build_loss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']

# 自动扫描并导入损失模块到注册表中
# 扫描 'losses' 文件夹下所有的文件，收集以 '_loss.py' 结尾的文件
loss_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# scandir 函数扫描目录，获取以 '_loss.py' 结尾的文件名，并去掉扩展名

# 导入所有的损失模块
_model_modules = [importlib.import_module(f'losses.{file_name}') for file_name in loss_filenames]


# 通过动态导入所有符合条件的损失模块，模块路径为 'losses.<文件名>'

def build_loss(opt):
    """根据配置选项构建损失函数。

    Args:
        opt (dict): 配置字典。必须包含：
            type (str): 损失函数类型。

    Returns:
        loss (nn.Module): 构建的损失函数对象。
    """
    # 复制传入的配置字典，避免修改原始配置
    opt = deepcopy(opt)

    # 从配置中弹出 'type' 键，获取损失函数的类型（字符串）
    loss_type = opt.pop('type')

    # 从注册表中获取对应类型的损失函数并传入配置参数
    loss = LOSS_REGISTRY.get(loss_type)(**opt)

    # 获取日志记录器
    logger = get_root_logger()

    # 记录创建的损失函数信息
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')

    # 返回构建好的损失函数
    return loss