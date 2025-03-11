from copy import deepcopy

from utils.registry import METRIC_REGISTRY
from .fid_lpips import calculate_lpips, calculate_fid
from .psnr_ssim import calculate_psnr, calculate_ssim

__all__ = ['calculate_metric', 'calculate_psnr', 'calculate_ssim', 'calculate_fid', 'calculate_lpips']


def calculate_metric(data, opt):
    """根据数据和配置计算指标。

    该函数根据输入的配置和数据来选择并计算对应的指标，常用于评估图像质量。

    Args:
        data (dict): 输入的数据，通常包含图像等要计算指标的内容。
        opt (dict): 配置字典，必须包含如下字段：
            type (str): 要计算的指标类型。例如，'psnr'、'ssim' 或 'niqe' 等。

    Returns:
        metric (float): 计算得到的指标值。
    """
    opt = deepcopy(opt)  # 深拷贝 opt 配置，避免修改原始配置
    metric_type = opt.pop('type')  # 从配置中弹出 'type' 字段，确定要计算的指标类型

    # 根据指标类型从 METRIC_REGISTRY 中获取对应的指标计算函数
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)

    return metric
