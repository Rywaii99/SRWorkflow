import functools
import torch
from torch.nn import functional as F


def reduce_loss(loss, reduction):
    """根据指定方式进行损失缩减。

    Args:
        loss (Tensor): 每个元素的损失张量。
        reduction (str): 缩减方式，选项有 'none'、'mean' 和 'sum'。

    Returns:
        Tensor: 缩减后的损失张量。
    """
    # 获取缩减方式对应的枚举值（0: 'none', 1: 'mean', 2: 'sum'）
    reduction_enum = F._Reduction.get_enum(reduction)

    if reduction_enum == 0:
        # 如果是 'none'，不做任何缩减，直接返回原始损失
        return loss
    elif reduction_enum == 1:
        # 如果是 'mean'，计算损失的均值
        return loss.mean()
    else:
        # 如果是 'sum'，计算损失的总和
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """应用逐元素权重并缩减损失。

    Args:
        loss (Tensor): 每个元素的损失。
        weight (Tensor): 每个元素的权重，默认是 None。
        reduction (str): 缩减方式，选项有 'none'、'mean' 和 'sum'，默认是 'mean'。

    Returns:
        Tensor: 缩减后的损失值。
    """
    # 如果指定了权重，应用逐元素权重
    if weight is not None:
        # 确保权重张量的维度与损失张量的维度一致
        assert weight.dim() == loss.dim()
        # 权重的通道数可以是 1，或者与损失张量的通道数相同
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        # 将损失与权重逐元素相乘
        loss = loss * weight

    # 如果没有指定权重或缩减方式是 'sum'，直接对损失进行缩减
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # 如果缩减方式是 'mean'，则在有权重的情况下计算加权平均
    elif reduction == 'mean':
        if weight.size(1) > 1:
            # 如果权重的通道数大于 1，求权重的总和
            weight = weight.sum()
        else:
            # 如果权重的通道数等于 1，按通道数进行调整
            weight = weight.sum() * loss.size(1)
        # 使用权重进行加权平均
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """为给定的损失函数创建一个加权版本的装饰器。

    使用此装饰器时，损失函数必须具有如下签名：
    `loss_func(pred, target, **kwargs)`。该函数只需要计算逐元素损失，而无需进行缩减操作。
    该装饰器会将权重和缩减方式参数添加到损失函数中。
    装饰后的函数签名变为：
    `loss_func(pred, target, weight=None, reduction='mean', **kwargs)`。

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # 计算逐元素损失
        loss = loss_func(pred, target, **kwargs)
        # 根据权重和缩减方式处理损失
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


def get_local_weights(residual, ksize):
    """为生成 LDL 的伪影图计算局部权重。

    该函数只会被 `get_refined_artifact_map` 函数调用。

    Args:
        residual (Tensor): 预测图像与真实图像之间的残差。
        ksize (Int): 局部窗口的大小。

    Returns:
        Tensor: 每个像素的权重，用于判定该像素是否为伪影像素。
    """

    # 计算填充的大小
    pad = (ksize - 1) // 2
    # 对残差进行反射填充
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    # 使用展开操作来获取每个局部窗口中的所有像素
    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    # 计算每个局部窗口的方差，作为局部权重
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):
    """计算 LDL 的伪影图（伪影/细节：通过局部判别学习方法生成的超分辨率图像伪影映射）。

    该方法基于论文 "A Locally Discriminative Learning Approach to Realistic Image Super-Resolution" 中提出的方法。

    Args:
        img_gt (Tensor): 真实图像。
        img_output (Tensor): 优化模型生成的输出图像。
        img_ema (Tensor): EMA 模型生成的输出图像。
        ksize (Int): 局部窗口的大小。

    Returns:
        overall_weight (Tensor): 每个像素的权重，用于判定该像素是否为伪影像素（基于局部和全局观察计算得到的伪影权重）。
    """

    # 计算 EMA 图像和真实图像之间的残差
    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    # 计算超分辨率图像和真实图像之间的残差
    residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    # 基于超分辨率图像的残差计算块级权重（方差的 1/5 次方）
    patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True) ** (1 / 5)
    # 获取局部权重
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    # 综合块级权重和像素级权重
    overall_weight = patch_level_weight * pixel_level_weight

    # 如果超分辨率残差小于 EMA 残差，则将该像素的伪影权重设置为 0
    overall_weight[residual_sr < residual_ema] = 0

    return overall_weight
