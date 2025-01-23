import torch
from torch import nn as nn
from torch.nn import functional as F

from archs.VGG_arch import VGGFeatureExtractor
from utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    """计算 L1 损失（平均绝对误差）。

    Args:
        pred (Tensor): 预测的张量。
        target (Tensor): 真实的目标张量。

    Returns:
        Tensor: 逐元素的 L1 损失。
    """
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    """计算 MSE 损失（均方误差）。

    Args:
        pred (Tensor): 预测的张量。
        target (Tensor): 真实的目标张量。

    Returns:
        Tensor: 逐元素的 MSE 损失。
    """
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    """计算 Charbonnier 损失，常用于鲁棒 L1 损失，具有差异化的 L1 损失。

    Args:
        pred (Tensor): 预测的张量。
        target (Tensor): 真实的目标张量。
        eps (float): 一个用于控制零附近曲率的值。默认值为 1e-12。

    Returns:
        Tensor: Charbonnier 损失。
    """
    return torch.sqrt((pred - target )**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 损失（平均绝对误差，MAE）。

    Args:
        loss_weight (float): L1 损失的权重。默认值为 1.0。
        reduction (str): 指定输出结果的缩减方式。
            支持的选项是 'none' | 'mean' | 'sum'。默认值为 'mean'。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'不支持的缩减方式：{reduction}。支持的方式是: none, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """计算前向传播的 L1 损失。

        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            target (Tensor): 真实的目标张量，形状为 (N, C, H, W)。
            weight (Tensor, 可选): 逐元素权重，形状为 (N, C, H, W)，默认值为 None。

        Returns:
            Tensor: 计算得到的 L1 损失。
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE 损失（均方误差，L2 损失）。

    Args:
        loss_weight (float): MSE 损失的权重。默认值为 1.0。
        reduction (str): 指定输出结果的缩减方式。
            支持的选项是 'none' | 'mean' | 'sum'。默认值为 'mean'。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'不支持的缩减方式：{reduction}。支持的方式是: none, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """计算前向传播的 MSE 损失。

        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            target (Tensor): 真实的目标张量，形状为 (N, C, H, W)。
            weight (Tensor, 可选): 逐元素权重，形状为 (N, C, H, W)，默认值为 None。

        Returns:
            Tensor: 计算得到的 MSE 损失。
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier 损失（L1 损失的一个鲁棒变种，是 L1 损失的一个可微版本）。

    该方法在论文 "Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution" 中有所描述。

    参数：
        loss_weight (float): L1 损失的权重。默认值为 1.0。
        reduction (str): 指定输出结果的缩减方式。
            支持的选项是 'none' | 'mean' | 'sum'。默认值为 'mean'。
        eps (float): 一个控制零附近曲率的值。默认值为 1e-12。
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'不支持的缩减方式：{reduction}。支持的方式是: none, mean, sum')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """计算前向传播的 Charbonnier 损失。

        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            target (Tensor): 真实的目标张量，形状为 (N, C, H, W)。
            weight (Tensor, 可选): 逐元素权重，形状为 (N, C, H, W)，默认值为 None。

        Returns:
            Tensor: 计算得到的 Charbonnier 损失。
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """加权的总变差（TV）损失。

    参数：
        loss_weight (float): 损失的权重。默认值为 1.0。
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'不支持的缩减方式：{reduction}。支持的方式是: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        """计算前向传播的加权 TV 损失。

        Args:
            pred (Tensor): 预测的张量，形状为 (N, C, H, W)。
            weight (Tensor, 可选): 逐元素权重，形状为 (N, C, H, W)，默认值为 None。

        Returns:
            Tensor: 计算得到的加权 TV 损失。
        """
        if weight is None:
            # 如果没有指定权重，y 和 x 方向上的权重都为 None
            y_weight = None
            x_weight = None
        else:
            # 如果有权重，则在 y 和 x 方向上切分权重
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        # 计算 y 方向上的损失（纵向差异）
        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        # 计算 x 方向上的损失（横向差异）
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        # 返回 y 和 x 方向的总损失
        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """感知损失与常见的风格损失结合使用。

    Args:
        layer_weights (dict): 每个VGG特征层的权重。以下是一个例子：
            {'conv5_4': 1.}，表示在计算损失时，`conv5_4` 层（在 `relu5_4` 之前）的特征提取权重为 1.0。
        vgg_type (str): 用作特征提取器的 VGG 网络类型。默认是 'vgg19'。
        use_input_norm (bool): 如果为 `True`，则在 VGG 中对输入图像进行归一化。默认值为 `True`。
        range_norm (bool): 如果为 `True`，则将图像的范围从 [-1, 1] 归一化到 [0, 1]。默认值为 `False`。
        perceptual_weight (float): 如果 `perceptual_weight > 0`，则会计算感知损失，并将损失乘以权重。默认值为 1.0。
        style_weight (float): 如果 `style_weight > 0`，则会计算风格损失，并将损失乘以权重。默认值为 0.0。
        criterion (str): 用于感知损失的标准，支持 'l1', 'l2' 和 'fro'。默认值为 'l1'。
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        """初始化 PerceptualLoss 类。

        Args:
            layer_weights (dict): 每个VGG层的权重字典。
            vgg_type (str): VGG网络的类型，默认为 'vgg19'。
            use_input_norm (bool): 是否对输入图像进行归一化，默认为 True。
            range_norm (bool): 是否将图像范围从 [-1, 1] 转换到 [0, 1]，默认为 False。
            perceptual_weight (float): 感知损失的权重，默认为 1.0。
            style_weight (float): 风格损失的权重，默认为 0.0。
            criterion (str): 损失标准，'l1', 'l2' 或 'fro'，默认为 'l1'。
        """
        super(PerceptualLoss, self).__init__()

        # 设置损失权重
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights

        # 创建 VGG 特征提取器
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),  # 使用的层名称列表
            vgg_type=vgg_type,  # 使用的 VGG 类型（'vgg19' 或其他）
            use_input_norm=use_input_norm,  # 是否使用输入图像的归一化
            range_norm=range_norm)  # 是否对图像范围进行归一化

        # 设置感知损失的标准
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()  # L1 损失
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()  # L2 损失
        elif self.criterion_type == 'fro':
            self.criterion = None  # Frobenius 范数损失，不使用具体损失函数
        else:
            raise NotImplementedError(f'{criterion} 标准尚不支持。')

    def forward(self, x, gt):
        """前向传播函数。

        Args:
            x (Tensor): 输入的张量，形状为 (n, c, h, w)，n 是批次大小，c 是通道数，h 和 w 是图像的高和宽。
            gt (Tensor): 真实目标的张量，形状与输入张量相同。

        Returns:
            Tensor: 感知损失和风格损失的元组 (percep_loss, style_loss)。
        """
        # 提取 VGG 特征
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())  # 使用 `.detach()` 以避免梯度传递到真实图像

        # 计算感知损失
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                # 如果使用 Frobenius 范数计算损失
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight  # 按照权重缩放感知损失
        else:
            percep_loss = None  # 如果不计算感知损失，则设为 None

        # 计算风格损失
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                # 如果使用 Frobenius 范数计算风格损失
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight  # 按照权重缩放风格损失
        else:
            style_loss = None  # 如果不计算风格损失，则设为 None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """计算 Gram 矩阵。

        Args:
            x (torch.Tensor): 形状为 (n, c, h, w) 的张量。

        Returns:
            torch.Tensor: 计算得到的 Gram 矩阵。
        """
        n, c, h, w = x.size()  # 获取张量的尺寸
        features = x.view(n, c, w * h)  # 将张量重新排列为 (n, c, h * w) 形状
        features_t = features.transpose(1, 2)  # 转置得到 (n, h * w, c) 形状
        gram = features.bmm(features_t) / (c * h * w)  # 计算 Gram 矩阵，并进行归一化
        return gram
