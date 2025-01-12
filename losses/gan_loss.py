import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """定义 GAN 损失函数。

    Args:
        gan_type (str): 支持 'vanilla'、'lsgan'、'wgan'、'hinge' 类型的 GAN 损失。
        real_label_val (float): 真实标签的值。默认值为 1.0。
        fake_label_val (float): 假标签的值。默认值为 0.0。
        loss_weight (float): 损失权重。默认值为 1.0。
            注意，loss_weight 仅用于生成器，对于判别器始终为 1.0。
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        """初始化 GAN 损失类。

        Args:
            gan_type (str): GAN 类型，支持 'vanilla'、'lsgan'、'wgan'、'hinge'。
            real_label_val (float): 真实标签的值，默认为 1.0。
            fake_label_val (float): 假标签的值，默认为 0.0。
            loss_weight (float): 损失权重，默认为 1.0。
        """
        super(GANLoss, self).__init__()
        self.gan_type = gan_type  # GAN 类型
        self.loss_weight = loss_weight  # 损失权重
        self.real_label_val = real_label_val  # 真实标签的值
        self.fake_label_val = fake_label_val  # 假标签的值

        # 根据不同的 GAN 类型选择相应的损失函数
        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()  # vanilla GAN 使用二进制交叉熵损失
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()  # LSGAN 使用均方误差损失
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss  # WGAN 使用自定义的 WGAN 损失函数
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss  # WGAN Softplus 使用自定义的 Softplus 损失函数
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()  # hinge loss 使用 ReLU 激活函数
        else:
            raise NotImplementedError(f'GAN 类型 {self.gan_type} 尚未实现。')

    def _wgan_loss(self, input, target):
        """WGAN 损失。

        Args:
            input (Tensor): 输入张量，通常是判别器的输出。
            target (bool): 目标标签（是否为真实标签）。

        Returns:
            Tensor: WGAN 损失值。
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """WGAN Softplus 损失，Softplus 是 ReLU 函数的平滑近似。

        在 StyleGAN2 中：
            - 判别器的逻辑损失；
            - 生成器的非饱和损失。

        Args:
            input (Tensor): 输入张量，通常是判别器的输出。
            target (bool): 目标标签（是否为真实标签）。

        Returns:
            Tensor: WGAN Softplus 损失值。
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """获取目标标签。

        根据 GAN 类型和目标是否为真实标签，返回相应的标签值。

        Args:
            input (Tensor): 输入张量，通常是判别器的输出。
            target_is_real (bool): 是否是来自真实图像的标签（真实标签为 `True`，假标签为 `False`）。

        Returns:
            (bool | Tensor): 目标标签。如果是 WGAN 或 WGAN Softplus，则返回布尔值，否则返回 Tensor。
        """
        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real  # 对于 WGAN 和 WGAN Softplus，直接返回布尔值
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val  # 否则返回一个张量，所有元素的值为目标标签

    def forward(self, input, target_is_real, is_disc=False):
        """前向传播函数，计算损失。

        Args:
            input (Tensor): 损失模块的输入，即网络的输出。
            target_is_real (bool): 目标是否为真实图像。
            is_disc (bool): 是否计算判别器的损失。如果为 `True`，则计算判别器的损失；否则计算生成器的损失。

        Returns:
            Tensor: 计算出的 GAN 损失值。
        """
        target_label = self.get_target_label(input, target_is_real)  # 获取目标标签
        if self.gan_type == 'hinge':
            if is_disc:  # 对于 hinge-GAN 中的判别器
                input = -input if target_is_real else input  # 真实标签时取负输入
                loss = self.loss(1 + input).mean()  # 对于判别器，损失是 `1 + input` 的均值
            else:  # 对于 hinge-GAN 中的生成器
                loss = -input.mean()  # 对生成器，损失是输入的负均值
        else:  # 对于其他 GAN 类型
            loss = self.loss(input, target_label)  # 计算损失

        # 对于判别器，loss_weight 始终为 1.0
        return loss if is_disc else loss * self.loss_weight  # 如果是生成器，乘以损失权重


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    多尺度 GAN 损失类，接收多个预测结果的列表。

    该类用于计算多尺度 GAN 损失，适用于像 StyleGAN 这种多层次、多尺度的网络。
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        """初始化多尺度 GAN 损失类。

        Args:
            gan_type (str): GAN 类型，支持 'vanilla'、'lsgan'、'wgan'、'hinge' 等。
            real_label_val (float): 真实标签的值，默认为 1.0。
            fake_label_val (float): 假标签的值，默认为 0.0。
            loss_weight (float): 损失权重，默认为 1.0。
        """
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """前向传播函数，计算多尺度 GAN 损失。

        Args:
            input (list or list of lists): 输入为张量列表或张量列表的列表。
                如果是多尺度特征匹配，`input` 可以是一个包含多个层次输出的列表。
            target_is_real (bool): 目标是否为真实数据。
            is_disc (bool): 是否计算判别器的损失。如果为 `True`，则计算判别器损失，否则计算生成器损失。

        Returns:
            Tensor: 计算后的损失值。
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # 如果是多层次特征匹配，仅计算最后一层的 GAN 损失
                    pred_i = pred_i[-1]
                # 使用父类的 forward 函数计算损失，并取均值
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            # 返回所有层的平均损失
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 正则化，用于判别器的梯度惩罚。

    核心思想是：仅对真实数据的梯度进行惩罚。在生成器生成的真实数据分布与真实数据分布一致时，
    判别器对数据流形的梯度应该接近 0。梯度惩罚保证判别器无法创建与数据流形垂直的非零梯度，
    否则在 GAN 游戏中会受到损失。

    参考文献：《Which training methods for GANs do actually converge?》

    Args:
        real_pred (Tensor): 判别器对真实数据的预测输出。
        real_img (Tensor): 真实图像数据。

    Returns:
        Tensor: R1 正则化惩罚项。
    """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    """生成器路径正则化，用于控制生成器潜在空间路径的长度。

    通过路径长度正则化来控制生成器潜在空间的路径长度，避免路径在潜在空间中变化过大。
    该正则化在生成器的训练中起到稳定作用，尤其是在 StyleGAN2 中得到了广泛应用。

    Args:
        fake_img (Tensor): 生成器生成的假图像。
        latents (Tensor): 潜在空间向量。
        mean_path_length (float): 平均路径长度，用于平衡路径长度正则化。
        decay (float): 衰减系数，控制路径长度的变化速度。

    Returns:
        path_penalty (Tensor): 路径长度正则化的损失值。
        path_lengths_mean (Tensor): 路径长度的平均值。
        path_mean (Tensor): 经过衰减后的路径长度目标值。
    """
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    # 平均路径长度目标值经过衰减更新
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    # 路径长度正则化惩罚项
    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """计算 WGAN-GP 的梯度惩罚损失。

    该函数实现了 WGAN-GP 的梯度惩罚，用于稳定训练过程，尤其是当数据分布差异较大时。
    梯度惩罚的核心思想是，通过在生成数据和真实数据之间进行插值，迫使判别器的梯度约束在 [0, 1] 之间。

    Args:
        discriminator (nn.Module): 判别器网络。
        real_data (Tensor): 真实输入数据。
        fake_data (Tensor): 生成的假数据。
        weight (Tensor, optional): 权重张量，用于加权梯度惩罚。默认为 `None`。

    Returns:
        Tensor: 计算出的梯度惩罚损失。
    """
    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # 在真实数据和假数据之间进行插值
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # 计算插值数据的判别器输出
    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight  # 如果提供了权重，应用权重

    # 计算梯度惩罚，归一化梯度后计算平方误差
    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)  # 如果有权重，做归一化

    return gradients_penalty
