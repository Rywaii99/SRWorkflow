import math
import torch
from torch import nn as nn
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """初始化网络的权重。

    Args:
        module_list (list[nn.Module] | nn.Module): 需要初始化的网络模块，可以是单个模块或模块列表。
        scale (float): 权重初始化的缩放因子，尤其适用于残差块。默认值：1。
        bias_fill (float): 偏置初始化的填充值，默认值：0。
        kwargs (dict): 其他初始化函数的参数。
    """
    if not isinstance(module_list, list):
        module_list = [module_list] # 如果 module_list 不是一个列表（即只有一个模块），则将其转换为列表，统一处理。
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)  # 使用 Kaiming 正态分布初始化卷积层的权重
                m.weight.data *= scale  # 对权重进行缩放
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # 填充偏置
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)  # 使用 Kaiming 正态分布初始化全连接层的权重
                m.weight.data *= scale  # 对权重进行缩放
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # 填充偏置
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)  # 批归一化层的权重初始化为 1
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)  # 填充偏置


def make_layer(basic_block, num_basic_block, **kwarg):
    """通过堆叠相同的基本模块（block）来构建网络层。

    Args:
        basic_block (nn.module): 基本模块的类（例如，卷积块、残差块等）。
        num_basic_block (int): 堆叠的基本模块的数量。
        **kwarg (dict): 传递给每个基本模块的额外参数。

    Returns:
        nn.Sequential: 堆叠的基本模块，返回一个 `nn.Sequential` 容器。
    """
    layers = []  # 初始化一个空列表，用于存储堆叠的模块
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))  # 按照 num_basic_block 的数量，将基本模块添加到 layers 列表中
    return nn.Sequential(*layers)  # 将 layers 列表中的模块包装成 nn.Sequential 并返回


class ResidualBlockNoBN(nn.Module):
    """
    不带 Batch Normalization 的残差块。

    该模块实现了一个经典的残差块，但没有使用 Batch Normalization (BN) 层。

    Args:
        num_feat (int): 中间特征图的通道数。默认值: 64。
        res_scale (float): 残差缩放因子。默认值: 1。该值用于控制残差的影响程度。
        pytorch_init (bool): 如果设置为 True，则使用 PyTorch 默认的初始化方法，
                             否则使用 `default_init_weights` 进行初始化。默认值: False。
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        """
        初始化不带 BN 的残差块。

        Args:
            num_feat (int): 通道数，即特征图的深度。默认值为 64。
            res_scale (float): 残差块中的缩放因子。默认值为 1，用于调节残差的影响。
            pytorch_init (bool): 如果为 True，使用 PyTorch 默认的初始化方法，
                                  否则使用自定义的 `default_init_weights` 初始化。默认值为 False。
        """
        super(ResidualBlockNoBN, self).__init__()

        # 初始化缩放因子
        self.res_scale = res_scale

        # 定义第一个卷积层，输入和输出通道数均为 num_feat，卷积核大小为 3，步幅为 1，填充为 1。
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # 定义第二个卷积层，配置与第一个卷积层相同
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        # 定义激活函数 ReLU（在原地进行计算）
        self.relu = nn.ReLU(inplace=True)

        # 如果 pytorch_init 为 False，使用自定义的初始化方法初始化卷积层权重
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (Tensor): 输入张量，形状通常为 (B, C, H, W)，B为批量大小，C为通道数，H为高度，W为宽度。

        Returns:
            Tensor: 返回残差计算后的结果。其形状与输入 `x` 相同。
        """
        identity = x  # 保存输入的副本，用于后续残差计算

        # 执行卷积操作
        out = self.conv2(self.relu(self.conv1(x)))  # x 通过两次卷积和一次 ReLU 激活

        # 返回输入与经过卷积操作后的输出进行残差相加，乘上残差缩放因子
        return identity + out * self.res_scale


def pixel_unshuffle(x, scale):
    """ 像素反卷积（Pixel Unshuffle）。

    该操作通常用于图像超分辨率任务，目的是将图像中的像素进行“反卷积”或“逆下采样”，
    通过将高分辨率图像的像素按给定的尺度分解为低分辨率图像块的特征。

    Args:
        x (Tensor): 输入特征图，形状为 (b, c, hh, hw)，其中 b 为批量大小，c 为通道数，hh 和 hw 为特征图的高和宽。
        scale (int): 下采样比例。指定如何分解像素，`scale=2` 表示将每个 2x2 的块反卷积成一个通道。

    Returns:
        Tensor: 返回像素反卷积后的特征图，形状为 (b, c * scale^2, h, w)，其中 h 和 w 是反卷积后的图像的高和宽。
    """
    b, c, hh, hw = x.size()  # 获取输入张量的形状，b: 批量大小, c: 通道数, hh: 高度, hw: 宽度

    out_channel = c * (scale**2)  # 输出的通道数等于原始通道数乘以 scale 的平方，因为每个像素会“展开”成多个特征通道

    # 确保图像的高和宽能被 scale 整除，以便进行像素反卷积
    assert hh % scale == 0 and hw % scale == 0

    h = hh // scale  # 计算反卷积后图像的高度
    w = hw // scale  # 计算反卷积后图像的宽度

    # 对输入张量进行重新排列，使得每个像素块（scale x scale）对应一个输出通道
    x_view = x.view(b, c, h, scale, w, scale)  # 将输入张量视图重塑为 (b, c, h, scale, w, scale)

    # 调整维度顺序，将 (scale, scale) 像素块的元素展开并调整顺序
    # 例如，如果 scale=2，输入会从 (b, c, h, 2, w, 2) 重排列为 (b, c, 2, 2, h, w)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)  # 最后重塑为 (b, out_channel, h, w)


class PixelUnshuffle(nn.Module):
    """
    像素反卷积（Pixel Unshuffle）模块，继承自 `nn.Module`。

    该操作通常用于图像下采样任务，将高分辨率图像的特征图按给定的比例分解为低分辨率图像块的特征。
    它是 `PixelShuffle` 的逆操作。

    Args:
        scale (int): 下采样比例。指定如何分解像素，`scale=2` 表示将每个 2x2 的块反卷积成一个通道。

    Shape:
        - 输入: (b, c, hh, hw)，其中 b 是批量大小，c 是通道数，hh 和 hw 是输入特征图的高度和宽度。
        - 输出: (b, c * scale^2, h, w)，其中 h 和 w 是输出特征图的高度和宽度。

    示例:
        >>> pixel_unshuffle = PixelUnshuffle(scale=2)
        >>> x = torch.randn(1, 3, 8, 8)  # 输入形状 (1, 3, 8, 8)
        >>> output = pixel_unshuffle(x)
        >>> print(output.shape)  # 输出形状 (1, 12, 4, 4)
    """
    def __init__(self, scale):
        """
        初始化 PixelUnshuffle 模块。

        Args:
            scale (int): 下采样比例。
        """
        super(PixelUnshuffle, self).__init__()
        self.scale = scale  # 设置下采样比例

    def forward(self, x):
        """
        前向传播函数，执行像素反卷积操作。

        Args:
            x (Tensor): 输入特征图，形状为 (b, c, hh, hw)。

        Returns:
            Tensor: 输出特征图，形状为 (b, c * scale^2, h, w)。
        """
        # 调用 torch.nn.functional.pixel_unshuffle 执行像素反卷积操作
        return F.pixel_unshuffle(x, self.scale)

    def extra_repr(self):
        """
        返回模块的额外信息，用于打印模块时显示。

        Returns:
            str: 模块的额外信息。
        """
        return f'scale={self.scale}'  # 返回下采样比例信息


class Upsample(nn.Sequential):
    """上采样模块 (Upsample module)。

    该模块用于上采样操作，它根据给定的 scale 因子放大输入特征图。支持的上采样比例为 2 的幂次（如 2, 4, 8）和 3。
    PixelShuffle 操作是一种用于图像上采样的技术。它通过将高分辨率的图像块重排为多个低分辨率的图像块，从而实现图像的“像素分解”。

    Args:
        scale (int): 上采样比例。支持的比例包括 2 的幂次（2^n）和 3。
        num_feat (int): 中间特征的通道数。
    """

    def __init__(self, scale, num_feat):
        m = []  # 用于存储子模块的列表

        # 判断 scale 是否为 2 的幂次（2^n），即通过与 (scale - 1) 进行与运算检查是否为幂次
        if (scale & (scale - 1)) == 0:  # 如果 scale 是 2 的幂次
            # 上采样为 2^n 次方
            for _ in range(int(math.log(scale, 2))):  # log(scale, 2) 获取幂次数
                # 每次上采样增加 Conv2d 和 PixelShuffle
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))  # 3x3 卷积，通道数扩展 4 倍
                m.append(nn.PixelShuffle(2))  # PixelShuffle 将 4 个特征通道重排列为 2x2 的特征块
        elif scale == 3:
            # 如果 scale 等于 3，使用类似的方式进行上采样
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))  # 3x3 卷积，通道数扩展 9 倍
            m.append(nn.PixelShuffle(3))  # PixelShuffle 将 9 个特征通道重排列为 3x3 的特征块
        else:
            # 如果 scale 不是 2 的幂次且也不是 3，抛出错误
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')

        # 调用父类的构造函数，初始化 Sequential 容器
        super(Upsample, self).__init__(*m)


class LayerNormFunction(torch.autograd.Function):
    """
    自定义的Layer Normalization实现，继承自torch.autograd.Function。
    该类实现了LayerNorm的前向传播和反向传播逻辑。

    参考文献:
    - Ba, Jimmy, et al. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
    """

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        """
        前向传播：计算输入x的Layer Normalization。

        Args:
            ctx (torch.autograd.FunctionContext): 用于保存计算过程中需要反向传播的变量。
            x (torch.Tensor): 输入的四维张量，形状为(N, C, H, W)，分别表示批量大小、通道数、高度和宽度。
            weight (torch.Tensor): 权重参数，形状为(C,)，通常初始化为1。
            bias (torch.Tensor): 偏置参数，形状为(C,)，通常初始化为0。
            eps (float): 防止除零的一个小常数，默认为1e-6。

        Returns:
            torch.Tensor: 经Layer Normalization后的输出张量，形状与输入相同(N, C, H, W)。
        """
        ctx.eps = eps  # 保存eps，用于反向传播时使用
        N, C, H, W = x.size()  # 获取输入x的形状
        mu = x.mean(1, keepdim=True)  # 计算每个通道的均值
        var = (x - mu).pow(2).mean(1, keepdim=True)  # 计算每个通道的方差
        y = (x - mu) / (var + eps).sqrt()  # 标准化操作 (x - mu) / sqrt(var + eps)
        ctx.save_for_backward(y, var, weight)  # 保存y, var和weight，供反向传播使用

        # 应用缩放和偏移 (weight * y + bias)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y  # 返回Layer Normalization后的结果

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算输入x、weight和bias的梯度。

        Args:
            ctx (torch.autograd.FunctionContext): 保存了前向传播过程中需要反向传播的变量。
            grad_output (torch.Tensor): 来自下一层的梯度，形状与前向传播的输出相同(N, C, H, W)。

        Returns:
            tuple: 包含输入x的梯度、weight的梯度、bias的梯度和eps的梯度。
                - 输入x的梯度：`gx`
                - weight的梯度：`(grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)`
                - bias的梯度：`grad_output.sum(dim=3).sum(dim=2).sum(dim=0)`
                - eps的梯度：`None`，此处未计算
        """
        eps = ctx.eps  # 获取eps，防止除零

        N, C, H, W = grad_output.size()  # 获取grad_output的形状
        y, var, weight = ctx.saved_variables  # 获取前向传播时保存的变量
        g = grad_output * weight.view(1, C, 1, 1)  # 按照权重调整grad_output
        mean_g = g.mean(dim=1, keepdim=True)  # 计算grad_output的均值

        mean_gy = (g * y).mean(dim=1, keepdim=True)  # 计算grad_output与y的加权均值
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)  # 计算x的梯度

        # 返回梯度信息
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """
    LayerNorm2d层：实现了二维输入的Layer Normalization操作。

    参考文献:
    - Ba, Jimmy, et al. "Layer normalization." arXiv preprint arXiv:1607.06450 (2016).
    """

    def __init__(self, channels, eps=1e-6):
        """
        初始化LayerNorm2d层。

        Args:
            channels (int): 输入张量的通道数，即输入x的第二维C。
            eps (float, optional): 防止除零的常数，默认为1e-6。
        """
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))  # 初始化权重为1
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))  # 初始化偏置为0
        self.eps = eps  # 保存eps

    def forward(self, x):
        """
        前向传播：执行Layer Normalization操作。

        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)，表示批量大小、通道数、高度和宽度。

        Returns:
            torch.Tensor: 返回Layer Normalization处理后的输出张量，形状与输入相同(N, C, H, W)。
        """
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# 处理多个输入的Sequential类
class MySequential(nn.Sequential):
    """
    MySequential是一个自定义的顺序容器类，继承自`nn.Sequential`。
    该类重载了`forward`方法，以支持处理多个输入并将它们传递给容器中的每个模块。
    """

    def forward(self, *inputs):
        """
        前向传播方法，处理多个输入。

        该方法会依次将输入传递给 `MySequential` 中的每个模块（层），
        如果输入是元组，则会将每个元素分别传递给模块。
        适用于需要多个输入并进行处理的场景。

        Args:
            *inputs (tuple): 输入数据，可以是一个或多个张量。

        Returns:
            inputs: 经过每个模块处理后的输出。输出形式与输入形式相同，具体取决于模块的行为。
        """
        # 遍历MySequential容器中的每个模块（层）
        for module in self._modules.values():
            # 如果输入是元组类型，意味着有多个输入
            if type(inputs) == tuple:
                # 将多个输入分别传递给当前模块
                inputs = module(*inputs)
            else:
                # 如果只有单个输入，直接传递给模块
                inputs = module(inputs)
        return inputs


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False
