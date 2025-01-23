"""
修改自 https://github.com/yulunzhang/RCAN
"""

from archs.arch_util import Upsample, MeanShift, default_conv
import torch.nn as nn
from utils.registry import ARCH_REGISTRY


def make_model(args, parent=False):
    """
    创建并返回RCAN模型实例。

    Args:
        args: 包含模型配置的参数对象。
        parent: 是否作为父模型调用（默认为False）。

    Returns:
        RCAN模型实例。
    """
    return RCAN(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    """
    通道注意力层（Channel Attention Layer），用于增强特征图中重要通道的响应。
    """

    def __init__(self, channel, reduction=16):
        """
        初始化通道注意力层。

        Args:
            channel: 输入特征图的通道数。
            reduction: 通道缩减比例（默认为16）。
        """
        super(CALayer, self).__init__()
        # 全局平均池化：将特征图从空间维度压缩到1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力机制：通过两个卷积层计算通道权重
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入特征图。

        Returns:
            加权后的特征图。
        """
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # 将通道权重应用于输入特征图


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    """
    残差通道注意力块（Residual Channel Attention Block），结合了残差连接和通道注意力机制。
    """

    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        初始化残差通道注意力块。

        Args:
            conv: 卷积层函数。
            n_feat: 特征图的通道数。
            kernel_size: 卷积核大小。
            reduction: 通道缩减比例。
            bias: 是否使用偏置（默认为True）。
            bn: 是否使用批量归一化（默认为False）。
            act: 激活函数（默认为ReLU）。
            res_scale: 残差缩放因子（默认为1）。
        """
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入特征图。

        Returns:
            经过残差连接和通道注意力机制处理后的特征图。
        """
        res = self.body(x)
        res += x  # 残差连接
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    """
    残差组（Residual Group），由多个残差通道注意力块组成。
    """

    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        """
        初始化残差组。

        Args:
            conv: 卷积层函数。
            n_feat: 特征图的通道数。
            kernel_size: 卷积核大小。
            reduction: 通道缩减比例。
            act: 激活函数。
            res_scale: 残差缩放因子。
            n_resblocks: 残差块的数量。
        """
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入特征图。

        Returns:
            经过残差组处理后的特征图。
        """
        res = self.body(x)
        res += x  # 残差连接
        return res


## Residual Channel Attention Network (RCAN)
@ARCH_REGISTRY.register()
class RCAN(nn.Module):
    """
    残差通道注意力网络（Residual Channel Attention Network），用于图像超分辨率任务。
    """

    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=64, scale=2, reduction=16, conv=default_conv):
        """
        初始化RCAN模型。

        Args:
            args: 包含模型配置的参数对象。
            conv: 卷积层函数（默认为common.default_conv）。
        """
        super(RCAN, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)
        n_colors = 3
        rgb_range = 255
        res_scale = 1

        # # RGB mean for DIV2K
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)
        # self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        # RGB mean for AID
        # rgb_mean = (0.4019, 0.4120, 0.3700)
        # rgb_std = (0.1570, 0.1447, 0.1400)
        # self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)

        # 定义头部模块
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # 定义主体模块
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # 定义尾部模块
        modules_tail = [
            Upsample(scale, n_feats),
            conv(n_feats, n_colors, kernel_size)]

        # self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入图像。

        Returns:
            超分辨率重建后的图像。
        """
        # x = self.sub_mean(x)  # 减去均值
        x = self.head(x)  # 头部卷积

        res = self.body(x)  # 主体残差组
        res += x  # 残差连接

        x = self.tail(res)  # 尾部上采样和卷积
        # x = self.add_mean(x)  # 加上均值

        return x

    def load_state_dict(self, state_dict, strict=False):
        """
        加载模型参数。

        Args:
            state_dict: 包含模型参数的字典。
            strict: 是否严格匹配参数（默认为False）。
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
