import torch
from torch import nn as nn
from torch.nn import functional as F

from utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """
    残差密集块（Residual Dense Block）。
    用于ESRGAN中的RRDB块。

    Args:
        num_feat (int): 中间特征的通道数。
        num_grow_ch (int): 每次增长的通道数。
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        # 第一个卷积层，输入通道数为num_feat，输出通道数为num_grow_ch
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        # 第二个卷积层，输入通道数为num_feat + num_grow_ch，输出通道数为num_grow_ch
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        # 第三个卷积层，输入通道数为num_feat + 2 * num_grow_ch，输出通道数为num_grow_ch
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # 第四个卷积层，输入通道数为num_feat + 3 * num_grow_ch，输出通道数为num_grow_ch
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        # 第五个卷积层，输入通道数为num_feat + 4 * num_grow_ch，输出通道数为num_feat
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 初始化卷积层的权重
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, num_feat, H, W)

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_feat, H, W)
        """
        # 经过第一个卷积层和LeakyReLU激活函数
        x1 = self.lrelu(self.conv1(x))  # 形状: (B, num_grow_ch, H, W)
        # 将输入x和x1在通道维度上拼接，然后经过第二个卷积层和LeakyReLU激活函数
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))  # 形状: (B, num_grow_ch, H, W)
        # 将输入x、x1和x2在通道维度上拼接，然后经过第三个卷积层和LeakyReLU激活函数
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))  # 形状: (B, num_grow_ch, H, W)
        # 将输入x、x1、x2和x3在通道维度上拼接，然后经过第四个卷积层和LeakyReLU激活函数
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))  # 形状: (B, num_grow_ch, H, W)
        # 将输入x、x1、x2、x3和x4在通道维度上拼接，然后经过第五个卷积层
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))  # 形状: (B, num_feat, H, W)
        # 经验上，使用0.2来缩放残差以获得更好的性能
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    残差嵌套残差密集块（Residual in Residual Dense Block）。
    用于ESRGAN中的RRDB-Net。

    Args:
        num_feat (int): 中间特征的通道数。
        num_grow_ch (int): 每次增长的通道数。
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        # 定义三个残差密集块
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, num_feat, H, W)

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_feat, H, W)
        """
        # 经过第一个残差密集块
        out = self.rdb1(x)  # 形状: (B, num_feat, H, W)
        # 经过第二个残差密集块
        out = self.rdb2(out)  # 形状: (B, num_feat, H, W)
        # 经过第三个残差密集块
        out = self.rdb3(out)  # 形状: (B, num_feat, H, W)
        # 经验上，使用0.2来缩放残差以获得更好的性能
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """
    由残差嵌套残差密集块组成的网络，用于ESRGAN。
    ESRGAN: 增强型超分辨率生成对抗网络。
    我们将ESRGAN扩展到x2和x1的缩放比例。
    注意: 这是RRDBNet中缩放比例为1和2的一种选择。
    我们首先使用像素反洗牌（pixel-unshuffle，像素洗牌的逆操作）来减小空间尺寸并增大通道尺寸，然后将输入送入主ESRGAN架构。

    Args:
        num_in_ch (int): 输入的通道数。
        num_out_ch (int): 输出的通道数。
        scale (int): 缩放比例，默认为4。
        num_feat (int): 中间特征的通道数，默认为64。
        num_block (int): 主干网络中的块数，默认为23。
        num_grow_ch (int): 每次增长的通道数，默认为32。
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        # 如果缩放比例为2，将输入通道数乘以4
        if scale == 2:
            num_in_ch = num_in_ch * 4
        # 如果缩放比例为1，将输入通道数乘以16
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        # 第一个卷积层，输入通道数为num_in_ch，输出通道数为num_feat
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # 主干网络，由num_block个RRDB块组成
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        # 主干网络后的卷积层，输入通道数为num_feat，输出通道数为num_feat
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 上采样部分的卷积层
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 高分辨率卷积层，输入通道数为num_feat，输出通道数为num_feat
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # 最后一个卷积层，输入通道数为num_feat，输出通道数为num_out_ch
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # 定义LeakyReLU激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量，形状为 (B, num_in_ch, H, W)

        Returns:
            torch.Tensor: 输出张量，形状为 (B, num_out_ch, H * scale, W * scale)
        """
        # 如果缩放比例为2，进行像素反洗牌操作，将空间尺寸缩小为原来的1/2，通道数扩大为原来的4倍
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)  # 形状: (B, num_in_ch * 4, H // 2, W // 2)
        # 如果缩放比例为1，进行像素反洗牌操作，将空间尺寸缩小为原来的1/4，通道数扩大为原来的16倍
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)  # 形状: (B, num_in_ch * 16, H // 4, W // 4)
        else:
            feat = x
        # 经过第一个卷积层
        feat = self.conv_first(feat)  # 形状: (B, num_feat, H', W')，H'和W'取决于前面的缩放操作
        # 经过主干网络和主干网络后的卷积层
        body_feat = self.conv_body(self.body(feat))  # 形状: (B, num_feat, H', W')
        # 残差连接
        feat = feat + body_feat  # 形状: (B, num_feat, H', W')
        # 第一次上采样，使用最近邻插值将空间尺寸扩大为原来的2倍
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))  # 形状: (B, num_feat, H' * 2, W' * 2)
        # 第二次上采样，使用最近邻插值将空间尺寸扩大为原来的2倍
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))  # 形状: (B, num_feat, H' * 4, W' * 4)
        # 经过高分辨率卷积层和LeakyReLU激活函数
        out = self.lrelu(self.conv_hr(feat))  # 形状: (B, num_feat, H' * 4, W' * 4)
        # 经过最后一个卷积层
        out = self.conv_last(out)  # 形状: (B, num_out_ch, H' * 4, W' * 4)
        return out
