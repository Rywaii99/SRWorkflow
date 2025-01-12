import torch
from torch import nn as nn

from archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class EDSR(nn.Module):
    """EDSR网络结构，增强深度残差网络用于单图像超分辨率。

    论文：Enhanced Deep Residual Networks for Single Image Super-Resolution。
    参考GitHub仓库：https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): 输入图像的通道数。
        num_out_ch (int): 输出图像的通道数。
        num_feat (int): 中间特征图的通道数。默认：64。
        num_block (int): 网络主体中的残差块数量。默认：16。
        upscale (int): 上采样倍数，支持 2^n 和 3。默认：4。
        res_scale (float): 残差块中的残差缩放因子。默认：1。
        img_range (float): 图像的像素范围，默认：255。
        rgb_mean (tuple[float]): RGB均值，用于图像预处理。默认：(0.4488, 0.4371, 0.4040)，从DIV2K数据集计算。
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat=64, num_block=16, upscale=4, res_scale=1, img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EDSR, self).__init__()

        # 图像范围和均值初始化
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # 网络结构初始化
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)  # 第一个卷积层
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat, res_scale=res_scale,
                               pytorch_init=True)  # 残差块
        self.conv_after_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)  # 残差块后的卷积层
        self.upsample = Upsample(upscale, num_feat)  # 上采样模块
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)  # 输出卷积层

    def forward(self, x):
        self.mean = self.mean.type_as(x)  # 将均值转换为与输入相同的数据类型

        # 预处理输入：减去均值并按图像范围进行缩放
        x = (x - self.mean) * self.img_range

        # 通过第一个卷积层
        x = self.conv_first(x)

        # 通过残差块和卷积层处理
        res = self.conv_after_body(self.body(x))
        res += x  # 加入残差

        # 上采样和最后的卷积层
        x = self.conv_last(self.upsample(res))

        # 恢复像素范围并加回均值，得到输出图像
        x = x / self.img_range + self.mean

        return x
