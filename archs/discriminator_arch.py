from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator(nn.Module):
    """VGG风格的判别器，输入大小为128 x 128或256 x 256。

    用于训练SRGAN、ESRGAN和VideoGAN。

    Args:
        num_in_ch (int): 输入的通道数，默认值为3。
        num_feat (int): 基础特征的通道数，默认值为64。
    """

    def __init__(self, num_in_ch, num_feat, input_size=128):
        super(VGGStyleDiscriminator, self).__init__()
        self.input_size = input_size
        # 确保输入尺寸为 128 或 256
        assert self.input_size == 128 or self.input_size == 256, (
            f'input size must be 128 or 256, but received {input_size}')

        # 定义第一层卷积层
        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(num_feat, affine=True)

        # 定义第二层卷积层
        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(num_feat * 2, affine=True)

        # 定义第三层卷积层
        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(num_feat * 4, affine=True)

        # 定义第四层卷积层
        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        # 定义第五层卷积层
        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        # 如果输入尺寸为 256，则定义额外的卷积层
        if self.input_size == 256:
            self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
            self.bn5_0 = nn.BatchNorm2d(num_feat * 8, affine=True)
            self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
            self.bn5_1 = nn.BatchNorm2d(num_feat * 8, affine=True)

        # 定义全连接层
        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # 激活函数
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        # 确保输入尺寸与模型定义的尺寸一致
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        # 第一层卷积
        feat = self.lrelu(self.conv0_0(x))  # 输入大小： (B, C_in, H, W)，输出大小： (B, num_feat, H, W)
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # 输出空间尺寸减半，输出大小： (B, num_feat, H/2, W/2)

        # 第二层卷积
        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))  # 输入大小：(B, num_feat, H/2, W/2)，输出大小：(B, num_feat*2, H/2, W/2)
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # 输出空间尺寸减半，输出大小：(B, num_feat*2, H/4, W/4)

        # 第三层卷积
        feat = self.lrelu(
            self.bn2_0(self.conv2_0(feat)))  # 输入大小：(B, num_feat*2, H/4, W/4)，输出大小：(B, num_feat*4, H/4, W/4)
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # 输出空间尺寸减半，输出大小：(B, num_feat*4, H/8, W/8)

        # 第四层卷积
        feat = self.lrelu(
            self.bn3_0(self.conv3_0(feat)))  # 输入大小：(B, num_feat*4, H/8, W/8)，输出大小：(B, num_feat*8, H/8, W/8)
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # 输出空间尺寸减半，输出大小：(B, num_feat*8, H/16, W/16)

        # 第五层卷积
        feat = self.lrelu(
            self.bn4_0(self.conv4_0(feat)))  # 输入大小：(B, num_feat*8, H/16, W/16)，输出大小：(B, num_feat*8, H/16, W/16)
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # 输出空间尺寸减半，输出大小：(B, num_feat*8, H/32, W/32)

        # 如果输入尺寸为 256，则进行额外的卷积
        if self.input_size == 256:
            feat = self.lrelu(
                self.bn5_0(self.conv5_0(feat)))  # 输入大小：(B, num_feat*8, H/32, W/32)，输出大小：(B, num_feat*8, H/32, W/32)
            feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # 输出空间尺寸减半，输出大小：(B, num_feat*8, H/64, W/64)

        # 将特征图展平：将大小为 (B, num_feat*8, H/64, W/64) 的张量展平为 (B, num_feat*8 * H/64 * W/64)
        feat = feat.view(feat.size(0), -1)  # 输入大小：(B, num_feat*8, H/64, W/64)，输出大小：(B, num_feat*8 * H/64 * W/64)

        # 全连接层
        feat = self.lrelu(self.linear1(feat))  # 输入大小：(B, num_feat*8 * H/64 * W/64)，输出大小：(B, 100)
        out = self.linear2(feat)  # 输入大小：(B, 100)，输出大小：(B, 1)

        return out


# 注册一个名为 UNetDiscriminatorSN 的模型类，并指定后缀为 'basicsr'
@ARCH_REGISTRY.register(suffix='basicsr')
class UNetDiscriminatorSN(nn.Module):
    """定义了一个带有谱归一化(Spectral Normalization)的U-Net判别器。

    用于Real-ESRGAN：使用纯合成数据训练真实世界盲超分辨率。

    Arg:
        num_in_ch (int): 输入的通道数，默认值为3。
        num_feat (int): 基础特征的通道数，默认值为64。
        skip_connection (bool): 是否使用跳跃连接，默认值为True。
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        # 使用 spectral_norm 进行归一化
        norm = spectral_norm
        # 第一层卷积
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # 下采样层
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # 上采样层
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # 额外的卷积层
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # 输入 x 的尺寸: (batch_size, num_in_ch, H, W)

        # 下采样
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        # x0 的尺寸: (batch_size, num_feat, H, W)
        # 第一层卷积 conv0 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        # x1 的尺寸: (batch_size, num_feat * 2, H/2, W/2)
        # 第二层卷积 conv1 使用 kernel_size=4, stride=2, padding=1，将空间尺寸减半

        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        # x2 的尺寸: (batch_size, num_feat * 4, H/4, W/4)
        # 第三层卷积 conv2 使用 kernel_size=4, stride=2, padding=1，将空间尺寸减半

        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)
        # x3 的尺寸: (batch_size, num_feat * 8, H/8, W/8)
        # 第四层卷积 conv3 使用 kernel_size=4, stride=2, padding=1，将空间尺寸减半

        # 上采样
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        # x3 的尺寸: (batch_size, num_feat * 8, H/4, W/4)
        # 使用双线性插值将空间尺寸放大 2 倍

        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        # x4 的尺寸: (batch_size, num_feat * 4, H/4, W/4)
        # 第五层卷积 conv4 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        # 如果使用跳跃连接，则将当前特征与下采样特征相加
        if self.skip_connection:
            x4 = x4 + x2
            # x4 的尺寸: (batch_size, num_feat * 4, H/4, W/4)
            # 跳跃连接将 x4 和 x2 相加，x2 的尺寸与 x4 相同

        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        # x4 的尺寸: (batch_size, num_feat * 4, H/2, W/2)
        # 使用双线性插值将空间尺寸放大 2 倍

        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        # x5 的尺寸: (batch_size, num_feat * 2, H/2, W/2)
        # 第六层卷积 conv5 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        # 如果使用跳跃连接，则将当前特征与下采样特征相加
        if self.skip_connection:
            x5 = x5 + x1
            # x5 的尺寸: (batch_size, num_feat * 2, H/2, W/2)
            # 跳跃连接将 x5 和 x1 相加，x1 的尺寸与 x5 相同

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        # x5 的尺寸: (batch_size, num_feat * 2, H, W)
        # 使用双线性插值将空间尺寸放大 2 倍

        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        # x6 的尺寸: (batch_size, num_feat, H, W)
        # 第七层卷积 conv6 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        # 如果使用跳跃连接，则将当前特征与下采样特征相加
        if self.skip_connection:
            x6 = x6 + x0
            # x6 的尺寸: (batch_size, num_feat, H, W)
            # 跳跃连接将 x6 和 x0 相加，x0 的尺寸与 x6 相同

        # 额外的卷积层
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        # out 的尺寸: (batch_size, num_feat, H, W)
        # 第八层卷积 conv7 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        # out 的尺寸: (batch_size, num_feat, H, W)
        # 第九层卷积 conv8 不改变空间尺寸（kernel_size=3, stride=1, padding=1）

        out = self.conv9(out)
        # out 的尺寸: (batch_size, 1, H, W)
        # 第十层卷积 conv9 不改变空间尺寸（kernel_size=3, stride=1, padding=1），输出通道数为 1

        return out
