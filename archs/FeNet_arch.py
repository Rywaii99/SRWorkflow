import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import ARCH_REGISTRY


class CALayer(nn.Module):
    """
    CALayer 类：通道注意力层（Channel Attention Layer）。
    通过全局平均池化和全连接层生成通道注意力权重，用于增强特征图的通道信息。
    """

    def __init__(self, num_fea):
        """
        初始化 CALayer。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(CALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将特征图的空间维度压缩为 1x1
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),  # 1x1 卷积，减少通道数
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),  # 1x1 卷积，恢复通道数
            nn.Sigmoid()  # Sigmoid 激活函数，生成通道注意力权重
        )

    def forward(self, fea):
        """
        前向传播：计算通道注意力权重并应用于输入特征图。

        Args:
            fea (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过通道注意力加权后的特征图，形状为 (N, C, H, W)。
        """
        # fea: (N, C, H, W) -> conv_du -> (N, C, 1, 1) -> 广播到 (N, C, H, W)
        return self.conv_du(fea)


class LLBlock(nn.Module):
    """
    LLBlock 类：轻量级格子块（Lightweight Lattice Block）。
    通过卷积和通道注意力机制增强特征图的表达能力。
    """

    def __init__(self, num_fea):
        """
        初始化 LLBlock。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(LLBlock, self).__init__()
        self.channel1 = num_fea // 2  # 将通道数分为两部分
        self.channel2 = num_fea - self.channel1

        # 卷积块，包含多个卷积层和激活函数
        self.convblock = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),  # 3x3 卷积
            nn.LeakyReLU(0.05),  # LeakyReLU 激活函数
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),  # 3x3 卷积
            nn.LeakyReLU(0.05),  # LeakyReLU 激活函数
            nn.Conv2d(self.channel1, self.channel1, 3, 1, 1),  # 3x3 卷积
        )

        # 通道注意力层
        self.A_att_conv = CALayer(self.channel1)
        self.B_att_conv = CALayer(self.channel2)

        # 1x1 卷积，用于特征融合
        self.fuse1 = nn.Conv2d(num_fea, self.channel1, 1, 1, 0)
        self.fuse2 = nn.Conv2d(num_fea, self.channel2, 1, 1, 0)
        self.fuse = nn.Conv2d(num_fea, num_fea, 1, 1, 0)

    def forward(self, x):
        """
        前向传播：通过卷积和通道注意力机制处理输入特征图。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 处理后的特征图，形状为 (N, C, H, W)。
        """
        # x: (N, C, H, W) -> split -> x1: (N, C//2, H, W), x2: (N, C//2, H, W)
        x1, x2 = torch.split(x, [self.channel1, self.channel2], dim=1)

        # x1: (N, C//2, H, W) -> convblock -> (N, C//2, H, W)
        x1 = self.convblock(x1)

        # x1: (N, C//2, H, W) -> A_att_conv -> (N, C//2, H, W)
        A = self.A_att_conv(x1)

        # P: (N, C//2, H, W) + (N, C//2, H, W) -> (N, C, H, W)
        P = torch.cat((x2, A * x1), dim=1)

        # x2: (N, C//2, H, W) -> B_att_conv -> (N, C//2, H, W)
        B = self.B_att_conv(x2)

        # Q: (N, C//2, H, W) + (N, C//2, H, W) -> (N, C, H, W)
        Q = torch.cat((x1, B * x2), dim=1)

        # c: (N, C, H, W) + (N, C, H, W) -> (N, C, H, W)
        c = torch.cat((self.fuse1(P), self.fuse2(Q)), dim=1)

        # out: (N, C, H, W) -> fuse -> (N, C, H, W)
        out = self.fuse(c)
        return out


class AF(nn.Module):
    """
    AF 类：注意力融合模块（Attention Fusion）。
    通过通道注意力机制和卷积操作融合两个特征图。
    """

    def __init__(self, num_fea):
        """
        初始化 AF。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(AF, self).__init__()
        self.CA1 = CALayer(num_fea)  # 通道注意力层 1
        self.CA2 = CALayer(num_fea)  # 通道注意力层 2
        self.fuse = nn.Conv2d(num_fea * 2, num_fea, 1)  # 1x1 卷积，用于特征融合

    def forward(self, x1, x2):
        """
        前向传播：融合两个特征图。

        Args:
            x1 (torch.Tensor): 第一个输入特征图，形状为 (N, C, H, W)。
            x2 (torch.Tensor): 第二个输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 融合后的特征图，形状为 (N, C, H, W)。
        """
        # x1: (N, C, H, W) -> CA1 -> (N, C, H, W)
        x1 = self.CA1(x1) * x1

        # x2: (N, C, H, W) -> CA2 -> (N, C, H, W)
        x2 = self.CA2(x2) * x2

        # out: (N, 2C, H, W) -> fuse -> (N, C, H, W)
        return self.fuse(torch.cat((x1, x2), dim=1))


class FEB(nn.Module):
    """
    FEB 类：特征增强块（Feature Enhancement Block）。
    通过多个轻量级格子块和注意力融合模块增强特征图的表达能力。
    """

    def __init__(self, num_fea):
        """
        初始化 FEB。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(FEB, self).__init__()
        self.CB1 = LLBlock(num_fea)  # 轻量级格子块 1
        self.CB2 = LLBlock(num_fea)  # 轻量级格子块 2
        self.CB3 = LLBlock(num_fea)  # 轻量级格子块 3
        self.AF1 = AF(num_fea)  # 注意力融合模块 1
        self.AF2 = AF(num_fea)  # 注意力融合模块 2

    def forward(self, x):
        """
        前向传播：通过多个轻量级格子块和注意力融合模块增强特征图。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 增强后的特征图，形状为 (N, C, H, W)。
        """
        # x: (N, C, H, W) -> CB1 -> (N, C, H, W)
        x1 = self.CB1(x)

        # x1: (N, C, H, W) -> CB2 -> (N, C, H, W)
        x2 = self.CB2(x1)

        # x2: (N, C, H, W) -> CB3 -> (N, C, H, W)
        x3 = self.CB3(x2)

        # f1: (N, C, H, W) + (N, C, H, W) -> AF1 -> (N, C, H, W)
        f1 = self.AF1(x3, x2)

        # f2: (N, C, H, W) + (N, C, H, W) -> AF2 -> (N, C, H, W)
        f2 = self.AF2(f1, x1)

        # out: (N, C, H, W) + (N, C, H, W) -> (N, C, H, W)
        return x + f2


class RB(nn.Module):
    """
    RB 类：残差块（Residual Block）。
    通过卷积层和残差连接增强特征图的表达能力。
    """

    def __init__(self, num_fea):
        """
        初始化 RB。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(RB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_fea, num_fea * 2, 3, 1, 1),  # 3x3 卷积，增加通道数
            nn.LeakyReLU(0.05),  # LeakyReLU 激活函数
            nn.Conv2d(num_fea * 2, num_fea, 3, 1, 1),  # 3x3 卷积，恢复通道数
        )

    def forward(self, x):
        """
        前向传播：通过卷积层和残差连接增强特征图。

        Args:
            x (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 增强后的特征图，形状为 (N, C, H, W)。
        """
        # x: (N, C, H, W) -> conv -> (N, C, H, W) + x -> (N, C, H, W)
        return self.conv(x) + x


class BFModule(nn.Module):
    """
    BFModule 类：特征融合模块（Feature Fusion Module）。
    通过多个卷积层和激活函数融合不同层次的特征图。
    """

    def __init__(self, num_fea):
        """
        初始化 BFModule。

        Args:
            num_fea (int): 输入特征图的通道数。
        """
        super(BFModule, self).__init__()
        self.conv4 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，减少通道数
        self.conv3 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，减少通道数
        self.fuse43 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，用于特征融合
        self.conv2 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，减少通道数
        self.fuse32 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，用于特征融合
        self.conv1 = nn.Conv2d(num_fea, num_fea // 2, 1, 1, 0)  # 1x1 卷积，减少通道数

        self.act = nn.ReLU(inplace=True)  # ReLU 激活函数

    def forward(self, x_list):
        """
        前向传播：融合不同层次的特征图。

        Args:
            x_list (list of torch.Tensor): 输入特征图列表，包含四个不同层次的特征图。

        Returns:
            torch.Tensor: 融合后的特征图，形状为 (N, C//2, H, W)。
        """
        # H4: (N, C, H, W) -> conv4 -> (N, C//2, H, W)
        H4 = self.act(self.conv4(x_list[3]))

        # H3_half: (N, C, H, W) -> conv3 -> (N, C//2, H, W)
        H3_half = self.act(self.conv3(x_list[2]))

        # H3: (N, C//2, H, W) + (N, C//2, H, W) -> fuse43 -> (N, C//2, H, W)
        H3 = self.fuse43(torch.cat([H4, H3_half], dim=1))

        # H2_half: (N, C, H, W) -> conv2 -> (N, C//2, H, W)
        H2_half = self.act(self.conv2(x_list[1]))

        # H2: (N, C//2, H, W) + (N, C//2, H, W) -> fuse32 -> (N, C//2, H, W)
        H2 = self.fuse32(torch.cat([H3, H2_half], dim=1))

        # H1_half: (N, C, H, W) -> conv1 -> (N, C//2, H, W)
        H1_half = self.act(self.conv1(x_list[0]))

        # H1: (N, C//2, H, W) + (N, C//2, H, W) -> (N, C, H, W)
        H1 = torch.cat([H2, H1_half], dim=1)

        return H1


@ARCH_REGISTRY.register()
class FENet(nn.Module):
    """
    FENet 类：超轻量级特征增强网络（Super Lightweight Feature Enhancement Network）。
    通过多个特征增强块和特征融合模块实现图像的超分辨率重建。
    """

    def __init__(self, upscale_factor=2, in_channels=3, num_fea=48, out_channels=3, num_LBs=4):
        """
        初始化 FENet。

        Args:
            upscale_factor (int): 上采样倍数，默认为 2。
            in_channels (int): 输入图像的通道数，默认为 3。
            num_fea (int): 特征图的通道数，默认为 48。
            out_channels (int): 输出图像的通道数，默认为 3。
            num_LBs (int): 特征增强块的数量，默认为 4。
        """
        super(FENet, self).__init__()

        self.num_LBs = num_LBs
        self.upscale_factor = upscale_factor

        # 特征提取层
        self.fea_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_fea, 3, 1, 1),  # 3x3 卷积，提取特征
            nn.Conv2d(num_fea, num_fea, 3, 1, 1)  # 3x3 卷积，进一步提取特征
        )

        # 特征增强块
        LBs = []
        for i in range(num_LBs):
            LBs.append(FEB(num_fea))  # 添加多个特征增强块
        self.LBs = nn.ModuleList(LBs)

        # 特征融合模块
        self.BFM = BFModule(num_fea)

        # 重建层
        self.upsample = nn.Sequential(
            nn.Conv2d(num_fea, num_fea, 3, 1, 1),  # 3x3 卷积，准备上采样
            nn.Conv2d(num_fea, out_channels * (upscale_factor ** 2), 3, 1, 1),  # 3x3 卷积，生成上采样后的特征图
            nn.PixelShuffle(upscale_factor)  # 像素重组，实现上采样
        )

    def forward(self, x):
        """
        前向传播：实现图像的超分辨率重建。

        Args:
            x (torch.Tensor): 输入图像，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 超分辨率重建后的图像，形状为 (N, C, H*upscale_factor, W*upscale_factor)。
        """
        # bi: (N, C, H, W) -> interpolate -> (N, C, H*upscale_factor, W*upscale_factor)
        bi = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        # fea: (N, C, H, W) -> fea_conv -> (N, num_fea, H, W)
        fea = self.fea_conv(x)

        # 特征增强块
        outs = []
        temp = fea
        for i in range(self.num_LBs):
            # temp: (N, num_fea, H, W) -> LBs[i] -> (N, num_fea, H, W)
            temp = self.LBs[i](temp)
            outs.append(temp)

        # H: (N, num_fea, H, W) -> BFM -> (N, num_fea//2, H, W)
        H = self.BFM(outs)

        # out: (N, num_fea//2, H, W) + (N, num_fea, H, W) -> upsample -> (N, C, H*upscale_factor, W*upscale_factor)
        out = self.upsample(H + fea)

        # 残差连接
        return out + bi
