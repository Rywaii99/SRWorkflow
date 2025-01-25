"""
SCFDSRN 基于 U-Net 结构实现
"""
import torch
from torch import nn as nn
import torch.nn.functional as F
from archs.arch_util import Upsample, PixelUnshuffle, LayerNorm2d
from utils.registry import ARCH_REGISTRY


# ================================
# =      Resizing modules        =
# ================================
class DownsampleLayer(nn.Module):
    def __init__(self, n_feat):
        super(DownsampleLayer, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 像素折叠/下采样
        )

    def forward(self, x):
        return self.body(x)


class UpsampleLayer(nn.Module):
    def __init__(self, n_feat):
        super(UpsampleLayer, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


# ================================
# =     Functional modules       =
# ================================

class DWGate(nn.Module):
    """
    DWGate 类：实现基于深度可分离卷积的门控机制。
    通过深度可分离卷积对输入特征进行变换，并将结果分成两部分，返回它们的逐元素乘积。
    """

    def __init__(self, c, dw_expansion_factor=2):
        """
        初始化 DWGate 模块。

        Args:
            c (int): 输入特征的通道数。
            dw_expansion_factor (int, optional): 深度可分离卷积的通道扩展因子，默认值为 2。
        """
        super(DWGate, self).__init__()
        self.dim = c  # 输入通道数
        self.dw_channels = self.dim * dw_expansion_factor  # 扩展后的通道数

        # 深度可分离卷积层
        self.dw_conv = nn.Conv2d(
            in_channels=self.dw_channels,  # 输入通道数
            out_channels=self.dw_channels,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步幅
            padding=1,  # 填充
            groups=self.dw_channels,  # 分组数，实现深度可分离卷积
            bias=True  # 是否使用偏置
        )

    def forward(self, x):
        """
        前向传播：对输入特征进行深度可分离卷积，将结果分成两部分并返回它们的逐元素乘积。

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, C, H, W)，其中：
                - N 是批量大小，
                - C 是通道数（等于 self.dw_channels），
                - H 是高度，
                - W 是宽度。

        Returns:
            torch.Tensor: 返回两部分特征的逐元素乘积，形状为 (N, C // 2, H, W)。
        """
        # 输入张量 x 的形状: (N, C, H, W)，其中 C = self.dw_channels

        # 对输入特征进行深度可分离卷积
        x = self.dw_conv(x)  # 输出形状: (N, C, H, W)

        # 将卷积结果沿通道维度分成两部分
        x1, x2 = x.chunk(2, dim=1)  # 输出形状: x1 -> (N, C // 2, H, W), x2 -> (N, C // 2, H, W)

        # 返回两部分特征的逐元素乘积
        return x1 * x2  # 输出形状: (N, C // 2, H, W)


# Residual Gated Convolutional Feed-Forward Network (RGCFFN).
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        """
        初始化 FeedForward 模块。

        Args:
            dim (int): 输入特征的通道数。
            ffn_expansion_factor (int): FFN 扩展因子，用于计算隐藏层特征的通道数。
            bias (bool): 是否在卷积层中使用偏置。
        """
        super(FeedForward, self).__init__()

        # 计算隐藏层特征的通道数
        hidden_features = int(dim * ffn_expansion_factor)

        # 1x1 卷积层：将输入特征从 dim 通道扩展到 hidden_features * 2 通道
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 使用 DWGate 模块实现深度可分离卷积和门控机制
        self.dwg = DWGate(hidden_features, dw_expansion_factor=2)

        # 1x1 卷积层：将隐藏层特征从 hidden_features 通道恢复为 dim 通道
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        # 可学习的残差连接权重
        self.zeta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        """
        前向传播：实现 FeedForward 模块的计算过程。

        Args:
            x (torch.Tensor): 输入张量，形状为 (N, C, H, W)，其中：
                - N 是批量大小，
                - C 是通道数（等于 dim），
                - H 是高度，
                - W 是宽度。

        Returns:
            torch.Tensor: 输出张量，形状与输入 x 相同。
        """
        identity = x  # 保存输入特征，用于残差连接；形状: (N, C, H, W)

        # 1x1 卷积扩展通道数
        x = self.project_in(x)  # 输入形状: (N, C, H, W) -> 输出形状: (N, hidden_features * 2, H, W)

        # 使用 DWGate 模块进行深度可分离卷积和门控机制
        x = self.dwg(x)  # 输入形状: (N, hidden_features * 2, H, W) -> 输出形状: (N, hidden_features, H, W)

        # 1x1 卷积恢复通道数
        x = self.project_out(x)  # 输入形状: (N, hidden_features, H, W) -> 输出形状: (N, C, H, W)

        # 残差连接：identity + x * zeta
        return identity + x * self.zeta  # 输出形状: (N, C, H, W)


# ================================
# =     Attention modules        =
# ================================

# contrast-aware channel attention module
def mean_channels(F):
    """
    计算输入特征图每个通道的均值。

    Args:
        F (torch.Tensor): 输入特征图，形状为 (N, C, H, W)，其中 N 是批量大小，C 是通道数，H 是高度，W 是宽度。

    Returns:
        torch.Tensor: 每个通道的均值，形状为 (N, C, 1, 1)。
    """
    assert (F.dim() == 4)  # 确保输入是4维张量
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)  # 在空间维度 (H, W) 上求和
    return spatial_sum / (F.size(2) * F.size(3))  # 计算均值


def stdv_channels(F):
    """
    计算输入特征图每个通道的标准差。

    Args:
        F (torch.Tensor): 输入特征图，形状为 (N, C, H, W)。

    Returns:
        torch.Tensor: 每个通道的标准差，形状为 (N, C, 1, 1)。
    """
    assert (F.dim() == 4)  # 确保输入是4维张量
    F_mean = mean_channels(F)  # 计算每个通道的均值
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))  # 计算方差
    return F_variance.pow(0.5)  # 返回标准差


class CCALayer(nn.Module):
    """
    Contrast Channel Attention (CCA) 层。
    通过结合通道的标准差和全局平均池化，生成通道注意力权重。
    """

    def __init__(self, channel, reduction=16):
        """
        初始化CCA层。

        Args:
            channel (int): 输入特征的通道数。
            reduction (int, optional): 通道压缩的比例，默认值为16。
        """
        super(CCALayer, self).__init__()

        # 标准差计算函数
        self.contrast = stdv_channels

        # 全局平均池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 通道注意力机制的全连接层
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),  # 通道压缩
            nn.GELU(),  # !!改进，原 nn.ReLU(inplace=True)
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),  # 通道扩展
            nn.Sigmoid()  # Sigmoid激活函数，生成注意力权重
        )

    def forward(self, x):
        """
        前向传播：计算对比通道注意力权重，并应用于输入特征。

        Args:
            x (torch.Tensor): 输入特征张量，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 加权后的特征张量，形状与输入相同。
        """
        # 计算对比特征（标准差）和全局平均池化特征
        y = self.contrast(x) + self.avg_pool(x)  # 输出形状: (N, C, 1, 1)

        # 通过全连接层生成通道注意力权重
        y = self.conv_du(y)  # 输出形状: (N, C, 1, 1)

        # 应用通道注意力权重
        return x * y  # 输出形状: (N, C, H, W)

# ================================
# =           Blocks             =
# ================================


class GCABlock(nn.Module):
    """
    Gated Convolutional Attention Block (GCAB)：结合深度可分离卷积、门控机制和通道注意力机制的特征增强模块。
    """

    def __init__(self, c, dw_expansion_factor=2, ffn_expansion_factor=2, drop_out_rate=0.):
        """
        初始化 GCAB 模块。

        Args:
            c (int): 输入通道数。
            dw_expansion_factor (int, optional): 深度可分离卷积的通道扩展因子，默认值为 2。
            ffn_expansion_factor (int, optional): 前馈网络的通道扩展因子，默认值为 2。
            drop_out_rate (float, optional): Dropout 的概率，默认值为 0，表示不使用 Dropout。
        """
        super().__init__()

        # 计算扩展后的通道数
        self.dim = c  # 输入通道数
        self.dw_channel = c * dw_expansion_factor  # 深度可分离卷积扩展后的通道数
        self.ffn_channel = c * ffn_expansion_factor  # 前馈网络扩展后的通道数

        # 1x1 卷积层：扩展通道数
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, bias=True)

        # 3x3 深度可分离卷积层：对扩展后的特征进行空间卷积
        self.conv2 = nn.Conv2d(
            in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1,
            groups=self.dw_channel, bias=True
        )

        # 1x1 卷积层：恢复通道数
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

        # 深度可分离卷积门控机制
        self.dwg = DWGate(c, dw_expansion_factor)

        # 通道注意力机制
        self.cca = CCALayer(self.dw_channel // 2)

        # 前馈网络
        self.ffn = FeedForward(c, ffn_expansion_factor, bias=True)

        # LayerNorm 层
        self.norm1 = LayerNorm2d(c)  # 第一个 LayerNorm
        self.norm2 = LayerNorm2d(c)  # 第二个 LayerNorm

        # Dropout 层
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()  # 第一个 Dropout
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()  # 第二个 Dropout

        # 可学习的残差连接权重
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)  # 第一个残差连接的权重
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)  # 第二个残差连接的权重

    def forward(self, inp):
        """
        前向传播：GCAB 的计算过程，包括卷积操作、门控机制、通道注意力、前馈网络等。

        Args:
            inp (torch.Tensor): 输入张量，形状为 (N, C, H, W)，其中：
                - N 是批量大小，
                - C 是通道数，
                - H 是高度，
                - W 是宽度。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        x = inp  # 输入形状: (N, C, H, W)

        # 第一个 LayerNorm
        x = self.norm1(x)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 1x1 卷积扩展通道数
        x = self.conv1(x)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C * dw_expansion_factor, H, W)

        # 3x3 深度可分离卷积
        x = self.conv2(x)  # 输入形状: (N, C * dw_expansion_factor, H, W) -> 输出形状: (N, C * dw_expansion_factor, H, W)

        # 深度可分离卷积门控机制
        x = self.dwg(x)  # 输入形状: (N, C * dw_expansion_factor, H, W) -> 输出形状: (N, C * dw_expansion_factor // 2, H, W)

        # 通道注意力机制
        # 输入形状: (N, C * dw_expansion_factor // 2, H, W) -> 输出形状: (N, C * dw_expansion_factor // 2, H, W)
        x = x * self.cca(x)

        # 1x1 卷积恢复通道数
        x = self.conv3(x)  # 输入形状: (N, C * dw_expansion_factor // 2, H, W) -> 输出形状: (N, C, H, W)

        # 第一个 Dropout 操作
        x = self.dropout1(x)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 第一个残差连接：y = inp + x * beta
        y = inp + x * self.beta  # 输入形状: inp -> (N, C, H, W), x -> (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 第二个 LayerNorm
        y_norm = self.norm2(y)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 前馈网络
        ffn_out = self.ffn(y_norm)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 第二个 Dropout 操作
        ffn_out = self.dropout2(ffn_out)  # 输入形状: (N, C, H, W) -> 输出形状: (N, C, H, W)

        # 第二个残差连接：最终输出 y + ffn_out * gamma
        return y + ffn_out * self.gamma  # 输入形状: y -> (N, C, H, W), ffn_out -> (N, C, H, W) -> 输出形状: (N, C, H, W)


@ARCH_REGISTRY.register()
class GCABNet(nn.Module):
    """
    GCABNet：一个基于 GCABlock 的网络，包含编码器、解码器和中间块，适用于图像恢复任务。

    Args:
        img_channel (int): 输入图像的通道数，默认为 3（RGB 图像）。
        n_feat (int): 特征图的初始通道数，默认为 32。
        middle_blk_num (int): 中间块的个数，默认为 1。
        enc_blk_nums (list): 编码器中每个阶段的 GCABlock 数量列表。
        dec_blk_nums (list): 解码器中每个阶段的 GCABlock 数量列表。
        scale (int): 输入图像的缩放比例，默认为 2。

    Shape:
        - 输入: (N, C, H, W)，其中 N 是批量大小，C 是通道数，H 和 W 是图像的高度和宽度。
        - 输出: (N, C, H, W)，与输入图像形状相同。
    """

    def __init__(self,
                 img_channel=3,
                 n_feat=32,
                 middle_blk_num=1,
                 enc_blk_nums=[],
                 dec_blk_nums=[],
                 scale=2):
        super().__init__()

        # 输入卷积层，将输入图像映射到特征空间
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=n_feat, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        # 输出卷积层，将特征图映射回图像空间
        self.ending = nn.Conv2d(in_channels=n_feat, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1, bias=True)

        # 编码器模块列表
        self.encoders = nn.ModuleList()
        # 解码器模块列表
        self.decoders = nn.ModuleList()
        # 中间块模块列表
        self.middle_blks = nn.ModuleList()
        # 上采样模块列表
        self.ups = nn.ModuleList()
        # 下采样模块列表
        self.downs = nn.ModuleList()
        # 缩放比例
        self.scale = scale

        # 初始化特征通道数
        chan = n_feat

        # 构建编码器
        for num in enc_blk_nums:
            # 添加编码器块（多个 GCABlock）
            self.encoders.append(
                nn.Sequential(
                    *[GCABlock(chan) for _ in range(num)]
                )
            )
            # 添加下采样模块
            self.downs.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan // 2, 1, bias=False),  # 1x1 卷积减少通道数
                    PixelUnshuffle(2)  # 像素反卷积下采样
                )
            )
            # 更新特征通道数
            chan = chan * 2

        # 构建中间块
        self.middle_blks = nn.Sequential(
            *[GCABlock(chan) for _ in range(middle_blk_num)]
        )

        # 构建解码器
        for num in dec_blk_nums:
            # 添加上采样模块
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),  # 1x1 卷积增加通道数
                    nn.PixelShuffle(2)  # 像素重排上采样
                )
            )
            # 更新特征通道数
            chan = chan // 2
            # 添加解码器块（多个 GCABlock）
            self.decoders.append(
                nn.Sequential(
                    *[GCABlock(chan) for _ in range(num)]
                )
            )

        # 计算填充大小，确保输入图像大小可以被 2^len(encoders) 整除
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        """
        前向传播函数。

        Args:
            inp (torch.Tensor): 输入图像，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 输出图像，形状为 (N, C, H, W)。
        """
        # 如果缩放比例大于 1，对输入图像进行双三次插值上采样
        if self.scale > 1:
            inp = F.interpolate(inp, scale_factor=self.scale, mode='bicubic', align_corners=False)

        # 获取输入图像的形状
        B, C, H, W = inp.shape
        # 检查并调整输入图像大小
        inp = self.check_image_size(inp)

        # 输入卷积层
        x = self.intro(inp)

        # 编码器阶段
        encs = []  # 保存编码器的中间特征图，用于跳跃连接
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)  # 通过编码器块
            encs.append(x)  # 保存中间特征图
            x = down(x)  # 下采样

        # 中间块阶段
        x = self.middle_blks(x)

        # 解码器阶段
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)  # 上采样
            x = x + enc_skip  # 跳跃连接，添加编码器的中间特征图
            x = decoder(x)  # 通过解码器块

        # 输出卷积层
        x = self.ending(x)
        # 残差连接，添加输入图像
        x = x + inp

        # 返回裁剪后的输出图像，确保输出大小与输入一致
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """
        检查并调整输入图像的大小，以确保它可以被正确处理。

        Args:
            x (torch.Tensor): 输入图像，形状为 (N, C, H, W)。

        Returns:
            torch.Tensor: 调整大小后的图像。
        """
        _, _, h, w = x.size()
        # 计算需要填充的高度和宽度
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # 对输入图像进行填充
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x



if __name__ == '__main__':
    img_channel = 3
    n_feat = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    net = GCABNet(img_channel, n_feat, middle_blk_num, enc_blks, dec_blks)

    from thop import profile

    flops, params = profile(net, inputs=(torch.randn(1, 3, 256, 256),))
    print(f"FLOPs:{flops / 1e9}G, Params:{params / 1e6}M")  # FLOPs:95.928426624G, Params:37.388235M

    # # 创建输入张量
    # tensor = torch.randn(1, 64, 128, 128)
    # block = GCABlock(64)
    # # 前向传播
    # out = block(tensor)
    # print(out.shape)  # 输出形状: (1, 64, 128, 128)
