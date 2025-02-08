import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch_util import default_conv, Upsample
from utils.registry import ARCH_REGISTRY


# 非局部块（Non-Local Block 2D）
# ref: https://github.com/AlexHex7/Non-local_pytorch/blob/master/Non-Local_pytorch_0.4.1_to_1.1.0/lib/non_local_dot_product.py
# ref: https://github.com/yulunzhang/RNAN/blob/master/SR/code/model/common.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = inter_channels  # 中间层通道数

        # g, theta, phi 都是 1x1 卷积，作用是映射输入特征到中间通道空间
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)  # 初始化 W 权重为零
        nn.init.constant_(self.W.bias, 0)  # 初始化 W 偏置为零

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)  # 获取输入数据的 batch 大小

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # g(x) [B, C', H*W]
        g_x = g_x.permute(0, 2, 1)  # 转置为 [B, H*W, C']

        # 计算 theta(x) 和 phi(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # theta(x) [B, C', H*W]
        theta_x = theta_x.permute(0, 2, 1)  # 转置为 [B, H*W, C']

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # phi(x) [B, C', H*W]
        f = torch.matmul(theta_x, phi_x)  # 计算 f = theta(x) * phi(x) 的矩阵乘积 [B, H*W, H*W]

        f_div_C = F.softmax(f, dim=-1)  # 在最后一维上做 softmax 操作，得到权重矩阵 [B, H*W, H*W]

        # 计算输出
        y = torch.matmul(f_div_C, g_x)  # [B, H*W, C']
        y = y.permute(0, 2, 1).contiguous()  # 转置并保持连续内存布局 [B, C', H*W]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 恢复为 [B, C', H, W]

        W_y = self.W(y)  # 对输出 y 进行卷积，恢复到输入通道数
        z = W_y + x  # 加上原始输入，得到残差连接的结果

        return z  # 返回结果


class BasicBlock(nn.Sequential):
    """
    BasicBlock 是一个基本的卷积块，可以选择性地添加 BatchNorm 和激活函数。

    Args:
        conv: 用于卷积的操作
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        bias: 是否使用偏置
        bn: 是否使用 BatchNorm
        act: 激活函数
    """

    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=True,
            bn=False, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))  # 如果使用 BatchNorm，则添加
        if act is not None:
            m.append(act)  # 如果指定了激活函数，则添加
        super(BasicBlock, self).__init__(*m)


class AdjustedNonLocalBlock(nn.Module):
    """
    基于 NonLocalBlock2D 的调整版非局部块，用于捕捉输入特征图之间的长程依赖关系。
    该模块包含了多个卷积层，能够对特征图进行相应的转换和加权操作，最后通过残差连接输出结果。

    Args:
        in_channels: 输入特征图的通道数。
        inter_channels: 中间通道数，通常是 `in_channels` 的一半或其他合适的值。
    """
    def __init__(self, in_channels, inter_channels):
        super(AdjustedNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # 定义 g、W、theta、phi 的卷积层
        self.g = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                           kernel_size=1, stride=1, padding=0)  # 对输入特征图进行变换
        self.W = nn.Conv2d(in_channels=inter_channels, out_channels=in_channels,
                           kernel_size=1, stride=1, padding=0)  # 变换后通过卷积恢复通道数
        nn.init.constant_(self.W.weight, 0)  # 初始化 W 的权重为 0
        nn.init.constant_(self.W.bias, 0)  # 初始化 W 的偏置为 0

        # 定义 theta 和 phi 的卷积层
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)  # 用于计算 theta(x1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)  # 用于计算 phi(x0)

    def forward(self, x0, x1):
        """
        前向传播函数，计算非局部操作。

        Args:
            x0: 输入的第一个特征图，通常用于 `phi` 和 `g` 的计算。
            x1: 输入的第二个特征图，通常用于 `theta` 的计算。

        Returns:
            z: 输出的特征图，包含了非局部操作后的加权结果，具有残差连接。
        """
        batch_size = x0.size(0)

        # 对输入 x0 和 x1 分别计算 g(x0)、theta(x1) 和 phi(x0)
        g_x = self.g(x0).view(batch_size, self.inter_channels, -1)  # 对 x0 进行变换
        g_x = g_x.permute(0, 2, 1)  # 转置，方便后续计算矩阵乘积

        theta_x = self.theta(x1).view(batch_size, self.inter_channels, -1)  # 对 x1 进行变换
        theta_x = theta_x.permute(0, 2, 1)  # 转置

        phi_x = self.phi(x0).view(batch_size, self.inter_channels, -1)  # 对 x0 进行变换
        f = torch.matmul(theta_x, phi_x)  # 计算矩阵乘积：相似性矩阵

        # 计算 softmax，得到注意力权重
        f_div_C = F.softmax(f, dim=-1)  # 对最后一个维度进行 softmax

        # 获取加权的 g(x0) 特征
        y = torch.matmul(f_div_C, g_x)  # 通过加权矩阵得到新的特征
        y = y.permute(0, 2, 1).contiguous()  # 转置并保持连续内存布局
        y = y.view(batch_size, self.inter_channels, *x0.size()[2:])  # 恢复为原始尺寸

        # 对加权后的特征图 y 进行卷积变换
        W_y = self.W(y)

        # 残差连接，将原始输入 x0 与 W_y 相加
        z = W_y + x0  # 添加残差

        return z  # 返回最终的输出



class HSEM(nn.Module):
    """
    自适应多尺度自相似性模块（HSEM），通过多尺度自相似性学习和非局部融合提高图像特征的表达能力。

    该模块包括基础尺度计算、下采样尺度计算以及非局部特征融合模块，目的是捕捉不同尺度的自相似性信息，并通过非局部块进行融合。

    Args:
        conv: 用于卷积操作的函数，默认为 `default_conv`。
        n_feats: 特征图通道数。
        kernel_size: 卷积核大小。
        bias: 是否使用偏置项，默认为 True。
        bn: 是否使用批归一化，默认为 False。
        act: 激活函数，默认为 `ReLU` 激活。
    """

    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(HSEM, self).__init__()

        # 初始化基础尺度模块：包含一个 SSEM 模块用于基本尺度的自相似性计算
        base_scale = []
        base_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))  # [B, n_feats, H, W]

        # 初始化下采样尺度模块：包含一个 SSEM 模块用于下采样尺度的自相似性计算
        down_scale = []
        down_scale.append(SSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))  # [B, n_feats, H/2, W/2]

        # 初始化尾部处理模块：用于最后的特征处理
        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act))  # [B, n_feats, H, W]

        # 初始化非局部融合块，用于融合不同尺度的特征
        self.NonLocal_base = AdjustedNonLocalBlock(n_feats, n_feats // 2)

        # 将各个模块组合成顺序容器
        self.base_scale = nn.Sequential(*base_scale)
        self.down_scale = nn.Sequential(*down_scale)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入的图像张量，形状为 [B, C, H, W]，其中：
                - B: 批大小
                - C: 通道数
                - H: 高度
                - W: 宽度

        Returns:
            add_out: 输出的特征图，形状为 [B, n_feats, H, W]，与输入图像尺寸相同
        """

        add_out = x

        # 1. 基础尺度计算
        x_base = self.base_scale(x)  # [B, n_feats, H, W]

        # 2. 下采样尺度计算
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear')  # [B, n_feats, H/2, W/2]
        x_down = self.down_scale(x_down)  # [B, n_feats, H/2, W/2]

        # 3. 将下采样尺度的特征图尺寸调整为基础尺度特征图的尺寸
        x_down = F.interpolate(x_down, size=(x_base.shape[2], x_base.shape[3]), mode='bilinear')  # [B, n_feats, H, W]

        # 4. 使用非局部融合模块进行多尺度特征融合
        ms = self.NonLocal_base(x_base, x_down)  # [B, n_feats, H, W]

        # 5. 通过尾部处理模块进行最后的特征提取
        ms = self.tail(ms)  # [B, n_feats, H, W]

        # 6. 进行残差连接，将原始输入与处理后的特征进行加和
        add_out = add_out + ms  # [B, n_feats, H, W]

        return add_out  # 输出特征图，形状为 [B, n_feats, H, W]


class SSEM(nn.Module):
    """
    单尺度自相似性模块（SSEM），用于图像特征的提取与自相似性学习。

    该模块通过结合主干分支（MB）和注意力分支（AB）提取图像的自相似特征，并利用 Sigmoid 激活函数进行特征加权。

    Args:
        conv: 用于卷积操作的函数，默认为 `default_conv`。
        n_feats: 特征图通道数。
        kernel_size: 卷积核大小。
        bias: 是否使用偏置项，默认为 True。
        bn: 是否使用批归一化，默认为 False。
        act: 激活函数，默认为 `ReLU` 激活。
    """

    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(SSEM, self).__init__()

        # 头部基本模块（卷积 + 激活 + 批归一化）
        head = []
        head.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))  # [B, n_feats, H, W]

        # 主干分支（MB）：由两个 BasicBlock 组成
        MB = []  # 主分支
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))  # [B, n_feats, H, W]
        MB.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))  # [B, n_feats, H, W]

        # 注意力分支（AB）：通过 NonLocalBlock2D 提取全局特征
        AB = []  # 注意力分支
        AB.append(NonLocalBlock2D(n_feats, n_feats // 2))  # [B, n_feats, H, W]
        AB.append(nn.Conv2d(n_feats, n_feats, 1, padding=0, bias=True))  # [B, n_feats, H, W]

        # Sigmoid 激活函数，用于特征加权
        sigmoid = []
        sigmoid.append(nn.Sigmoid())  # [B, n_feats, H, W]

        # 尾部处理：最后一层 BasicBlock
        tail = []
        tail.append(BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn))  # [B, n_feats, H, W]

        # 将各个模块组合为一个顺序容器
        self.head = nn.Sequential(*head)
        self.MB = nn.Sequential(*MB)
        self.AB = nn.Sequential(*AB)
        self.sigmoid = nn.Sequential(*sigmoid)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入的图像张量，形状为 [B, C, H, W]，其中：
                - B: 批大小
                - C: 通道数
                - H: 高度
                - W: 宽度

        Returns:
            add_out: 输出的特征图，形状为 [B, n_feats, H, W]，与输入图像尺寸相同
        """

        # 将输入通过头部基本模块进行处理
        add_out = x
        x_head = self.head(x)  # [B, n_feats, H, W]

        # 通过主干分支（MB）进行特征提取
        x_MB = self.MB(x_head)  # [B, n_feats, H, W]

        # 通过注意力分支（AB）提取全局自相似特征
        x_AB = self.AB(x_head)  # [B, n_feats, H, W]
        x_AB = self.sigmoid(x_AB)  # [B, n_feats, H, W]，通过 Sigmoid 激活进行归一化

        # 将主干分支的输出与注意力分支的输出相乘（加权操作）
        x_MB_AB = x_MB * x_AB  # [B, n_feats, H, W]，通过注意力加权后的特征

        # 最后的尾部处理
        x_tail = self.tail(x_MB_AB)  # [B, n_feats, H, W]

        # 将尾部处理的输出与原始输入进行残差连接
        add_out = add_out + x_tail  # [B, n_feats, H, W]

        return add_out  # 输出图像的特征图，形状为 [B, n_feats, H, W]


# multi-scale self-similarity block
class BasicModule(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True,
                 bn=False, act=nn.ReLU(True)):
        super(BasicModule, self).__init__()

        head = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        body = []
        body.append(HSEM(conv, n_feats, kernel_size, bias=bias, bn=bn, act=act))

        tail = [
            BasicBlock(conv, n_feats, n_feats, kernel_size, bias=bias, bn=bn, act=act)
            for _ in range(2)
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        add_out = x

        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)

        add_out = add_out + x

        return add_out


@ARCH_REGISTRY.register()
class HSENet(nn.Module):
    """
    HSENet 网络类，利用多层基础模块（BasicModule）构建图像超分辨率模型。

    Args:
        args: 配置参数，包含模型所需的各类超参数。
            - n_feats: 特征图通道数
            - scale: 超分辨率的尺度因子
            - n_colors: 输入图像的通道数（如RGB为3）
            - n_basic_modules: 基础模块（BasicModule）的数量
        conv: 用于卷积操作的函数，默认为 `default_conv`。
    """

    def __init__(self, scale=2, n_feats=64, n_basic_modules=10, conv=default_conv):
        super(HSENet, self).__init__()

        self.n_feats = n_feats  # 特征图通道数
        kernel_size = 3  # 卷积核大小
        self.scale = scale  # 超分辨率的尺度因子
        self.act = nn.ReLU(True)  # 激活函数（ReLU）
        n_colors = 3

        self.n_BMs = n_basic_modules  # 基础模块的数量

        # 头部模块（网络的输入层，通常是将图像的RGB通道数映射到n_feats通道数）
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # 定义主干体（多个 BasicModule，用于特征提取）
        self.body_modulist = nn.ModuleList([
            BasicModule(conv, n_feats, kernel_size, act=self.act)
            for _ in range(self.n_BMs)
        ])

        # 定义尾部模块（用于上采样和输出图像）
        m_tail = [
            Upsample(scale, n_feats),  # 上采样层，用于图像尺寸的放大
            conv(n_feats, n_colors, kernel_size)  # 输出层，恢复到输入图像的通道数
        ]

        # 网络的头部（输入部分）和尾部（输出部分）
        self.head = nn.Sequential(*m_head)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        """
        前向传播函数

        Args:
            x: 输入的图像张量，形状为 [B, C, H, W]，其中：
                - B: 批大小
                - C: 通道数（如RGB为3）
                - H: 高度
                - W: 宽度

        Returns:
            x: 输出图像，形状为 [B, C, H*scale, W*scale]，即超分辨率图像
        """

        # 输入图像通过头部卷积层（head）
        x = self.head(x)  # [B, n_feats, H, W] -> 特征图尺寸与输入相同，只是通道数变为 n_feats

        # 保存初始输入，以便后续残差连接
        add_out = x  # [B, n_feats, H, W]

        # 主体部分（多个 BasicModule 进行特征提取）
        for i in range(self.n_BMs):
            x = self.body_modulist[i](x)  # 每个 BasicModule 输出的形状仍为 [B, n_feats, H, W]
        add_out = add_out + x  # 残差连接，保持形状 [B, n_feats, H, W]

        # 通过尾部模块（上采样和输出卷积）
        x = self.tail(add_out)  # [B, n_feats, H*scale, W*scale] -> [B, n_colors, H*scale, W*scale]

        return x  # 输出超分辨率图像，形状为 [B, n_colors, H*scale, W*scale]
