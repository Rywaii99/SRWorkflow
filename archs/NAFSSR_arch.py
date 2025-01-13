# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
NAFSSR: Stereo Image Super-Resolution Using NAFNet
@InProceedings{Chu2022NAFSSR,
  author    = {Xiaojie Chu and Liangyu Chen and Wenqing Yu},
  title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
  booktitle = {CVPRW},
  year      = {2022},
}
'''

'''
在这里我们只使用的单输入的NAFSSR，因为我们想要的是单幅图像超分辨率任务的模型。
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from archs.local_arch import Local_Base
from archs.NAFNet_arch import LayerNorm2d, NAFBlock
from archs.arch_util import MySequential
from utils.registry import ARCH_REGISTRY


class DropPath(nn.Module):
    """
    DropPath 是一种用于训练时随机丢弃路径的模块，通常用于正则化。
    在训练过程中，它会以给定的概率丢弃某些路径（即层的输出），
    以强制模型学习更鲁棒的特征。它类似于 Dropout，但作用于整个路径而不仅仅是特征。

    Args:
        drop_rate (float): 丢弃率，表示丢弃路径的概率，范围为 [0, 1]。
        module (nn.Module): 需要应用 DropPath 的子模块。
    """

    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate  # 丢弃率
        self.module = module  # 需要应用 DropPath 的子模块

    def forward(self, *feats):
        """
        前向传播方法，执行 DropPath 操作。

        在训练时，根据丢弃率随机丢弃路径，即部分层的输出。
        如果不丢弃路径，则执行模块的前向传播操作，并按比例调整输出，
        以确保训练时的期望输出与没有丢弃路径时一致。

        Args:
            *feats (tuple): 输入特征，可以是多个输入张量。

        Returns:
            tuple: 处理后的特征，可能包含调整后的特征（如果丢弃路径）。
        """
        # 如果是训练模式，并且随机生成的值小于丢弃率，则返回原始输入特征，表示丢弃当前路径
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        # 否则，执行模块的前向传播，得到新的特征
        new_feats = self.module(*feats)

        # 如果是训练模式，调整输出特征的值，使得丢弃路径的期望值等于没有丢弃时的值
        factor = 1. / (1 - self.drop_rate) if self.training else 1.

        # 如果是训练模式，且丢弃率不为 0，则对输出进行调整
        if self.training and factor != 1.:
            new_feats = tuple([x + factor * (new_x - x) for x, new_x in zip(feats, new_feats)])

        return new_feats


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)

    SCAM 模块用于立体图像（Stereo Image）之间的跨模态注意力机制，旨在通过利用左右视图（left 和 right）之间的关系，
    强化它们之间的信息交互，从而提升图像的特征表示能力，特别是在图像超分辨率和深度估计等任务中。

    SCAM 的核心思想是通过注意力机制对左右图像进行交叉注意力（Cross Attention）计算，从而实现信息的互补和融合。

    Args:
        c (int): 输入特征的通道数。
    '''

    def __init__(self, c):
        super().__init__()
        # 初始化注意力计算的缩放因子
        self.scale = c ** -0.5

        # 为左视图和右视图分别初始化 LayerNorm 层
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)

        # 定义左右视图的投影层，用于生成查询（Q）向量
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        # 定义左右视图的缩放因子 β 和 γ，使用可学习的参数
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # 定义用于生成值（V）向量的投影层
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        '''
        前向传播方法，计算左右图像之间的跨模态注意力，并将其结果加权返回。

        Args:
            x_l (Tensor): 左视图输入，形状为 (B, C, H, W)，B 为批量大小，C 为通道数，H 和 W 分别为高度和宽度。
            x_r (Tensor): 右视图输入，形状与 x_l 相同。

        Returns:
            Tuple[Tensor, Tensor]: 返回经过交叉注意力机制加权后的左右视图特征，形状与输入相同。
        '''
        # 对左视图进行归一化处理，并通过卷积生成查询（Q）向量
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        # 对右视图进行归一化处理，并通过卷积生成转置的查询（Q）向量
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (转置)

        # 对左右视图进行投影，生成值（V）向量
        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # 计算注意力矩阵，(B, H, W, c) 与 (B, H, c, W) 进行矩阵乘法，得到 (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        # 计算从右视图到左视图的注意力加权特征
        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        # 计算从左视图到右视图的注意力加权特征
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # 对注意力加权特征进行缩放
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma

        # 返回经过交叉注意力机制加权后的左右视图特征
        return x_l + F_r2l, x_r + F_l2r


class NAFBlockSR(nn.Module):
    """
    NAFBlockSR 是用于超分辨率（Super-Resolution）的 NAFBlock 模块，结合了 NAFBlock 和可选的空间通道注意力机制（SCAM）。
    它通过 NAFBlock 提供有效的特征提取和处理，同时支持通过 SCAM 模块进行特征融合，增强图像的细节恢复能力。

    Args:
        c (int): 输入特征的通道数。
        fusion (bool): 是否使用空间通道注意力机制（SCAM）进行特征融合，默认为 False，不使用 SCAM。
        drop_out_rate (float): DropOut 的丢弃率，默认为 0，不使用 DropOut。

    """

    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        # 初始化 NAFBlock 模块，传入通道数和 DropOut 丢弃率
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)
        # 如果 fusion 为 True，则初始化 SCAM 模块，用于特征融合
        self.fusion = SCAM(c) if fusion else None

    def forward(self, *feats):
        """
        前向传播方法，首先通过 NAFBlock 处理输入特征，
        如果启用了 SCAM，则进一步进行特征融合。

        Args:
            *feats (tuple): 输入特征，可以是多个输入张量。

        Returns:
            tuple: 处理后的特征（可能经过 SCAM 融合后的结果）。
        """
        # 通过 NAFBlock 处理每个输入特征
        feats = tuple([self.blk(x) for x in feats])

        # 如果启用了 SCAM 特征融合，进行融合操作
        if self.fusion:
            feats = self.fusion(*feats)

        return feats


@ARCH_REGISTRY.register()
class NAFNetSR(nn.Module):
    """
    NAFNetSR：用于超分辨率（Super-Resolution）任务的NAFNet网络。
    本网络基于NAFBlock构建，并使用PixelShuffle进行上采样。
    """

    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_path_rate=0., drop_out_rate=0.,
                 fusion_from=-1, fusion_to=-1, dual=False):
        """
        初始化NAFNetSR网络。

        Args:
            up_scale (int, optional): 超分辨率放大倍数，默认为4。
            width (int, optional): 网络的宽度（通道数），默认为48。
            num_blks (int, optional): 网络中的NAFBlock块数，默认为16。
            img_channel (int, optional): 输入图像的通道数，默认为3（即RGB图像）。
            drop_path_rate (float, optional): DropPath的丢弃概率，默认为0。
            drop_out_rate (float, optional): Dropout的丢弃概率，默认为0。
            fusion_from (int, optional): 从某层开始进行特征融合，默认为-1，表示不进行特征融合。
            fusion_to (int, optional): 到某层结束进行特征融合，默认为-1，表示不进行特征融合。
            dual (bool, optional): 是否为双视图输入（立体超分辨率），默认为False。

        """
        super().__init__()

        # 如果是双输入（立体图像超分辨率），dual为True
        self.dual = dual

        # 输入层：将图像通道数转为宽度（通道数）
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        # 网络主体：由多个NAFBlockSR（包含DropPath）组成
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                NAFBlockSR(
                    width,
                    fusion=False,  # 设置不进行特征融合
                    drop_out_rate=drop_out_rate
                )) for i in range(num_blks)]  # num_blks决定NAFBlock的数量
        )

        # 上采样层：通过Conv2d和PixelShuffle进行上采样
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),  # 通过1x1卷积扩展通道数
            nn.PixelShuffle(up_scale)  # 使用PixelShuffle进行上采样
        )
        self.up_scale = up_scale  # 放大倍数

    def forward(self, inp):
        """
        前向传播：进行超分辨率处理，返回高分辨率图像。

        Args:
            inp (torch.Tensor): 输入张量，形状为(N, C, H, W)，N是批量大小，C是通道数，H是高度，W是宽度。

        Returns:
            torch.Tensor: 输出的超分辨率图像，形状为(N, C, H * up_scale, W * up_scale)。
        """
        # 使用双线性插值进行上采样，得到高分辨率的目标图像
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')

        if self.dual:
            # 如果是双视图输入，将输入拆分为左视图和右视图
            inp = inp.chunk(2, dim=1)
        else:
            inp = (inp,)  # 如果不是双视图输入，将输入打包成一个元组

        # 对每个输入视图进行特征提取
        feats = [self.intro(x) for x in inp]

        # 通过主体部分（多个NAFBlockSR）处理特征
        feats = self.body(*feats)

        # 将多个视图的输出进行拼接
        out = torch.cat([self.up(x) for x in feats], dim=1)

        # 添加高分辨率目标图像（残差连接）
        out = out + inp_hr
        return out


@ARCH_REGISTRY.register()
class NAFNetSRLocal(Local_Base, NAFNetSR):
    """
    NAFNetSRLocal：基于NAFNetSR的本地处理版本，继承了Local_Base和NAFNetSR的功能。
    用于本地优化和调整，适应不同的训练和推理需求。
    """

    def __init__(self, *args, train_size=(1, 3, 64, 64), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        """
        初始化NAFNetSRLocal模型。

        Args:
            *args: 传递给NAFNetSR的参数。
            train_size (tuple, optional): 训练时的输入图像大小，默认为(1, 3, 64, 64)。
            fast_imp (bool, optional): 是否启用快速实现，默认为False。
            fusion_from (int, optional): 从某层开始进行特征融合，默认为-1，表示不进行特征融合。
            fusion_to (int, optional): 到某层结束进行特征融合，默认为1000，表示进行整个网络的特征融合。
        """
        # 初始化Local_Base和NAFNetSR
        Local_Base.__init__(self)
        NAFNetSR.__init__(self, *args, **kwargs)

        N, C, H, W = train_size  # 获取训练图像的尺寸
        base_size = (int(H * 1.5), int(W * 1.5))  # 计算基准图像大小，扩展1.5倍

        # 设置为评估模式，并使用no_grad进行推理时不计算梯度
        self.eval()
        with torch.no_grad():
            # 执行模型的转换，调整输入大小、训练大小等参数
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

if __name__ == '__main__':
    num_blks = 128
    width = 106
    droppath = 0.3
    # train_size = (1, 6, 30, 90)

    net = NAFNetSR(up_scale=2, width=width, num_blks=num_blks, drop_path_rate=droppath)

    input = torch.randn(1, 3, 192, 192)

    output = net(input)
    print(output.shape)

    # from basicsr.models.archs.arch_util import measure_inference_speed
    # net = net.cuda()
    # data = torch.randn((1, 6, 128, 128)).cuda()
    # measure_inference_speed(net, (data,))

    # thop
    from thop import profile

    flops, params = profile(net, inputs=(input,))
    print(f"FLOPs:{flops / 1e9}G, Params:{params / 1e6}M")
# FLOPs:328.140818176G, Params:10.44822M
