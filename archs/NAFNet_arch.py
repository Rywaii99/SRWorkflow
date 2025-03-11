# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.arch_util import LayerNorm2d
from archs.local_arch import Local_Base
from utils.registry import ARCH_REGISTRY


class SimpleGate(nn.Module):
    """
    SimpleGate 类：用于实现一个简单的门控机制。
    将输入x分成两个部分，然后返回它们的逐元素乘积。
    """

    def forward(self, x):
        """
        前向传播：将输入x分成两个部分，并返回它们的逐元素乘积。

        Args:
            x (torch.Tensor): 输入张量，形状为(N, C, H, W)，N是批量大小，C是通道数，H是高度，W是宽度。

        Returns:
            torch.Tensor: 返回x1和x2逐元素乘积后的结果，形状与输入x相同。
        """
        x1, x2 = x.chunk(2, dim=1)  # 将输入x沿通道维度拆分成两部分
        return x1 * x2  # 返回逐元素乘积


class NAFBlock(nn.Module):
    """
    NAFBlock 类：用于构建NAF（Non-Linear Attention-based Feature）块。
    包含卷积层、简单门控（SimpleGate）和通道注意力机制。
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        """
        初始化NAFBlock模块。

        Args:
            c (int): 输入通道数。
            DW_Expand (int, optional): 深度可分离卷积扩展因子，默认值为2。
            FFN_Expand (int, optional): FFN扩展因子，默认值为2。
            drop_out_rate (float, optional): Dropout的概率，默认值为0，表示不使用Dropout。
        """
        super().__init__()

        # 计算扩展后的通道数
        self.dim = c
        self.dw_channel = c * DW_Expand
        self.ffn_channel = c * FFN_Expand

        # 卷积层：1x1卷积，通道数扩展
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1, bias=True)
        # 深度可分离卷积：3x3卷积，通道数不变
        self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=1, stride=1, groups=self.dw_channel, bias=True)
        # 卷积层：将通道数恢复为c的一半
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

        # 简化的通道注意力机制（SCA）
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
        )

        # SimpleGate模块
        self.sg = SimpleGate()

        # FFN部分
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=self.ffn_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=self.ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, bias=True)

        # LayerNorm
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 参数beta和gamma，用于后续的加权
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        """
        前向传播：NAFBlock的计算过程，包括卷积操作、门控机制、通道注意力、FFN等。

        Args:
            inp (torch.Tensor): 输入张量，形状为(N, C, H, W)，N是批量大小，C是通道数，H是高度，W是宽度。

        Returns:
            torch.Tensor: 输出张量，形状与输入相同。
        """
        x = inp

        # 第一个LayerNorm
        x = self.norm1(x)

        # 卷积操作：1x1 -> 3x3 -> SimpleGate -> 通道注意力 -> 1x1卷积
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)  # 应用通道注意力
        x = self.conv3(x)

        # Dropout操作
        x = self.dropout1(x)

        # 残差连接：y = inp + x * beta
        y = inp + x * self.beta

        # 第二个LayerNorm
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        # 第二个Dropout操作
        x = self.dropout2(x)

        # 残差连接：最终输出y + x * gamma
        return y + x * self.gamma

    def flops(self, h, w):
        """
        计算NAFBlock的FLOPs（浮点运算量）。

        Args:
            h (int): 输入图像的高度。
            w (int): 输入图像的宽度。

        Returns:
            int: NAFBlock的FLOPs值。
        """
        flops = 0
        # norm1
        flops += h * w * self.dim
        # conv1
        flops += h * w * self.dim * self.dw_channel
        # conv2
        flops += h * w * self.dw_channel * self.dw_channel * 9 // self.dw_channel
        # sg
        flops += h * w * self.dw_channel
        # sca
        flops += h * w * self.dw_channel // 2
        # conv3
        flops += h * w * self.dw_channel // 2 * self.dim
        # norm2
        flops += h * w * self.dim
        # conv4
        flops += h * w * self.dim * self.ffn_channel
        # sg
        flops += h * w * self.ffn_channel
        # conv5
        flops += h * w * self.ffn_channel // 2 * self.dim

        return flops


# @ARCH_REGISTRY.register()
class NAFNet(nn.Module):
    """
    NAFNet：一个基于NAFBlock的网络，包含编码器、解码器、和中间块，适用于图像恢复任务。
    """

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], scale=1):
        """
        初始化NAFNet模型。

        Args:
            img_channel (int, optional): 输入图像的通道数，默认为3。
            width (int, optional): 模型的宽度，默认为16。
            middle_blk_num (int, optional): 中间块的数量，默认为1。
            enc_blk_nums (list, optional): 编码器块的数量列表，默认为空列表。
            dec_blk_nums (list, optional): 解码器块的数量列表，默认为空列表。
            scale (int, optional): 图像缩放因子，默认为1（不缩放）。
        """
        super().__init__()

        self.img_channel = img_channel
        self.width = width
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.scale = scale

        chan = width
        # 构建编码器
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # 构建中间块
        self.middle_blks = nn.Sequential(
            *[NAFBlock(chan) for _ in range(middle_blk_num)]
        )

        # 构建解码器
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        """
        前向传播：NAFNet的计算过程，包括编码器、解码器、中间块等。

        Args:
            inp (torch.Tensor): 输入图像，形状为(N, C, H, W)，N是批量大小，C是通道数，H是高度，W是宽度。

        Returns:
            torch.Tensor: 输出图像，形状与输入图像相同。
        """
        if self.scale > 1:
            inp = F.interpolate(inp, scale_factor=self.scale, mode='bicubic', align_corners=False)

        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        # 编码器
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # 中间块
        x = self.middle_blks(x)

        # 解码器
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip  # 跳跃连接
            x = decoder(x)

        # 输出
        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def flops(self, h, w):
        """
        计算NAFNet的FLOPs（浮点运算量）。

        Args:
            h (int): 输入图像的高度。
            w (int): 输入图像的宽度。

        Returns:
            int: NAFNet的FLOPs值。
        """
        flops = 0
        # intro
        flops += h * w * self.img_channel * self.width * 9
        # encoders
        for enc in self.encoders:
            for blk in enc:
                flops += blk.flops(h, w)
            h = h // 2
            w = w // 2
            # down
            flops += h * w * self.width * self.width * 2 * 4
        # middle
        for mid in self.middle_blks:
            flops += mid.flops(h, w)
        # decoders
        for dec in self.decoders:
            for blk in dec:
                flops += blk.flops(h, w)
            # up
            flops += h * w * self.width * self.width * 2
            h = h * 2
            w = w * 2
        # ending
        flops += h * w * self.width * self.img_channel * 9

        return flops

    def check_image_size(self, x):
        """
        检查并调整输入图像的大小，以确保它可以被正确处理。

        Args:
            x (torch.Tensor): 输入图像，形状为(N, C, H, W)。

        Returns:
            torch.Tensor: 调整大小后的图像。
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# @ARCH_REGISTRY.register()
class NAFNetLocal(Local_Base, NAFNet):
    """
    NAFNetLocal类：继承NAFNet并添加本地处理功能。
    """

    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        """
        初始化NAFNetLocal模型。

        Args:
            *args: NAFNet构造函数的参数。
            train_size (tuple): 训练图像的尺寸，默认为(1, 3, 256, 256)。
            fast_imp (bool): 是否使用快速实现，默认为False。
        """
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    enc_blks = [2, 2, 4, 8]
    middle_blk_num = 12
    dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks, scale=4)


    inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info

    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    # print(macs, params)

    # thop
    from thop import profile
    flops, params = profile(net, inputs=(torch.randn(16, 3, 128, 128),))
    print(f"FLOPs:{flops/1e9}G, Params:{params/1e6}M")
    print(f"FLOPs:{net.flops(128, 128)/1e9}G, Params:{params/1e6}M")