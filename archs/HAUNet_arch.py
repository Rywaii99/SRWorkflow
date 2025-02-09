import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from archs.local_arch import AvgPool2d
from archs.arch_util import LayerNorm2d
from utils.registry import ARCH_REGISTRY


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1
            setattr(model, n, pool)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Reconstruct(nn.Module):
    def __init__(self, scale_factor):
        super(Reconstruct, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        if self.scale_factor!=1:
            x = nn.Upsample(scale_factor=self.scale_factor)(x)
        return x


class qkvblock(nn.Module):
    """
        这个类实现了一个基于多尺度特征输入的自注意力模块（QKVBlock），
        结合了深度可分离卷积和多头注意力机制。它通过输入三个不同尺度的特征图来生成输出，并且使用自定义的门控机制和前馈网络（FFN）进行特征变换。

        Args:
            c (int): 输入特征图的通道数。
            num_heads (int, 可选): 多头注意力的头数。默认值是2。
            FFN_Expand (int, 可选): FFN（前馈网络）扩展因子。默认值是2。

        Attributes:
            num_heads (int): 多头注意力的头数。
            kv (nn.Conv2d): 用于生成键（K）和值（V）的卷积层。
            kv_dwconv (nn.Conv2d): 深度可分离卷积，用于进一步处理生成的键值特征。
            q (nn.Conv2d): 查询（Q）的卷积层。
            q_dwconv (nn.Conv2d): 深度可分离卷积，用于处理查询特征。
            q1, q2 (nn.Conv2d): 用于生成不同尺度查询（Q）的卷积层。
            q1_dwconv, q2_dwconv (nn.Conv2d): 深度可分离卷积，用于处理不同尺度的查询特征。
            project_out (nn.Conv2d): 输出的卷积层，用于投影最终的注意力输出。
            temperature, temperature1, temperature2 (nn.Parameter): 温度系数，用于调节不同头的注意力权重。
            sg (SimpleGate): 自定义的门控机制。
            conv4, conv5, conv4_1, conv5_1, conv4_2, conv5_2 (nn.Conv2d): 前馈网络（FFN）的卷积层，用于特征变换。
            normq, normq1, normq2 (LayerNorm2d): 用于不同尺度特征的归一化层。
            norm2, norm2_1, norm2_2 (LayerNorm2d): 用于输入的归一化层，配合前馈网络（FFN）使用。
            beta, gamma, beta1, gamma1, beta2, gamma2 (nn.Parameter): 可学习的缩放系数，用于调节输入特征图。
            relu (nn.ReLU): 激活函数，用于前馈网络。
            softmax (nn.Softmax): 用于计算注意力权重的 softmax 层。

        """
    def __init__(self, c, num_heads=2, FFN_Expand=2):
        """
        初始化函数，设置各个模块的卷积层、归一化层、温度系数和门控机制等。

        Args：
            c (int): 输入特征图的通道数。
            num_heads (int): 多头注意力的头数，默认为2。
            FFN_Expand (int): FFN的扩展因子，默认为2。
        """

        super().__init__()

        self.num_heads = num_heads  # 多头注意力的头数，默认2
        # 定义查询、键和值的卷积层
        self.kv = nn.Conv2d(c * 3, c * 6, kernel_size=1)  # 用于生成k和v的卷积层，输入通道3*c，输出通道6*c
        self.kv_dwconv = nn.Conv2d(c * 6, c * 6, 3, padding=1, groups=c * 6)  # 深度可分离卷积

        self.q = nn.Conv2d(c, c, kernel_size=1)  # 查询卷积层
        self.q_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)  # 查询深度可分离卷积

        self.q1 = nn.Conv2d(c, c, kernel_size=1)  # 第二个查询卷积层
        self.q1_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)  # 第二个查询深度可分离卷积

        self.q2 = nn.Conv2d(c, c, kernel_size=1)  # 第三个查询卷积层
        self.q2_dwconv = nn.Conv2d(c, c, 3, padding=1, groups=c)  # 第三个查询深度可分离卷积

        # 输出的卷积层
        self.project_out = nn.Conv2d(c, c, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 温度系数，用于调节注意力权重
        self.project_out1 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out2 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # SimpleGate模块（自定义门控机制）
        self.sg = SimpleGate()

        # FFN扩展卷积层
        ffn_channel = FFN_Expand * c  # FFN扩展的通道数
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # 第二组FFN卷积层
        self.conv4_1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=True)
        self.conv5_1 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                 groups=1, bias=True)

        # 第三组FFN卷积层
        self.conv4_2 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                                 bias=True)
        self.conv5_2 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                                 groups=1, bias=True)

        # LayerNorm用于标准化
        self.normq = LayerNorm2d(c)
        self.normq1 = LayerNorm2d(c)
        self.normq2 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.norm2_1 = LayerNorm2d(c)
        self.norm2_2 = LayerNorm2d(c)

        # 缩放系数（可学习的参数）
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # 激活函数
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, encs):
        """
        前向传播函数，处理多尺度输入，计算注意力并生成输出。

        Args:
            encs (list of tensors): 输入的多尺度特征图，格式为 [enc0, enc1, enc2]
                - enc0: 大尺度特征图，尺寸 [B, C, H, W]
                - enc1: 中尺度特征图，尺寸 [B, C, H/2, W/2]
                - enc2: 小尺度特征图，尺寸 [B, C, H/4, W/4]

        Returns:
            outs (list of tensors): 输出的多尺度特征图，包含三个尺度的输出
                - outs[0]: 尺寸与 enc0 相同
                - outs[1]: 尺寸与 enc1 相同
                - outs[2]: 尺寸与 enc2 相同
        """

        # 获取输入的三个尺度的特征图
        enc0 = encs[0]  # 尺寸 [B, C, H, W]
        enc1 = nn.Upsample(scale_factor=2)(encs[1])  # 上采样，尺寸 [B, C, H, W]
        enc2 = nn.Upsample(scale_factor=4)(encs[2])  # 上采样，尺寸 [B, C, H, W]

        # 对不同尺度的特征进行 LayerNorm 标准化
        q = self.normq(enc0)  # 尺寸 [B, C, H, W]
        q1 = self.normq1(enc1)  # 尺寸 [B, C, H, W]
        q2 = self.normq2(enc2)  # 尺寸 [B, C, H, W]

        # 拼接查询、键和值并进行卷积操作
        kv_attn = torch.cat((q, q1, q2), dim=1)  # 尺寸 [B, 3*C, H, W]
        kv = self.kv_dwconv(self.kv(kv_attn))  # 通道数变为 [B, 6*C, H, W]
        k, v = kv.chunk(2, dim=1)  # 分离键和值，尺寸分别为 [B, 3*C, H, W]

        # 重排键和值为多头注意力的格式
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 尺寸 [B, num_heads, C, H*W]
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 尺寸 [B, num_heads, C, H*W]
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化键

        # 对查询进行卷积和归一化
        q = self.q_dwconv(self.q(q))  # 尺寸 [B, C, H, W]
        b, c_q, h, w = q.shape  # 获取查询的尺寸
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 尺寸 [B, num_heads, C, H*W]
        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化查询

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 尺寸 [B, num_heads, H*W, H*W]
        attn = self.relu(attn)  # ReLU激活
        attn = self.softmax(attn)  # Softmax计算注意力
        out = (attn @ v)  # 尺寸 [B, num_heads, C, H*W]

        # 重排为输出格式
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 尺寸 [B, C, H, W]
        x = self.project_out(out)  # 通过卷积进行投影，尺寸 [B, C, H, W]
        y = enc0 + x * self.beta  # 残差连接，尺寸 [B, C, H, W]
        x = self.conv4(self.norm2(y))  # 通过前馈网络，尺寸 [B, FFN_Expand*C, H, W]
        x = self.sg(x)  # 应用门控机制
        x = self.conv5(x)  # 通道压缩，尺寸 [B, C, H, W]
        out0 = y + x * self.gamma  # 最终输出，尺寸 [B, C, H, W]

        # 对 q1 和 q2 进行相同的操作
        q1 = self.q1_dwconv(self.q1(q1))  # 尺寸 [B, C, H, W]
        b, c_q, h, w = q1.shape
        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 尺寸 [B, num_heads, C, H*W]
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        attn1 = (q1 @ k.transpose(-2, -1)) * self.temperature1
        attn1 = self.relu(attn1)
        attn1 = self.softmax(attn1)
        out1 = (attn1 @ v)  # 尺寸 [B, num_heads, C, H*W]
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x1 = self.project_out1(out1)  # 尺寸 [B, C, H, W]
        y1 = enc1 + x1 * self.beta1
        x1 = self.conv4_1(self.norm2_1(y1))
        x1 = self.sg(x1)
        x1 = self.conv5_1(x1)
        out1 = y1 + x1 * self.gamma1  # 输出1，尺寸 [B, C, H, W]

        # 对 q2 进行相同操作
        q2 = self.q2_dwconv(self.q2(q2))  # 尺寸 [B, C, H, W]
        b, c_q, h, w = q2.shape
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        attn2 = (q2 @ k.transpose(-2, -1)) * self.temperature2
        attn2 = self.relu(attn2)
        attn2 = self.softmax(attn2)
        out2 = (attn2 @ v)  # 尺寸 [B, num_heads, C, H*W]
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x2 = self.project_out1(out2)  # 尺寸 [B, C, H, W]
        y2 = enc2 + x2 * self.beta2
        x2 = self.conv4_2(self.norm2_2(y2))
        x2 = self.sg(x2)
        x2 = self.conv5_2(x2)
        out2 = y2 + x2 * self.gamma2  # 输出2，尺寸 [B, C, H, W]

        # 上采样各尺度输出
        out1 = nn.Upsample(scale_factor=0.5)(out1)  # 尺寸 [B, C, H, W]
        out2 = nn.Upsample(scale_factor=0.25)(out2)  # 尺寸 [B, C, H, W]

        # 返回所有输出
        outs = [out0, out1, out2]
        return outs  # 尺寸 [B, C, H, W]， [B, C, H/2, W/2]， [B, C, H/4, W/4]


class lateral_nafblock(nn.Module):
    """
    该类实现了一个侧向的NAF块（Lateral NAFBlock），
    通过多个 `qkvblock`（自注意力模块）堆叠来处理输入的特征图。每个 `qkvblock` 模块使用多头自注意力机制来捕捉输入特征图之间的关系，
    并生成经过变换的输出特征图。

    Args:
        c (int): 输入特征图的通道数。
        num_heads (int, 可选): 多头注意力的头数，默认为3。
        num_block (int, 可选): 堆叠的 `qkvblock` 模块数量，默认为1。

    Attributes:
        num_heads (int): 多头注意力的头数。
        qkv (nn.Sequential): 由多个 `qkvblock` 模块组成的序列，每个 `qkvblock` 模块都执行自注意力机制。

    """

    def __init__(self, c, num_heads=3, num_block=1):
        """
        初始化函数，设置 `qkvblock` 模块的堆叠，并指定每个模块的头数。

        Args:
            c (int): 输入特征图的通道数。
            num_heads (int): 多头注意力的头数，默认为3。
            num_block (int): 堆叠的 `qkvblock` 模块数量，默认为1。
        """
        super().__init__()

        self.num_heads = num_heads  # 多头注意力的头数，默认为3

        # 创建多个 `qkvblock` 模块，依次堆叠
        self.qkv = nn.Sequential(
            *[qkvblock(c, num_heads=num_heads) for _ in range(num_block)]
        )

    def forward(self, encs):
        """
        前向传播函数，处理输入的特征图，并通过堆叠的 `qkvblock` 模块进行特征提取。

        Args:
            encs (list of tensors): 输入的特征图列表，其中每个特征图的尺寸为 (B, C, H, W)，
                                    B 是批次大小，C 是通道数，H 是高度，W 是宽度。

        Returns:
            outs (list of tensors): 经过多个 `qkvblock` 模块处理后的输出特征图，尺寸与输入相同。
        """
        outs = encs  # 初始输出为输入特征图

        # 依次通过每个 `qkvblock` 模块进行处理
        for qkv in self.qkv:
            outs = qkv(outs)  # 每次输出经过一个 qkvblock 处理，输出尺寸不变

        return outs


class S_CEMBlock(nn.Module):
    """
    该类实现了一个结合了多头自注意力（Multi-head Attention）和前馈网络（FFN）的模块，
    其核心思想是利用两个不同的注意力机制对输入进行处理，并通过 `SimpleGate` 和残差连接进行增强。

    Args:
        c (int): 输入特征图的通道数。
        DW_Expand (int, 可选): 深度可分离卷积扩展因子，默认为2。
        num_heads (int, 可选): 多头注意力的头数，默认为3。
        FFN_Expand (int, 可选): FFN扩展因子，默认为2。
        drop_out_rate (float, 可选): Dropout的比率，默认为0（即不使用Dropout）。

    Attributes:
        qkv (nn.Conv2d): 用于生成查询、键、值的卷积层。
        qkv_dwconv (nn.Conv2d): 深度可分离卷积，处理查询、键、值的输出。
        project_out (nn.Conv2d): 用于投影注意力输出的卷积层。
        temperature (nn.Parameter): 用于调整多头注意力的温度参数。
        sg (SimpleGate): 用于输入门控机制。
        conv4 (nn.Conv2d): 用于前馈网络的卷积层。
        conv5 (nn.Conv2d): 用于前馈网络的卷积层。
        norm1 (LayerNorm2d): 第一层归一化层。
        norm2 (LayerNorm2d): 第二层归一化层。
        dropout1 (nn.Dropout/nn.Identity): 用于第一层的Dropout（如果 `drop_out_rate` > 0）。
        dropout2 (nn.Dropout/nn.Identity): 用于第二层的Dropout（如果 `drop_out_rate` > 0）。
        beta, beta2 (nn.Parameter): 控制残差连接的权重。
        gamma (nn.Parameter): 控制前馈网络输出的权重。
        relu (nn.ReLU): 激活函数。
        softmax (nn.Softmax): Softmax激活函数，应用于注意力计算。
    """

    def __init__(self, c, DW_Expand=2, num_heads=3, FFN_Expand=2, drop_out_rate=0.):
        """
        初始化 `S_CEMBlock` 模块，设置卷积层、注意力机制以及前馈网络。

        Args:
            c (int): 输入特征图的通道数。
            DW_Expand (int, 可选): 深度可分离卷积扩展因子，默认为2。
            num_heads (int, 可选): 多头注意力的头数，默认为3。
            FFN_Expand (int, 可选): FFN扩展因子，默认为2。
            drop_out_rate (float, 可选): Dropout的比率，默认为0（即不使用Dropout）。
        """
        super().__init__()

        self.num_heads = num_heads  # 多头注意力的头数

        # 查询、键、值的生成卷积层
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, kernel_size=3, stride=1, padding=1, groups=c * 3)

        # 输出投影卷积层
        self.project_out = nn.Conv2d(c, c, kernel_size=1)

        # 注意力的温度参数（可调）
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out2 = nn.Conv2d(c, c, kernel_size=1)
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # SimpleGate，用于门控机制
        self.sg = SimpleGate()

        # 前馈网络的卷积层
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # 归一化层
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout层（如果drop_out_rate > 0，则使用，否则使用nn.Identity）
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 残差连接中的缩放参数
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta2 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # 激活函数
        self.relu = nn.ReLU()

        # Softmax用于注意力计算
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        """
        前向传播函数，进行多头自注意力计算、前馈网络计算和门控操作。

        Args:
            inp (tensor): 输入的特征图，形状为 (B, C, H, W)，
                          其中 B 是批次大小，C 是通道数，H 是高度，W 是宽度。

        Returns:
            y (tensor): 输出特征图，形状与输入相同 (B, C, H, W)。
        """
        x = inp  # 输入特征图
        x = self.norm1(x)  # 归一化层1，输出尺寸 (B, C, H, W)
        b, c, h, w = x.shape  # 获取输入特征图的形状

        # 通过卷积层生成查询、键、值
        qkv = self.qkv_dwconv(self.qkv(x))  # 通过 qkv 卷积层，输出尺寸 (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # 将输出切分为 q, k, v，尺寸为 (B, C, H, W)

        # 对查询、键、值进行排列以适应多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)

        # 查询和键分别进行转置，便于计算注意力
        qs = q.clone().permute(0, 1, 3, 2)  # (B, num_heads, H*W, C)
        ks = k.clone().permute(0, 1, 3, 2)  # (B, num_heads, H*W, C)
        vs = v.clone().permute(0, 1, 3, 2)  # (B, num_heads, H*W, C)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)  # (B, num_heads, C, H*W)
        k = torch.nn.functional.normalize(k, dim=-1)  # (B, num_heads, C, H*W)

        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, num_heads, H*W, H*W)
        attn = self.relu(attn)  # 激活函数
        attn = self.softmax(attn)  # Softmax 归一化

        # 通过注意力加权求得输出
        outc = (attn @ v)  # (B, num_heads, C, H*W)

        # 对 qs 和 ks 进行归一化，并计算第二种注意力
        qs = torch.nn.functional.normalize(qs, dim=-1)  # (B, num_heads, H*W, C)
        ks = torch.nn.functional.normalize(ks, dim=-1)  # (B, num_heads, H*W, C)

        # 计算第二种注意力得分
        attns = (qs @ ks.transpose(-2, -1)) * self.temperature2  # (B, num_heads, H*W, H*W)
        attns = self.relu(attns)  # 激活函数
        attns = self.softmax(attns)  # Softmax 归一化

        # 通过第二种注意力加权求得输出
        outs = (attns @ vs)  # (B, num_heads, C, H*W)
        outs = outs.permute(0, 1, 3, 2)  # (B, num_heads, H*W, C)

        # 将多头输出重排列为通道数为 C 的输出
        outc = rearrange(outc, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outs = rearrange(outs, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 投影和残差连接
        xc = self.project_out(outc)  # (B, C, H, W)
        xc = self.dropout1(xc)  # Dropout
        xs = self.project_out2(outs)  # (B, C, H, W)
        xs = self.dropout1(xs)  # Dropout

        # 残差连接的加权
        y = inp + xc * self.beta + xs * self.beta2  # (B, C, H, W)

        # 前馈网络操作
        x = self.conv4(self.norm2(y))  # (B, FFN_Channel, H, W)
        x = self.sg(x)  # SimpleGate的门控操作
        x = self.conv5(x)  # (B, C, H, W)

        x = self.dropout2(x)  # Dropout

        # 最终输出，通过残差连接
        return y + x * self.gamma  # (B, C, H, W)


class CEMBlock(nn.Module):
    """
    CEMBlock 是一个结合了多头自注意力机制（Multi-head Attention）和前馈网络（FFN）的模块。
    它使用 SimpleGate 进行动态门控，结合残差连接提升网络性能。

    Args:
        c (int): 输入特征图的通道数。
        DW_Expand (int, 可选): 深度可分离卷积扩展因子，默认为2（目前未在代码中直接使用）。
        num_heads (int, 可选): 多头注意力的头数，默认为3。
        FFN_Expand (int, 可选): 前馈网络扩展因子，默认为2。
        drop_out_rate (float, 可选): Dropout比率，默认为0（即不使用Dropout）。

    Attributes:
        qkv (nn.Conv2d): 用于生成查询（Q）、键（K）和值（V）的卷积层。
        qkv_dwconv (nn.Conv2d): 深度可分离卷积，用于处理查询、键、值的输出。
        project_out (nn.Conv2d): 用于输出特征的卷积层。
        temperature (nn.Parameter): 多头注意力的温度参数，用于调整注意力计算。
        sg (SimpleGate): 动态门控机制模块，用于控制网络的输出。
        conv4 (nn.Conv2d): 用于前馈网络计算的卷积层。
        conv5 (nn.Conv2d): 用于前馈网络计算的卷积层。
        norm1 (LayerNorm2d): 第一层归一化层。
        norm2 (LayerNorm2d): 第二层归一化层。
        dropout1 (nn.Dropout/nn.Identity): 第一层Dropout，如果drop_out_rate > 0，否则为Identity。
        dropout2 (nn.Dropout/nn.Identity): 第二层Dropout，如果drop_out_rate > 0，否则为Identity。
        beta (nn.Parameter): 控制输入和输出之间的残差连接的参数。
        gamma (nn.Parameter): 控制前馈网络输出的缩放参数。
        relu (nn.ReLU): 激活函数ReLU。
        softmax (nn.Softmax): 用于注意力计算的Softmax函数。
    """

    def __init__(self, c, DW_Expand=2, num_heads=3, FFN_Expand=2, drop_out_rate=0.):
        """
        初始化 CEMBlock 模块，设置卷积层、注意力机制、前馈网络、门控机制等模块。

        Args:
            c (int): 输入特征图的通道数。
            DW_Expand (int, 可选): 深度可分离卷积扩展因子，默认为2。
            num_heads (int, 可选): 多头注意力的头数，默认为3。
            FFN_Expand (int, 可选): 前馈网络扩展因子，默认为2。
            drop_out_rate (float, 可选): Dropout比率，默认为0（即不使用Dropout）。
        """
        super().__init__()

        self.num_heads = num_heads  # 设置多头注意力的头数

        # 生成查询、键、值的卷积层，qkv的输出通道数是c的三倍
        self.qkv = nn.Conv2d(c, c * 3, kernel_size=1)
        # 深度可分离卷积操作，输出通道数是c的三倍
        self.qkv_dwconv = nn.Conv2d(c * 3, c * 3, 3, padding=1, groups=c * 3)
        # 输出特征的投影卷积层
        self.project_out = nn.Conv2d(c, c, kernel_size=1)
        # 多头注意力的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # SimpleGate，用于控制输出的门控机制
        self.sg = SimpleGate()

        # 前馈网络的卷积层
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # 归一化层
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout层（如果drop_out_rate > 0，则使用Dropout，否则使用nn.Identity）
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 残差连接的缩放参数
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # 激活函数ReLU
        self.relu = nn.ReLU()

        # Softmax用于注意力计算
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        """
        前向传播函数，执行自注意力计算、前馈网络计算、门控机制、残差连接等操作。

        Args:
            inp (tensor): 输入特征图，形状为 (B, C, H, W)，
                          其中 B 是批次大小，C 是通道数，H 是高度，W 是宽度。

        Returns:
            y (tensor): 输出特征图，形状与输入相同 (B, C, H, W)。
        """
        x = inp  # 输入特征图
        x = self.norm1(x)  # 第一层归一化，输出尺寸为 (B, C, H, W)

        b, c, h, w = x.shape  # 获取输入特征图的形状

        # 通过卷积层生成查询、键、值，并使用深度可分离卷积进行处理
        qkv = self.qkv_dwconv(self.qkv(x))  # 形状为 (B, 3*C, H, W)
        q, k, v = qkv.chunk(3, dim=1)  # 将输出切分为 q, k, v，形状为 (B, C, H, W)

        # 对查询、键、值进行重排，适应多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # (B, num_heads, C, H*W)

        # 对查询和键进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)  # (B, num_heads, C, H*W)
        k = torch.nn.functional.normalize(k, dim=-1)  # (B, num_heads, C, H*W)

        # 计算注意力得分并应用温度参数
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, num_heads, H*W, H*W)
        attn = self.relu(attn)  # 激活函数
        attn = self.softmax(attn)  # Softmax 归一化

        # 使用注意力权重加权值
        out = (attn @ v)  # (B, num_heads, C, H*W)

        # 将多头输出重排列为通道数为C的输出
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # (B, C, H, W)

        # 投影输出，并应用Dropout
        x = self.project_out(out)  # (B, C, H, W)
        x = self.dropout1(x)  # Dropout

        # 残差连接加权
        y = inp + x * self.beta  # (B, C, H, W)

        # 前馈网络计算
        x = self.conv4(self.norm2(y))  # (B, FFN_Channel, H, W)
        x = self.sg(x)  # SimpleGate的门控操作
        x = self.conv5(x)  # (B, C, H, W)

        # 应用Dropout
        x = self.dropout2(x)  # Dropout

        # 最终输出，结合残差连接
        return y + x * self.gamma  # (B, C, H, W)


@ARCH_REGISTRY.register()
class HAUNet(nn.Module):
    """
    HAUNet 类：实现超分辨率网络，用于图像的上采样任务。
    本网络结合了多层编码器-解码器结构，使用了多个基于注意力的模块（CEMBlock、S_CEMBlock、lateral_nafblock），
    并且通过使用卷积转置层和像素重排（PixelShuffle）来实现图像的上采样。
    """

    def __init__(self, up_scale=4, img_channel=3, width=180, middle_blk_num=10, enc_blk_nums=[5,5], dec_blk_nums=[5,5], heads = [1,2,4]):
        """
        初始化 HAUNet 模块。

        Args:
            up_scale (int): 上采样倍数，默认值为4。
            img_channel (int): 输入图像的通道数，通常为3（RGB图像）。
            width (int): 初始通道数。
            middle_blk_num (int): 网络中的中间块数量。
            enc_blk_nums (list): 每一层编码器的块数。
            dec_blk_nums (list): 每一层解码器的块数。
            heads (list): 每层编码器和解码器中的头数。
        """
        super(HAUNet, self).__init__()

        # 介绍卷积层：将输入的图像通道数变换为指定的宽度
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        # 初始化编码器、解码器、上采样和下采样模块
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width  # 初始通道数
        ii = 0  # 用于控制编码器的层次

        # 构建编码器部分
        for numii in range(len(enc_blk_nums)):
            num = enc_blk_nums[numii]
            if numii < 1:
                # 第一层使用 S_CEMBlock
                self.encoders.append(
                    nn.Sequential(
                        *[S_CEMBlock(chan, num_heads=heads[ii]) for _ in range(num)]
                    )
                )
            else:
                # 后续层使用 CEMBlock
                self.encoders.append(
                    nn.Sequential(
                        *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(num)]
                    )
                )
            # 每一层后连接下采样操作
            self.downs.append(
                nn.Conv2d(chan, chan, 2, 2)
            )
            ii += 1  # 编码器层数计数器

        # 横向NAF模块（lateral_nafblock）连接
        self.lateral_nafblock = lateral_nafblock(chan)

        # 中间编码和解码块
        self.enc_middle_blks = \
            nn.Sequential(
                *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)]
            )
        self.dec_middle_blks = \
            nn.Sequential(
                *[CEMBlock(chan, num_heads=heads[ii]) for _ in range(middle_blk_num // 2)]
            )

        ii = 0
        # 构建解码器部分
        for numii in range(len(dec_blk_nums)):
            num = dec_blk_nums[numii]
            # 上采样操作（卷积转置）
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(chan, chan, kernel_size=2, stride=2)
                )
            )
            # 构建解码器块
            if numii < 1:
                self.decoders.append(
                    nn.Sequential(
                        *[CEMBlock(chan, num_heads=heads[1 - ii]) for _ in range(num)]
                    )
                )
            else:
                self.decoders.append(
                    nn.Sequential(
                        *[S_CEMBlock(chan, num_heads=heads[1 - ii]) for _ in range(num)]
                    )
                )
            ii += 1  # 解码器层数计数器

        self.dec_blk_nums = dec_blk_nums  # 保存解码器块的数量
        # 计算填充大小
        self.padder_size = 2 ** len(self.encoders)

        # 最后的上采样模块，将特征图重建为所需尺寸
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale ** 2, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale  # 上采样倍数

    def forward(self, inp):
        """
        前向传播过程：根据输入图像执行编码、解码和上采样过程，最后得到超分辨率图像。

        Args:
            inp (torch.Tensor): 输入图像，形状为 (N, C, H, W)，N为批大小，C为通道数，H为高度，W为宽度。

        Returns:
            torch.Tensor: 超分辨率后的输出图像，形状为 (N, C, H * up_scale, W * up_scale)。
        """
        # 先进行上采样，得到高分辨率的输入图像
        inp_hr = F.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')

        # 初始卷积：将输入图像通道数转换为指定的宽度
        x = self.intro(inp)

        encs = []  # 存储编码器的输出

        # 编码器阶段
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)  # 通过编码器块进行处理
            encs.append(x)  # 将编码器输出存储
            x = down(x)  # 下采样操作

        # 经过中间编码块处理
        x = self.enc_middle_blks(x)
        encs.append(x)  # 保存最后的输出
        # 通过横向NAF模块连接编码器的输出
        outs = self.lateral_nafblock(encs)
        x = outs[-1]  # 获取最后的输出

        # 解码器阶段
        x = self.dec_middle_blks(x)  # 中间解码块处理
        outs2 = outs[:2]  # 获取编码器的前两层输出
        for decoder, up, enc_skip in zip(self.decoders, self.ups, outs2[::-1]):
            # 上采样操作
            x = up(x)
            # 加入跳跃连接
            x = x + enc_skip
            # 通过解码器块进行处理
            x = decoder(x)

        # 最后的上采样：将特征图恢复到目标尺寸
        x = self.up(x)
        x = x + inp_hr  # 加上高分辨率的输入图像作为残差

        return x  # 返回超分辨率的输出图像


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    from thop import profile

    net = HAUNet(up_scale=4, width=96, enc_blk_nums=[5, 5], dec_blk_nums=[5, 5], middle_blk_num=10).cuda()
    torch.cuda.reset_max_memory_allocated()
    x = torch.rand(1, 3, 64, 64).cuda()
    y = net(x)
    # 获取模型最大内存消耗
    max_memory_reserved = torch.cuda.max_memory_reserved(device='cuda') / (1024 ** 2)

    print(f"模型最大内存消耗: {max_memory_reserved:.2f} MB")
    flops, params = profile(net, (x,))
    print('flops: %.4f G, params: %.4f M' % (flops / 1e9, params / 1000000.0))