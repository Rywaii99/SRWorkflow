import torch
import torch.nn as nn
from archs.arch_util import default_conv, Upsample
from utils.registry import ARCH_REGISTRY


## 通道注意力（CA）模块
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        CALayer 是通道注意力层，用于自适应地调整通道的权重。

        Args:
            channel (int): 输入特征图的通道数
            reduction (int): 用于缩小通道数的因子，默认值为16
        """
        super(CALayer, self).__init__()
        # 全局平均池化层：将每个通道的特征图压缩成一个值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 用于通道缩放的两层卷积
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()  # 使用Sigmoid函数进行归一化，得到通道权重
        )

    def forward(self, x):
        """
        前向传播：计算通道注意力，并与输入特征图相乘。

        Args:
            x (Tensor): 输入特征图，形状为 (batch_size, channel, height, width)

        Returns:
            Tensor: 通道加权后的特征图
        """
        y = self.avg_pool(x)  # 进行全局平均池化
        y = self.conv_du(y)  # 通过卷积层生成通道权重
        return x * y  # 将输入特征图与通道权重相乘


## 层注意力模块（LAM）
class LAM_Module(nn.Module):
    """ 层注意力模块（Layer Attention Module）"""

    def __init__(self, in_dim):
        """
        初始化LAM模块。

        Args:
            in_dim (int): 输入的特征图的通道数
        """
        super(LAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))  # 可训练的参数，用于调整注意力的强度
        self.softmax = nn.Softmax(dim=-1)  # 用于计算注意力权重

    def forward(self, x):
        """
        前向传播：计算层注意力，并与输入特征图相加。

        Args:
            x (Tensor): 输入特征图，形状为 (batch_size, N, C, H, W)

        Returns:
            Tensor: 加权后的特征图
        """
        m_batchsize, N, C, height, width = x.size()
        proj_query = x.view(m_batchsize, N, -1)  # 将输入变形为 (batch_size, N, C*H*W)
        proj_key = x.view(m_batchsize, N, -1).permute(0, 2, 1)  # 计算查询和键的注意力
        energy = torch.bmm(proj_query, proj_key)  # 计算能量矩阵
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy  # 缩放能量矩阵
        attention = self.softmax(energy_new)  # 计算注意力权重
        proj_value = x.view(m_batchsize, N, -1)  # 投影值

        out = torch.bmm(attention, proj_value)  # 对投影值应用注意力
        out = out.view(m_batchsize, N, C, height, width)  # 恢复特征图的形状

        out = self.gamma * out + x  # 调整输出并加上残差
        out = out.view(m_batchsize, -1, height, width)
        return out


## 通道-空间注意力模块（CSAM）
class CSAM_Module(nn.Module):
    """ 通道-空间注意力模块（Channel-Spatial Attention Module）"""

    def __init__(self, in_dim):
        """
        初始化CSAM模块。

        Args:
            in_dim (int): 输入特征图的通道数
        """
        super(CSAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 使用3D卷积进行空间注意力处理
        self.conv = nn.Conv3d(1, 1, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可训练的参数，用于调整注意力的强度
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, x):
        """
        前向传播：计算通道-空间注意力，并与输入特征图相加。

        Args:
            x (Tensor): 输入特征图，形状为 (batch_size, C, H, W)

        Returns:
            Tensor: 加权后的特征图
        """
        m_batchsize, C, height, width = x.size()
        out = x.unsqueeze(1)  # 扩展维度以适应3D卷积
        out = self.sigmoid(self.conv(out))  # 计算空间注意力

        out = self.gamma * out  # 调整注意力的强度
        out = out.view(m_batchsize, -1, height, width)  # 恢复特征图的形状
        x = x * out + x  # 将注意力加权后的特征图与原始特征图相加
        return x


## 残差通道注意力块（RCAB）
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        """
        初始化残差通道注意力块（RCAB）。

        Args:
            conv: 卷积操作
            n_feat (int): 特征图的通道数
            kernel_size (int): 卷积核的大小
            reduction (int): 通道注意力的缩放因子
            bias (bool): 是否使用偏置
            bn (bool): 是否使用批量归一化
            act: 激活函数
            res_scale (float): 残差缩放因子
        """
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))  # 添加卷积层
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))  # 如果使用批量归一化，则添加批量归一化层
            if i == 0:
                modules_body.append(act)  # 添加激活函数
        modules_body.append(CALayer(n_feat, reduction))  # 添加通道注意力层
        self.body = nn.Sequential(*modules_body)  # 将所有模块组成顺序模块
        self.res_scale = res_scale  # 残差缩放因子

    def forward(self, x):
        """
        前向传播：计算残差通道注意力块的输出。

        Args:
            x (Tensor): 输入特征图

        Returns:
            Tensor: 加权后的特征图
        """
        res = self.body(x)  # 通过卷积和通道注意力模块处理输入
        res += x  # 残差连接
        return res


## 残差组（RG）
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        """
        初始化残差组（Residual Group）。

        Args:
            conv: 卷积操作
            n_feat (int): 特征图的通道数
            kernel_size (int): 卷积核的大小
            reduction (int): 通道注意力的缩放因子
            act: 激活函数
            res_scale (float): 残差缩放因子
            n_resblocks (int): 残差块的数量
        """
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
            for _ in range(n_resblocks)]  # 创建多个残差通道注意力块
        modules_body.append(conv(n_feat, n_feat, kernel_size))  # 最后加上一个卷积层
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        前向传播：计算残差组的输出。

        Args:
            x (Tensor): 输入特征图

        Returns:
            Tensor: 加权后的特征图
        """
        res = self.body(x)  # 通过多个残差通道注意力块处理输入
        res += x  # 残差连接
        return res


## Holistic Attention Network（HAN）整体注意力网络
@ARCH_REGISTRY.register()
class HAN(nn.Module):
    def __init__(self, n_resgroups=10, n_resblocks=20, n_feats=128, reduction=16, scale=2, res_scale=1.0, conv=default_conv):
        """
        初始化HAN模型。

        Args:
            args (namespace): 配置文件或命令行参数，包含网络相关的超参数
            conv: 卷积操作，默认使用 `default_conv`
        """
        super(HAN, self).__init__()

        self.n_resgroups = n_resgroups
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        kernel_size = 3
        self.reduction = reduction
        self.scale = scale
        act = nn.ReLU(True)
        n_colors = 3

        # 定义头部模块
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # 定义主体模块（多个残差组）
        modules_body = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, act=act, res_scale=res_scale,
                          n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # 定义尾部模块
        modules_tail = [
            Upsample(scale, n_feats),
            conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.csa = CSAM_Module(n_feats)  # 使用通道-空间注意力模块
        self.la = LAM_Module(n_feats)  # 使用层注意力模块
        self.last_conv = nn.Conv2d(n_feats * 11, n_feats, 3, 1, 1)
        self.last = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        """
        前向传播：计算HAN网络的输出。

        Args:
            x (Tensor): 输入特征图

        Returns:
            Tensor: 输出特征图
        """
        x = self.head(x)  # 通过头部模块处理输入
        res = x
        for name, midlayer in self.body._modules.items():  # 通过主体模块处理输入
            res = midlayer(res)
            if name == '0':
                res1 = res.unsqueeze(1)
            else:
                res1 = torch.cat([res.unsqueeze(1), res1], 1)

        out1 = res
        res = self.la(res1)  # 通过层注意力模块
        out2 = self.last_conv(res)

        out1 = self.csa(out1)  # 通过通道-空间注意力模块
        out = torch.cat([out1, out2], 1)
        res = self.last(out)

        res += x  # 残差连接

        x = self.tail(res)  # 通过尾部模块恢复输出

        return x

    def load_state_dict(self, state_dict, strict=False):
        """
        加载模型参数。

        Args:
            state_dict (dict): 要加载的参数字典
            strict (bool): 是否严格检查键是否匹配
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
                        print('Replace pre-trained upsampler to new one...')  # 替换预训练的上采样器
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
