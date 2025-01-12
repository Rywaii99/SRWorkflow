# Modified from https://github.com/mseitzer/pytorch-fid/blob/master/pytorch_fid/inception.py # noqa: E501
# For FID metric

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from torchvision import models

# FID计算中使用的Inception模型的预训练权重下载链接
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501
# 本地预训练Inception模型路径
LOCAL_FID_WEIGHTS = 'experiments/pretrained_models/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501


class InceptionV3(nn.Module):
    """使用预训练的InceptionV3网络返回特征图"""

    # 默认返回的Inception块索引，对应的是最终平均池化层的输出
    DEFAULT_BLOCK_INDEX = 3

    # 根据特征维度返回的Inception块的索引映射
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # 第一个最大池化层的特征
        192: 1,  # 第二个最大池化层的特征
        768: 2,  # 输入到辅助分类器的特征
        2048: 3  # 最终平均池化层的特征
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """构建预训练的InceptionV3模型。

        Args:
            output_blocks (list[int]): 返回的特征图的块的索引。
                - 0: 对应于第一个最大池化层的输出
                - 1: 对应于第二个最大池化层的输出
                - 2: 对应于输入到辅助分类器的输出
                - 3: 对应于最终平均池化层的输出
            resize_input (bool): 如果为True，输入将被缩放为299x299像素。
                因为网络在没有完全连接层的情况下是完全卷积的，因此可以处理任意大小的输入，所以该参数不一定是必须的，默认为True。
            normalize_input (bool): 如果为True，将输入范围从(0, 1)缩放到Inception网络预期的范围(-1, 1)。
                默认为True。
            requires_grad (bool): 如果为True，网络的参数将计算梯度，这对于微调网络可能是有用的。默认为False。
            use_fid_inception (bool): 如果为True，使用TensorFlow FID实现中使用的预训练Inception模型。
                如果为False，则使用torchvision中的Inception模型。FID Inception模型具有不同的权重和稍微不同的结构。
                如果您想计算FID分数，建议将此参数设置为True，以获得可比较的结果。默认为True。
        """
        super(InceptionV3, self).__init__()

        # 初始化参数
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        # 确保请求的块索引不超过3
        assert self.last_needed_block <= 3, ('Last possible output block index is 3')

        # 存储网络的不同块
        self.blocks = nn.ModuleList()

        # 根据是否使用FID Inception选择模型
        if use_fid_inception:
            inception = fid_inception_v3()  # 使用FID版本的Inception模型
        else:
            # 使用torchvision中的Inception模型
            try:
                inception = models.inception_v3(pretrained=True, init_weights=False)
            except TypeError:
                inception = models.inception_v3(pretrained=True)

        # 第一块：输入到第一个最大池化层
        block0 = [
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # 第二块：第一个最大池化到第二个最大池化
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))

        # 第三块：第二个最大池化到辅助分类器
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # 第四块：辅助分类器到最终平均池化层
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        # 设置网络中所有参数是否计算梯度
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """获取Inception特征图。

        Args:
            x (Tensor): 输入张量，形状为 (b, 3, h, w)，其中b是批次大小，3是通道数，h是高度，w是宽度。
                输入值应处于范围(-1, 1)，如果normalize_input=True，可以输入范围(0, 1)的图像。

        Returns:
            list[Tensor]: 返回选择的输出块的特征图，按块索引升序排序。
        """
        output = []

        # 如果需要调整输入大小，则将输入图像调整为299x299
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # 如果需要归一化输入，则将输入从(0, 1)缩放到(-1, 1)
        if self.normalize_input:
            x = 2 * x - 1  # 缩放从(0, 1)到(-1, 1)

        # 通过每个块计算特征
        for idx, block in enumerate(self.blocks):
            x = block(x)
            # 如果当前块的索引在输出块列表中，则保存该块的特征
            if idx in self.output_blocks:
                output.append(x)

            # 如果达到所需的最后一个块，停止计算
            if idx == self.last_needed_block:
                break

        return output


def fid_inception_v3():
    """构建用于FID计算的预训练Inception模型。

    FID计算使用的Inception模型与torchvision中的Inception模型不同，它使用不同的权重，并且结构稍微有所不同。
    此方法首先构建torchvision的Inception模型，然后修改其必要的部分，使其符合FID计算的要求。
    """
    # 构建Inception模型
    try:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False, init_weights=False)
    except TypeError:
        inception = models.inception_v3(num_classes=1008, aux_logits=False, pretrained=False)

    # 修改Inception的各个层，以适应FID计算的要求
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)

    # 加载预训练权重
    if os.path.exists(LOCAL_FID_WEIGHTS):
        state_dict = torch.load(LOCAL_FID_WEIGHTS, map_location=lambda storage, loc: storage)
    else:
        state_dict = load_url(FID_WEIGHTS_URL, progress=True)

    inception.load_state_dict(state_dict)
    return inception


class FIDInceptionA(models.inception.InceptionA):
    """适用于FID计算的InceptionA块"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # 修复：TensorFlow的平均池化不包括填充零的部分
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    """适用于FID计算的InceptionC块"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # 修复：TensorFlow的平均池化不包括填充零的部分
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(models.inception.InceptionE):
    """用于FID计算的InceptionE_1块"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # 修复：TensorFlow的平均池化不包括填充零的部分
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(models.inception.InceptionE):
    """用于FID计算的InceptionE_2块"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # 修复：FID Inception模型使用最大池化而不是平均池化
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)