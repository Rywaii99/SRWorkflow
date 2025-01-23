import os
import torch
from collections import OrderedDict
from torch import nn as nn
from torchvision.models import vgg as vgg

from utils.registry import ARCH_REGISTRY

# 预训练VGG模型的路径
VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg19-dcbb9e9d.pth'

# VGG不同版本的层名配置
NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'pool5'
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
        'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
        'pool5'
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3', 'conv4_1',
        'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    ]
}


def insert_bn(names):
    """
    在每个卷积层后插入BN层。

    Args:
        names (list): 层名的列表。

    Returns:
        list: 添加了BN层后的层名列表。
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            # 如果当前层是卷积层，则在其后添加BN层
            position = name.replace('conv', '')
            names_bn.append('bn' + position)
    return names_bn


@ARCH_REGISTRY.register()
class VGGFeatureExtractor(nn.Module):
    """
    用于特征提取的VGG网络。

    本实现允许用户选择是否在输入特征中使用归一化以及VGG网络的类型。
    注意，预训练路径必须与VGG网络的类型相符。

    Args:
        layer_name_list (list[str]): 前向函数根据此列表返回对应的特征。
            例如：{'relu1_1', 'relu2_1', 'relu3_1'}。
        vgg_type (str): 设置VGG网络的类型，默认值是 'vgg19'。
        use_input_norm (bool): 如果为True，则对输入图像进行归一化。输入特征必须处于[0, 1]的范围内。默认值为True。
        range_norm (bool): 如果为True，将图像范围从[-1, 1]归一化到[0, 1]。默认值为False。
        requires_grad (bool): 如果为True，则VGG网络的参数会被优化。默认值为False。
        remove_pooling (bool): 如果为True，则移除VGG网络中的最大池化操作。默认值为False。
        pooling_stride (int): 最大池化操作的步幅，默认值为2。
    """

    def __init__(self,
                 layer_name_list,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 requires_grad=False,
                 remove_pooling=False,
                 pooling_stride=2):
        super(VGGFeatureExtractor, self).__init__()

        # 初始化参数
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # 根据所选VGG类型获取相应的层名
        self.names = NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            self.names = insert_bn(self.names)

        # 获取要使用的最大层索引
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx

        # 如果存在预训练模型，加载该模型的权重
        if os.path.exists(VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(pretrained=False)
            state_dict = torch.load(VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = getattr(vgg, vgg_type)(pretrained=True)

        # 获取VGG模型的前max_idx+1层
        features = vgg_net.features[:max_idx + 1]

        # 修改网络结构，去除或调整池化层
        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # 如果需要移除池化层，则跳过该层
                if remove_pooling:
                    continue
                else:
                    # 如果步幅需要调整，则修改池化层的步幅
                    modified_net[k] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        # 将修改后的VGG网络模块封装为Sequential
        self.vgg_net = nn.Sequential(modified_net)

        # 如果不需要梯度，则设置为评估模式，且不计算梯度
        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            # 如果需要梯度，则设置为训练模式
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        # 如果需要输入归一化，则使用VGG标准的均值和标准差进行归一化
        if self.use_input_norm:
            # VGG标准输入均值和标准差（适用于输入图像范围为[0, 1]的情况）
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (Tensor): 输入张量，形状为 (n, c, h, w)，其中n为批次大小，c为通道数，h为图像高度，w为图像宽度。

        Returns:
            Tensor: 前向传播结果，为一个包含中间特征的字典。
        """
        # 如果需要范围归一化，将图像范围从[-1, 1]归一化到[0, 1]
        if self.range_norm:
            x = (x + 1) / 2

        # 如果需要输入归一化，则按VGG标准进行均值和标准差归一化
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        output = {}
        # 遍历VGG网络的所有层
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            # 如果当前层的名称在要求的层列表中，则保存该层的输出
            if key in self.layer_name_list:
                output[key] = x.clone()

        return output
