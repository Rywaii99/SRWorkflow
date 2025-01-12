import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from tqdm import tqdm

from archs.inception import InceptionV3


def load_patched_inception_v3(device='cuda', resize_input=True, normalize_input=False):
    """
    加载经过修补的 InceptionV3 模型。

    该函数加载一个修改过的 InceptionV3 模型，并返回模型实例。该模型可以用于计算生成图像的特征。

    Args:
        device (str): 设备类型，默认为 'cuda'，表示使用 GPU 进行计算。
        resize_input (bool): 是否对输入图像进行缩放。默认值为 `True`，表示会缩放输入图像。
        normalize_input (bool): 是否对输入图像进行归一化处理。默认值为 `False`，表示不进行归一化。

    Returns:
        nn.Module: 返回加载好的 InceptionV3 模型。
    """
    # 创建一个 InceptionV3 模型，传入参数控制是否对输入进行缩放和归一化
    inception = InceptionV3([3], resize_input=resize_input, normalize_input=normalize_input)
    # 使用 DataParallel 包裹模型，支持多 GPU 训练
    inception = nn.DataParallel(inception).eval().to(device)
    return inception


@torch.no_grad()
def extract_inception_features(data_generator, inception, len_generator=None, device='cuda'):
    """
    提取图像的 Inception 特征。

    使用 InceptionV3 模型提取图像的特征，并返回所有图像的特征矩阵。

    Args:
        data_generator (generator): 数据生成器，用于提供批量数据。
        inception (nn.Module): 已加载的 InceptionV3 模型。
        len_generator (int, 可选): 数据生成器的长度，用于显示进度条。如果为 `None`，则不显示进度条。
        device (str): 设备类型，默认为 'cuda'，表示使用 GPU 进行计算。

    Returns:
        Tensor: 提取的特征矩阵，形状为 `(num_samples, feature_dim)`，即每个样本的特征。
    """
    if len_generator is not None:
        # 如果提供了数据生成器的长度，则显示进度条
        pbar = tqdm(total=len_generator, unit='batch', desc='Extract')
    else:
        pbar = None
    features = []  # 存储所有提取的特征

    # 遍历数据生成器中的数据
    for data in data_generator:
        if pbar:
            pbar.update(1)  # 更新进度条
        data = data.to(device)  # 将数据移动到指定设备上
        feature = inception(data)[0].view(data.shape[0], -1)  # 提取特征并展平为一维向量
        features.append(feature.to('cpu'))  # 将特征移回 CPU 并存储

    if pbar:
        pbar.close()  # 关闭进度条

    # 将所有特征拼接成一个大矩阵
    features = torch.cat(features, 0)
    return features


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    计算 Frechet 距离（FID）分数。

    FID 分数用于衡量两组图像（如真实图像和生成图像）之间的相似度，越低表示图像之间越相似。

    FID 计算公式：
    d^2 = ||mu1 - mu2||^2 + Tr(C1 + C2 - 2*sqrt(C1*C2))

    Args:
        mu1 (np.array): 第一组图像的均值向量（通过激活值计算得到）。
        sigma1 (np.array): 第一组图像的协方差矩阵（通过激活值计算得到）。
        mu2 (np.array): 第二组图像的均值向量（例如，真实图像的均值向量）。
        sigma2 (np.array): 第二组图像的协方差矩阵（例如，真实图像的协方差矩阵）。
        eps (float): 用于避免协方差矩阵奇异的正则化项。默认值为 `1e-6`。

    Returns:
        float: 计算得到的 FID 距离值。
    """
    # 确保均值和协方差矩阵的形状相同
    assert mu1.shape == mu2.shape, 'Two mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, ('Two covariances have different dimensions')

    # 计算协方差矩阵的平方根
    cov_sqrt, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # 如果协方差矩阵的乘积几乎是奇异的，进行数值修正
    if not np.isfinite(cov_sqrt).all():
        print(f'Product of cov matrices is singular. Adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # 数值误差可能导致协方差矩阵包含虚部，进行修正
    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))  # 获取虚部的最大值
            raise ValueError(f'Imaginary component {m}')
        cov_sqrt = cov_sqrt.real  # 取实部

    # 计算均值差异的平方
    mean_diff = mu1 - mu2
    mean_norm = mean_diff @ mean_diff  # 均值差的平方

    # 计算协方差矩阵的迹
    trace = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(cov_sqrt)

    # 计算 FID 距离
    fid = mean_norm + trace

    return fid
