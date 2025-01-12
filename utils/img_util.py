import cv2
import math
import numpy as np
import torch
import os
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    将 NumPy 数组格式的图像转换为 PyTorch 张量格式。

    Args:
        imgs (list[ndarray] | ndarray): 输入的图像，可以是单张图像（ndarray），
            或者是图像列表（list[ndarray]）。
        bgr2rgb (bool): 是否将 BGR 图像转换为 RGB 图像。默认为 True。
        float32 (bool): 是否将图像转换为 float32 类型。默认为 True。

    Returns:
        list[tensor] | tensor: 返回转换后的 PyTorch 张量。如果输入是多张图像，则返回包含多个张量的列表。
    """

    def _totensor(img, bgr2rgb, float32):
        """将单张图像转换为 PyTorch 张量"""
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')  # 如果图像是 float64 类型，将其转换为 float32
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 如果 bgr2rgb 为 True，转换 BGR 为 RGB 图像
        img = torch.from_numpy(img.transpose(2, 0, 1))  # 将 NumPy 数组转换为 PyTorch 张量，调整通道顺序为 (C, H, W)
        if float32:
            img = img.float()  # 如果 float32 为 True，将图像转换为 float 类型
        return img

    # 如果输入是图像列表，则逐个图像转换为张量
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    # 如果输入是单张图像，直接转换为张量
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    将 PyTorch 张量转换为 NumPy 数组格式的图像。
    在转换过程中，会将张量值限制在[min, max]范围内，并将值归一化到[0, 1]。

    Args:
        tensor (Tensor or list[Tensor]): 输入的张量，支持以下几种形状：
            1) 4D mini-batch 张量，形状为 (B x 3/1 x H x W)；
            2) 3D 张量，形状为 (3/1 x H x W)；
            3) 2D 张量，形状为 (H x W)。
            张量的通道顺序应该为 RGB 格式。
        rgb2bgr (bool): 是否将 RGB 图像转换为 BGR 图像。默认为 True。
        out_type (numpy type): 输出图像的类型。如果为 `np.uint8`，则输出图像的范围为 [0, 255]；如果为 `np.float32`，则输出范围为 [0, 1]。默认为 `np.uint8`。
        min_max (tuple[int]): 用于对图像进行裁剪的最小值和最大值，默认是 (0, 1)。

    Returns:
        (Tensor or list): 返回转换后的图像，可以是 3D 或 2D 的 NumPy 数组，通道顺序为 BGR。
    """

    # 检查输入是否为 Tensor 或 Tensor 列表
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    # 如果输入是单个 Tensor，将其转化为列表
    if torch.is_tensor(tensor):
        tensor = [tensor]

    result = []
    for _tensor in tensor:
        # 对张量进行处理
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)  # 去掉 batch 维度并进行数值裁剪
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])  # 归一化到 [0, 1]

        n_dim = _tensor.dim()  # 获取张量的维度
        if n_dim == 4:  # 如果是 4D 张量，表示多个图像的 mini-batch
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)  # 将通道转到最后一维
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 如果需要，转换 RGB 为 BGR
        elif n_dim == 3:  # 如果是 3D 张量，表示单张图像
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)  # 将通道转到最后一维
            if img_np.shape[2] == 1:  # 如果是灰度图像
                img_np = np.squeeze(img_np, axis=2)  # 去掉单通道
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 如果需要，转换 RGB 为 BGR
        elif n_dim == 2:  # 如果是 2D 张量，表示单通道图像
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')

        # 如果输出类型是 np.uint8，则将值映射到 [0, 255] 的范围并四舍五入
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()

        img_np = img_np.astype(out_type)  # 转换为指定的类型
        result.append(img_np)

    # 如果只返回一个结果，则直接返回该图像
    if len(result) == 1:
        result = result[0]

    return result


def imfrombytes(content, flag='color', float32=False):
    """从字节流读取图像。

    Args:
        content (bytes): 从文件或其他流中获得的图像字节数据。
        flag (str): 指定加载图像的颜色类型，可以是以下三种之一：
            - `color`: 彩色图像。
            - `grayscale`: 灰度图像。
            - `unchanged`: 原始图像（保持图像的原始通道数和颜色格式）。
        float32 (bool): 是否将图像转换为 `float32` 类型，并将其归一化到 [0, 1] 范围内。
            默认为 `False`，即不做归一化和类型转换。

    Returns:
        ndarray: 返回加载后的图像数组（`numpy` 数组）。
    """

    # 使用 numpy 的 frombuffer 将字节流转换为 uint8 类型的 numpy 数组
    img_np = np.frombuffer(content, np.uint8)

    # 定义不同标志对应的 OpenCV 图像读取方式
    imread_flags = {
        'color': cv2.IMREAD_COLOR,  # 加载彩色图像
        'grayscale': cv2.IMREAD_GRAYSCALE,  # 加载灰度图像
        'unchanged': cv2.IMREAD_UNCHANGED  # 加载原始图像（包括 alpha 通道）
    }

    # 使用 OpenCV 的 imdecode 函数解码字节数据为图像
    img = cv2.imdecode(img_np, imread_flags[flag])

    # 如果需要将图像转换为 float32 类型，并归一化到 [0, 1] 范围
    if float32:
        img = img.astype(np.float32) / 255.  # 转换类型并归一化到 [0, 1]

    return img  # 返回图像


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """将图像写入文件。

    Args:
        img (ndarray): 要写入的图像数据，通常是一个 NumPy 数组，表示图像。
        file_path (str): 图像保存的文件路径，包含文件名。
        params (None 或 list): 与 OpenCV 的 `imwrite` 接口相同的参数，用于设置图像保存时的附加选项，如压缩质量等。
        auto_mkdir (bool): 如果 `file_path` 的父文件夹不存在，是否自动创建它。默认为 `True`，表示会自动创建。

    Returns:
        bool: 返回是否成功写入图像。如果成功，返回 `True`，否则抛出错误。

    Description:
        该函数首先检查文件路径的父目录是否存在，如果不存在则自动创建该目录。
        然后，使用 OpenCV 的 `cv2.imwrite` 函数将图像写入指定的文件路径。
        如果图像保存失败，会抛出一个 `IOError` 错误。
    """

    # 如果 auto_mkdir 为 True，自动创建文件路径的父目录
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))  # 获取文件路径的父目录
        os.makedirs(dir_name, exist_ok=True)  # 如果目录不存在，则创建目录，exist_ok=True 表示如果目录已存在不报错

    # 使用 OpenCV 的 imwrite 函数将图像写入指定路径
    ok = cv2.imwrite(file_path, img, params)  # img 是图像数据，file_path 是保存路径，params 是可选参数

    # 如果写入失败，则抛出 IOError 异常
    if not ok:
        raise IOError('Failed in writing images.')

