import numpy as np
import torch


def rgb2ycbcr(img, y_only=False):
    """将RGB图像转换为YCbCr图像。

    该函数使用ITU-R BT.601标准定义的转换公式，适用于标准清晰度电视。
    参见：https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion。

    它与OpenCV中的`cv2.cvtColor`函数有所不同，后者使用的是JPEG转换公式。
    参见：https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion。

    Args:
        img (ndarray): 输入的图像。支持以下类型：
            1. np.uint8类型，取值范围为[0, 255]；
            2. np.float32类型，取值范围为[0, 1]。
        y_only (bool): 是否只返回Y通道。默认为False，表示返回完整的YCbCr图像。

    Returns:
        ndarray: 转换后的YCbCr图像，输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype  # 获取输入图像的数据类型
    img = _convert_input_type_range(img)  # 转换输入图像的类型和范围（统一为[0, 1]的浮点类型）

    # 如果只需要Y通道
    if y_only:
        out_img = np.dot(img, [65.481, 128.553, 24.966]) + 16.0  # 计算Y通道
    else:
        # 计算完整的YCbCr图像
        out_img = np.matmul(
            img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]

    out_img = _convert_output_type_range(out_img, img_type)  # 将输出图像转换回原始类型和范围
    return out_img


def bgr2ycbcr(img, y_only=False):
    """将BGR图像转换为YCbCr图像。

    该函数是`rgb2ycbcr`函数的BGR版本，使用的是ITU-R BT.601标准定义的转换公式。

    Args:
        img (ndarray): 输入的图像，支持np.uint8类型和np.float32类型。
        y_only (bool): 是否只返回Y通道。默认为False。

    Returns:
        ndarray: 转换后的YCbCr图像，输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype  # 获取输入图像的数据类型
    img = _convert_input_type_range(img)  # 转换输入图像的类型和范围（统一为[0, 1]的浮点类型）

    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0  # 计算Y通道
    else:
        # 计算完整的YCbCr图像
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]

    out_img = _convert_output_type_range(out_img, img_type)  # 将输出图像转换回原始类型和范围
    return out_img


def ycbcr2rgb(img):
    """将YCbCr图像转换为RGB图像。

    该函数使用ITU-R BT.601标准定义的转换公式，适用于标准清晰度电视。

    Args:
        img (ndarray): 输入的YCbCr图像，支持np.uint8类型和np.float32类型。

    Returns:
        ndarray: 转换后的RGB图像，输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype  # 获取输入图像的数据类型
    img = _convert_input_type_range(img) * 255  # 转换并放大到[0, 255]的范围
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]  # 计算RGB图像
    out_img = _convert_output_type_range(out_img, img_type)  # 将输出图像转换回原始类型和范围
    return out_img


def ycbcr2bgr(img):
    """将YCbCr图像转换为BGR图像。

    该函数是`ycbcr2rgb`函数的BGR版本，使用ITU-R BT.601标准定义的转换公式。

    Args:
        img (ndarray): 输入的YCbCr图像，支持np.uint8类型和np.float32类型。

    Returns:
        ndarray: 转换后的BGR图像，输出图像与输入图像具有相同的类型和取值范围。
    """
    img_type = img.dtype  # 获取输入图像的数据类型
    img = _convert_input_type_range(img) * 255  # 转换并放大到[0, 255]的范围
    out_img = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]  # 计算BGR图像
    out_img = _convert_output_type_range(out_img, img_type)  # 将输出图像转换回原始类型和范围
    return out_img


def _convert_input_type_range(img):
    """转换输入图像的类型和范围。

    将输入图像转换为np.float32类型，并将其范围转换为[0, 1]。

    Args:
        img (ndarray): 输入图像，支持np.uint8类型和np.float32类型。

    Returns:
        ndarray: 转换后的图像，类型为np.float32，范围为[0, 1]。
    """
    img_type = img.dtype  # 获取图像的类型
    img = img.astype(np.float32)  # 转换为np.float32类型
    if img_type == np.float32:
        pass  # 如果原图像已经是np.float32类型，不做任何操作
    elif img_type == np.uint8:
        img /= 255.  # 如果是np.uint8类型，转换为[0, 1]范围的浮点数
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')  # 异常处理
    return img


def _convert_output_type_range(img, dst_type):
    """根据目标类型转换图像的类型和范围。

    将图像转换为目标类型和目标范围。如果目标类型是np.uint8，则图像范围为[0, 255]；
    如果目标类型是np.float32，则图像范围为[0, 1]。

    Args:
        img (ndarray): 输入图像，类型为np.float32，范围为[0, 255]。
        dst_type (np.uint8 | np.float32): 目标类型。如果是np.uint8，则转换到[0, 255]的范围；
                                          如果是np.float32，则转换到[0, 1]的范围。

    Returns:
        ndarray: 转换后的图像，类型为目标类型，并且具有目标范围。
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()  # 如果目标类型是np.uint8，进行四舍五入处理
    else:
        img /= 255.  # 如果目标类型是np.float32，转换到[0, 1]的范围
    return img.astype(dst_type)  # 返回转换后的图像


def rgb2ycbcr_pt(img, y_only=False):
    """将RGB图像转换为YCbCr图像（PyTorch版本）。

    该函数实现了ITU-R BT.601标准定义的RGB到YCbCr的转换。

    Args:
        img (Tensor): 输入图像，形状为(n, 3, h, w)，范围为[0, 1]，浮点型，RGB格式。
        y_only (bool): 是否只返回Y通道。默认为False。

    Returns:
        Tensor: 转换后的图像，形状为(n, 3/1, h, w)，范围为[0, 1]，浮点型。
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)  # Y通道的权重
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0  # 计算Y通道
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)  # YCbCr的偏置值
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.  # 将图像范围归一化到[0, 1]
    return out_img
