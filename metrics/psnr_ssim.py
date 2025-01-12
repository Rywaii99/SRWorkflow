import cv2
import numpy as np
import torch
import torch.nn.functional as F

from metrics.metric_util import reorder_image, to_y_channel
from utils.color_util import rgb2ycbcr_pt
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 PSNR（峰值信噪比）。

    参考文献：https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        input_order (str): 输入图像的通道顺序，可以是 'HWC' 或 'CHW'。默认为 'HWC'。
        test_y_channel (bool): 是否在 YCbCr 的 Y 通道上计算。默认：False。

    Returns:
        float: 计算得到的 PSNR 值。
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 10. * np.log10(255. * 255. / mse)


@METRIC_REGISTRY.register()
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """计算 PSNR（峰值信噪比）（PyTorch 版本）。

    参考文献：https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。
        img2 (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        test_y_channel (bool): 是否在 Y 通道上计算。默认：False。

    Returns:
        float: 计算得到的 PSNR 值。
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """计算 SSIM（结构相似性）。

    参考文献：“图像质量评估：从误差可视性到结构相似性”

    结果与官方发布的 MATLAB 代码一致：https://ece.uwaterloo.ca/~z70wang/research/ssim/。

    对于三通道图像，SSIM 会分别计算每个通道的 SSIM，然后求平均。

    Args:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        input_order (str): 输入图像的通道顺序，可以是 'HWC' 或 'CHW'，默认为 'HWC'。
        test_y_channel (bool): 是否在 YCbCr 的 Y 通道上计算。默认：False。

    Returns:
        float: 计算得到的 SSIM 值。
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()


@METRIC_REGISTRY.register()
def calculate_ssim_pt(img, img2, crop_border, test_y_channel=False, **kwargs):
    """计算 SSIM（结构相似性）（PyTorch 版本）。

    参考文献：“图像质量评估：从误差可视性到结构相似性”

    结果与官方发布的 MATLAB 代码一致：https://ece.uwaterloo.ca/~z70wang/research/ssim/。

    对于三通道图像，SSIM 会分别计算每个通道的 SSIM，然后求平均。

    Args:
        img (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。
        img2 (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        test_y_channel (bool): 是否在 Y 通道上计算。默认：False。

    Returns:
        float: 计算得到的 SSIM 值。
    """
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    ssim = _ssim_pth(img * 255., img2 * 255.)
    return ssim


def _ssim(img, img2):
    """计算单通道图像的 SSIM（结构相似性）。

    被函数 `calculate_ssim` 调用。

    Args:
        img (ndarray): 范围为 [0, 255] 的图像，通道顺序为 'HWC'。
        img2 (ndarray): 范围为 [0, 255] 的图像，通道顺序为 'HWC'。

    Returns:
        float: 计算得到的 SSIM 值。
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]  # 使用 11x11 的窗口进行卷积，结果为 valid 模式
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def _ssim_pth(img, img2):
    """计算 SSIM（结构相似性）（PyTorch 版本）。

    被函数 `calculate_ssim_pt` 调用。

    Args:
        img (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。
        img2 (Tensor): 范围为 [0, 1] 的图像，形状为 (n, 3/1, h, w)。

    Returns:
        float: 计算得到的 SSIM 值。
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid 模式
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid 模式
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])
