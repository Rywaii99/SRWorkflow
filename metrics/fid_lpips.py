import numpy as np
import torch
import lpips
from metrics.metric_util import reorder_image, to_y_channel
from utils.registry import METRIC_REGISTRY
from .fid import load_patched_inception_v3
from scipy import linalg


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, device='cuda' if torch.cuda.is_available() else 'cpu',
                    net='alex', **kwargs):
    """计算 LPIPS（学习感知图像块相似度）。

    Args:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        input_order (str): 输入图像的通道顺序，可以是 'HWC' 或 'CHW'。默认为 'HWC'。
        test_y_channel (bool): 是否在 YCbCr 的 Y 通道上计算。默认：False。
        device (str): 计算设备，如 'cuda' 或 'cpu'。
        net (str): LPIPS 使用的网络，如 'alex' 或 'vgg'。

    Returns:
        float: 计算得到的 LPIPS 值。
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

    # 将图像转换为 torch.Tensor 并归一化到 [-1, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1

    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net=net).to(device)

    # 计算 LPIPS 值
    with torch.no_grad():
        lpips_value = loss_fn(img, img2).item()

    return lpips_value


@METRIC_REGISTRY.register()
def calculate_fid(img, img2, crop_border, input_order='HWC', test_y_channel=False, device='cuda' if torch.cuda.is_available() else 'cpu',
                  resize_input=True, normalize_input=False, **kwargs):
    """计算 FID（Frechet Inception Distance）。

    Args:
        img (ndarray): 范围为 [0, 255] 的图像。
        img2 (ndarray): 范围为 [0, 255] 的图像。
        crop_border (int): 每个图像边缘裁剪的像素数，这些像素不参与计算。
        input_order (str): 输入图像的通道顺序，可以是 'HWC' 或 'CHW'。默认为 'HWC'。
        test_y_channel (bool): 是否在 YCbCr 的 Y 通道上计算。默认：False。
        device (str): 计算设备，如 'cuda' 或 'cpu'。

    Returns:
        float: 计算得到的 FID 值。
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

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # 加载自己实现的 InceptionV3 模型
    inception = load_patched_inception_v3(device=device, resize_input=resize_input, normalize_input=normalize_input)

    def get_activations(images, model):
        with torch.no_grad():
            pred = model(images)
            if isinstance(pred, tuple):
                pred = pred[0]
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.cpu().numpy().reshape(images.shape[0], -1)
        return pred

    act1 = get_activations(img, inception)
    act2 = get_activations(img2, inception)

    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2, rowvar=False)

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid_value