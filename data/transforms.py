import cv2
import random
import torch


def mod_crop(img, scale):
    """Mod crop images, used during testing.
    根据给定的缩放因子 scale 对图像进行裁剪，使得裁剪后的图像尺寸能被 scale 整除。
    这通常用于图像预处理，尤其是在测试过程中。

    Args:
        img (ndarray): 输入图像。
        scale (int): 缩放因子。

    Returns:
        ndarray: 裁剪后的图像。
    """
    img = img.copy()  # 创建图像的副本，避免修改原图像
    if img.ndim in (2, 3):  # 确保图像是 2D 或 3D（灰度图或彩色图）
        h, w = img.shape[0], img.shape[1]  # 获取图像的高度和宽度
        h_remainder, w_remainder = h % scale, w % scale  # 计算高度和宽度分别对缩放因子的余数
        img = img[:h - h_remainder, :w - w_remainder, ...]  # 去掉余数部分，使得图像大小能被 scale 整除
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')  # 如果图像的维度不正确，抛出错误
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.
    进行配对的随机裁剪，支持 Numpy 数组和 Tensor 输入。
    It crops lists of lq and gt images with corresponding locations.
    裁剪低质量（LQ）图像和高质量（GT）图像，确保裁剪位置一致。

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT 图像，
            所有图像的形状必须相同。如果输入是 ndarray，它会被转换为列表。
        img_lqs (list[ndarray] | ndarray): LQ 图像，
            所有图像的形状必须相同。如果输入是 ndarray，它会被转换为列表。
        gt_patch_size (int): GT 图像的补丁大小。
        scale (int): 缩放因子。
        gt_path (str): GT 图像的路径。默认为 None。

    Returns:
        list[ndarray] | ndarray: 返回裁剪后的 GT 图像和 LQ 图像。
            如果结果仅有一个元素，则直接返回 ndarray。
    """
    # 确保输入是列表格式
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # 判断输入是 Tensor 还是 Numpy 数组
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    # 获取图像的高度和宽度
    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]  # LQ 图像的高度和宽度
        h_gt, w_gt = img_gts[0].size()[-2:]  # GT 图像的高度和宽度
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]  # LQ 图像的高度和宽度
        h_gt, w_gt = img_gts[0].shape[0:2]  # GT 图像的高度和宽度

    lq_patch_size = gt_patch_size // scale  # 计算 LQ 图像的补丁大小

    # 检查图像的尺寸与缩放因子是否匹配
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ' +
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ' +
                         f'({lq_patch_size}, {lq_patch_size}). ' +
                         f'Please remove {gt_path}.')

    # 随机选择 LQ 图像补丁的顶部和左侧坐标
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # 裁剪 LQ 图像
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # 裁剪对应的 GT 图像
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    # 如果只有一个图像，直接返回该图像
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Rotation. Default: True.
        flows (list[ndarray]): Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5  # 随机决定是否进行水平翻转
    vflip = rotation and random.random() < 0.5  # 随机决定是否进行垂直翻转
    rot90 = rotation and random.random() < 0.5  # 随机决定是否进行旋转

    def _augment(img):
        if hflip:  # 水平翻转
            cv2.flip(img, 1, img)
        if vflip:  # 垂直翻转
            cv2.flip(img, 0, img)
        if rot90:  # 顺时针旋转 90°
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # 水平翻转并调整流向
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1  # 翻转水平方向的流量
        if vflip:  # 垂直翻转并调整流向
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1  # 翻转垂直方向的流量
        if rot90:  # 顺时针旋转 90°
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]  # 交换流动的维度
        return flow

    if not isinstance(imgs, list):  # 如果输入不是列表，将其转换为列表
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]  # 对每张图像进行增强
    if len(imgs) == 1:  # 如果只有一张图像，直接返回图像
        imgs = imgs[0]

    if flows is not None:  # 如果有光流信息，也进行增强

        def img_rotate(img, angle, center=None, scale=1.0):
            """Rotate image.

            Args:
                img (ndarray): Image to be rotated.
                angle (float): Rotation angle in degrees. Positive values mean
                    counter-clockwise rotation.
                center (tuple[int]): Rotation center. If the center is None,
                    initialize it as the center of the image. Default: None.
                scale (float): Isotropic scale factor. Default: 1.0.
            """
            (h, w) = img.shape[:2]  # 获取图像的高度和宽度

            if center is None:  # 如果没有指定旋转中心，使用图像中心
                center = (w // 2, h // 2)

            matrix = cv2.getRotationMatrix2D(center, angle, scale)  # 计算旋转矩阵
            rotated_img = cv2.warpAffine(img, matrix, (w, h))  # 应用旋转矩阵进行图像旋转
            return rotated_img


"""
================= 测试专用 =================
"""

def paired_random_crop_test(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop. Support Numpy array and Tensor inputs.
    进行配对的随机裁剪，支持 Numpy 数组和 Tensor 输入。
    It crops lists of lq and gt images with corresponding locations.
    裁剪低质量（LQ）图像和高质量（GT）图像，确保裁剪位置一致。

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT 图像，
            所有图像的形状必须相同。如果输入是 ndarray，它会被转换为列表。
        img_lqs (list[ndarray] | ndarray): LQ 图像，
            所有图像的形状必须相同。如果输入是 ndarray，它会被转换为列表。
        gt_patch_size (int): GT 图像的补丁大小。
        scale (int): 缩放因子。
        gt_path (str): GT 图像的路径。默认为 None。

    Returns:
        list[ndarray] | ndarray: 返回裁剪后的 GT 图像和 LQ 图像。
            如果结果仅有一个元素，则直接返回 ndarray。
        top: 随机裁剪的y坐标
        left: 随机裁剪的x坐标
    """
    # 确保输入是列表格式
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # 判断输入是 Tensor 还是 Numpy 数组
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    # 获取图像的高度和宽度
    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]  # LQ 图像的高度和宽度
        h_gt, w_gt = img_gts[0].size()[-2:]  # GT 图像的高度和宽度
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]  # LQ 图像的高度和宽度
        h_gt, w_gt = img_gts[0].shape[0:2]  # GT 图像的高度和宽度

    lq_patch_size = gt_patch_size // scale  # 计算 LQ 图像的补丁大小

    # 检查图像的尺寸与缩放因子是否匹配
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ' +
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ' +
                         f'({lq_patch_size}, {lq_patch_size}). ' +
                         f'Please remove {gt_path}.')

    # 随机选择 LQ 图像补丁的顶部和左侧坐标
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # 裁剪 LQ 图像
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    # 裁剪对应的 GT 图像
    top_gt, left_gt = int(top * scale), int(left * scale)
    if input_type == 'Tensor':
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    # 如果只有一个图像，直接返回该图像
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs, top, left


def paired_random_crop_pltshow(img_gt, img_lq, gt_patch_size, scale):
    """对配对图像进行随机裁剪并在原图上绘制红框标记裁剪区域，显示裁剪后的 GT 和 LQ 图像。

    参数：
        img_gts (ndarray): 高质量 GT 图像。
        img_lqs (ndarray): 低质量 LQ 图像。
        gt_patch_size (int): GT 图像补丁的大小。
        scale (int): 缩放因子。

    返回：
        None
    """
    import matplotlib.pyplot as plt
    # 调用裁剪函数，获取裁剪后的图像以及裁剪区域坐标
    img_gts, img_lqs, top, left = paired_random_crop_test(img_gt, img_lq, gt_patch_size, scale)

    # 打印裁剪后的图像形状
    print("裁剪后的 GT 图像形状：", img_gts.shape)
    print("裁剪后的 LQ 图像形状：", img_lqs.shape)

    # 绘制图像
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # 显示原始 GT 图像
    axes[0].imshow(cv2.cvtColor(img_gts, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original GT Image")
    axes[0].axis('off')

    # 显示原始 LQ 图像
    axes[1].imshow(cv2.cvtColor(img_lqs, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Original LQ Image")
    axes[1].axis('off')

    # 在原始 GT 图像上绘制裁剪区域的红框
    axes[2].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axes[2].set_title("GT with Crop Area")
    axes[2].axis('off')
    rect_gt = plt.Rectangle((left * scale, top * scale), gt_patch_size * scale, gt_patch_size * scale,
                            linewidth=2, edgecolor='red', facecolor='none')
    axes[2].add_patch(rect_gt)

    # 在原始 LQ 图像上绘制裁剪区域的红框
    axes[3].imshow(cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB))
    axes[3].set_title("LQ with Crop Area")
    axes[3].axis('off')
    rect_lq = plt.Rectangle((left, top), gt_patch_size, gt_patch_size,
                            linewidth=2, edgecolor='red', facecolor='none')
    axes[3].add_patch(rect_lq)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 创建简单的 LQ 和 GT 图像
    img_lq_path = 'viaduct_98_lq.png'  # 低质量图像路径
    img_gt_path = 'viaduct_98.png'  # 高质量图像路径

    # 读取 LQ 和 GT 图像
    img_lq = cv2.imread(img_lq_path)
    img_gt = cv2.imread(img_gt_path)

    # 设置参数
    gt_patch_size = 50  # 高质量图像裁剪尺寸
    scale = 2  # 缩放因子 (600 / 300 = 2)

    # 调用新函数进行裁剪并显示结果
    paired_random_crop_pltshow(img_gt, img_lq, gt_patch_size, scale)