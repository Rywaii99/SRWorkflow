import numpy as np

from utils import bgr2ycbcr


def reorder_image(img, input_order='HWC'):
    """
    重新排列图像的通道顺序，使其符合 'HWC' 格式。

    根据输入的通道顺序，将图像调整为适当的格式。支持两种输入格式：'HWC' 和 'CHW'。
    如果输入图像是二维的（灰度图），则自动将其扩展为三维（添加一个通道维度）。

    Args:
        img (ndarray): 输入图像，可能的形状有 (h, w) 或 (c, h, w) 或 (h, w, c)。
        input_order (str): 输入图像的通道顺序。可以是 'HWC' 或 'CHW'。
            如果输入图像形状是 (h, w)，则 input_order 不会影响图像的顺序。默认值是 'HWC'。

    Returns:
        ndarray: 重新排列后的图像。

    :exception:
        如果 `input_order` 不是 'HWC' 或 'CHW'，则会抛出 ValueError 异常。
    """

    # 检查输入顺序是否有效
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")

    # 如果图像是二维的（灰度图），则扩展为三维（添加一个通道维度）
    if len(img.shape) == 2:
        img = img[..., None]  # 在最后添加一个维度，变为 (h, w, 1)

    # 如果输入顺序是 'CHW'，则将其转置为 'HWC' 格式
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)  # 将通道维放到最后，转换为 (h, w, c)

    return img


def to_y_channel(img):
    """
    将图像转换为 YCbCr 色彩空间中的 Y 通道。

    此函数将输入的 RGB 图像（假设是 BGR 格式）转换为 YCbCr 色彩空间中的 Y 通道，
    并将 Y 通道的值归一化到 [0, 255] 范围。

    Args:
        img (ndarray): 输入图像，范围在 [0, 255] 之间。

    Returns:
        ndarray: 仅包含 Y 通道的图像，值范围为 [0, 255]（浮动类型，不四舍五入）。
    """

    img = img.astype(np.float32) / 255.  # 将图像的像素值归一化到 [0, 1] 范围
    if img.ndim == 3 and img.shape[2] == 3:  # 如果是三通道图像
        # 使用 bgr2ycbcr 函数将图像转换为 YCbCr 色彩空间中的 Y 通道
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]  # 仅保留 Y 通道，添加一个通道维度，变为 (h, w, 1)

    return img * 255.  # 将 Y 通道的值还原到 [0, 255] 范围
