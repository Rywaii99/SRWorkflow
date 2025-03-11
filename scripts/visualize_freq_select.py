import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype, resize
import matplotlib.colors as mcolors
from matplotlib import cm


# 配置参数
class Config:
    k_list = [3, 5, 7]  # 频率分割参数(值越大保留频率越低)
    spatial_group = 3  # 分组卷积数
    img_path = "P0774_crop_1.png"  # 替换为你的图片路径
    img_size = 396  # 调整图像尺寸


cfg = Config()

# --------------------------------------------------
# 1. 图像加载与预处理
# --------------------------------------------------
# 读取图像 -> [加载你自己的图片]
img = read_image(cfg.img_path)  # [C,H,W]范围0-255
img = convert_image_dtype(img, torch.float32)  # 转换到0-1范围
img = resize(img, [cfg.img_size] * 2)  # 调整尺寸
img = img.unsqueeze(0)  # [1, C, H, W]
print("Input image shape:", img.shape)

# 显示原始图像
plt.figure(figsize=(12, 6))
plt.subplot(231), plt.imshow(img.squeeze(0).permute(1, 2, 0)), plt.title("Original Image")


# --------------------------------------------------
# 2. 定义简化的FrequencySelection模块
# --------------------------------------------------
class FreqDemo:
    def __init__(self, k_list, spatial_group):
        self.k_list = k_list
        self.spatial_group = spatial_group

        # 初始化注意力卷积层（简化版本）
        self.freq_weight_conv_list = torch.nn.ModuleList([
            torch.nn.Conv2d(3, spatial_group, 3, padding=1) for _ in k_list
        ])

    def sp_act(self, x):
        return torch.sigmoid(x) * 2  # 使用Sigmoid激活

    def forward(self, x):
        x_list = []
        pre_x = x.clone()
        b, c, h, w = x.shape

        # 转换为频域
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))

        # 可视化频域幅度谱
        # magnitude = torch.log(torch.abs(x_fft) + 1e-9)
        magnitude = torch.log(torch.abs(x_fft).mean(dim=1, keepdim=True) + 1e-9)
        magnitude = torch.clamp(magnitude, 0.0, 1.0)
        # plt.subplot(232), plt.imshow(magnitude.squeeze(0).permute(1, 2, 0)), plt.title("Frequency Spectrum")
        plt.subplot(232), plt.imshow(magnitude.squeeze().cpu().numpy()), plt.title("Frequency Spectrum")

        # 创建自定义颜色映射
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#6B0220", "#063163"], N=256)

        for idx, freq in enumerate(self.k_list):
            # --------------------------------------------------
            # 关键步骤1: 创建频域掩膜
            # --------------------------------------------------
            mask = torch.zeros_like(x[:, 0:1, :, :])  # 初始化mask，形状为 [b, 1, h, w]
            h_start = round(h / 2 - h / (2 * freq))
            h_end = round(h / 2 + h / (2 * freq))
            w_start = round(w / 2 - w / (2 * freq))
            w_end = round(w / 2 + w / (2 * freq))
            mask[:, :, h_start:h_end, w_start:w_end] = 1.0

            # 可视化掩膜
            # if idx == 0:  # 仅展示第一个掩膜
            #     tmp_mask = torch.clamp(mask, 0.0, 1.0)
            #     plt.subplot(233), plt.imshow(tmp_mask[0, 0], cmap='gray'), plt.title(f"Mask (freq={freq})")

            # --------------------------------------------------
            # 关键步骤2: 提取低频成分
            # --------------------------------------------------
            low_fft = x_fft * mask
            low_part = torch.fft.ifft2(torch.fft.ifftshift(low_fft), norm='ortho').real

            # 计算高频成分
            high_part = pre_x - low_part
            pre_x = low_part

            # --------------------------------------------------
            # 关键步骤3: 高频加权（模拟注意力机制）
            # --------------------------------------------------
            freq_weight = self.freq_weight_conv_list[idx](high_part)
            freq_weight = self.sp_act(freq_weight)
            weighted_high = freq_weight * high_part

            x_list.append(weighted_high)

            # 可视化处理结果
            plt_idx = 234 + idx
            plt.subplot(plt_idx)
            tmp_weighted_high = torch.clamp(weighted_high, 0.0, 1.0)
            single_channel = tmp_weighted_high.mean(dim=1).squeeze(0).cpu().detach().numpy()

            # 应用自定义颜色映射
            plt.imshow(single_channel)
            plt.title(f"Freq {freq} High Part")

        # 最终低频成分
        x_list.append(pre_x)
        plt.subplot(233), plt.imshow(pre_x.mean(0).squeeze(0).permute(1, 2, 0))
        plt.title("Final Low Part")

        return sum(x_list)  # 合并所有成分


# --------------------------------------------------
# 3. 运行处理流程
# --------------------------------------------------
processor = FreqDemo(cfg.k_list, cfg.spatial_group)
output = processor.forward(img)

# 显示融合结果
plt.tight_layout()
plt.show()
