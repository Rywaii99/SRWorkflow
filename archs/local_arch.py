import torch
import torch.nn as nn
import torch.nn.functional as F


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        """
        初始化AvgPool2d类的构造函数。

        Args:
            kernel_size (int or tuple of ints, optional): 卷积核大小。如果为None，则通过base_size计算。
            base_size (int or tuple, optional): 基础大小，用于计算kernel_size。
            auto_pad (bool, optional): 是否自动进行填充，默认为True。
            fast_imp (bool, optional): 是否使用快速实现，默认为False。
            train_size (tuple, optional): 用于训练时的尺寸，通常是输入张量的大小。

        """
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # 仅用于快速实现
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]  # 用于选择不同的步长
        self.max_r1 = self.rs[0]  # 最大步长r1
        self.max_r2 = self.rs[0]  # 最大步长r2
        self.train_size = train_size  # 训练时的尺寸

    def extra_repr(self) -> str:
        """
        返回层的字符串表示，便于调试。

        Returns:
            str: 层的字符串表示，包含kernel_size、base_size、stride和fast_imp等信息。
        """
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        """
        前向传播函数，执行池化操作。

        Args:
            x (torch.Tensor): 输入张量，形状通常为(N, C, H, W)。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        # 如果没有指定kernel_size，但指定了base_size，则根据train_size和输入的尺寸来计算kernel_size
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)  # 如果base_size是整数，则转为元组
            self.kernel_size = list(self.base_size)
            # 根据输入张量的尺寸和训练尺寸来计算kernel_size
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # 仅用于快速实现，计算最大步长
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        # 如果kernel_size大于等于输入的空间尺寸，则执行自适应池化
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)  # 执行自适应平均池化，将输出池化到1x1

        # 如果选择了快速实现（fast_imp），执行快速池化算法
        if self.fast_imp:
            h, w = x.shape[2:]  # 获取输入的高度和宽度
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                # 如果kernel_size大于输入尺寸，执行自适应平均池化
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                # 否则，选择合适的步长（r1 和 r2），进行池化
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # 约束步长的最大值
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                # 对输入张量进行累加求和
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape  # 获取累加后的张量的形状
                # 计算kernel_size的实际大小（k1, k2）
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                # 基于累加和计算池化结果
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                # 通过插值恢复到原始大小
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            # 如果没有选择快速实现，执行普通的平均池化
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum_(dim=-2)  # 累加求和
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # 对累加和进行填充，方便后续操作
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])  # 计算实际的kernel_size
            # 计算累加和的四个区域，用于实现平均池化
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3  # 根据四个区域的差值计算池化结果
            out = out / (k1 * k2)  # 对结果进行归一化

        # 如果需要自动填充（auto_pad），根据输出大小对结果进行填充
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # 计算需要填充的大小
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')  # 进行填充，使用复制模式

        return out


def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    """
    递归遍历模型的子模块，替换其中的AdaptiveAvgPool2d层为自定义的AvgPool2d层。

    Args:
        model (nn.Module): 输入的模型。
        base_size (int or tuple): 基础池化大小。
        train_size (tuple): 训练时的尺寸。
        fast_imp (bool): 是否启用快速实现。

    """
    for n, m in model.named_children():  # 遍历模型的每个子模块
        if len(list(m.children())) > 0:
            # 如果当前模块是复合模块，则递归进入子模块
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):  # 如果当前模块是AdaptiveAvgPool2d
            # 将其替换为自定义的AvgPool2d模块
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1  # 确保输出尺寸为1
            setattr(model, n, pool)  # 替换模块


class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        """
        将模型中的池化层替换为自定义的AvgPool2d层，并进行前向传播测试。

        Args:
            train_size (tuple): 训练数据的尺寸。

        """
        replace_layers(self, *args, train_size=train_size, **kwargs)  # 替换池化层
        imgs = torch.rand(train_size)  # 生成随机输入数据
        with torch.no_grad():
            self.forward(imgs)  # 执行前向传播
