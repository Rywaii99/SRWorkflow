from torch.utils import data as data
from torchvision.transforms.functional import normalize

from data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from data.transforms import augment, paired_random_crop
from utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """
    图像对数据集，用于图像恢复任务。

    读取 LQ（低质量图像，例如低分辨率、模糊、噪声等）和 GT（高质量图像）图像对。

    该数据集有三种模式：
    1. **lmdb**：使用 lmdb 文件。如果 `opt['io_backend']` 设置为 'lmdb'。
    2. **meta_info_file**：使用元信息文件生成路径。\
        如果 `opt['io_backend']` 不是 'lmdb'，且 `opt['meta_info_file']` 不为 None。
    3. **folder**：扫描文件夹生成路径。默认情况下使用此模式。

    Args:
        opt (dict): 配置文件，用于训练数据集，包含以下键：
        - dataroot_gt (str): GT（高质量图像）根目录路径。
        - dataroot_lq (str): LQ（低质量图像）根目录路径。
        - meta_info_file (str): 元信息文件路径。
        - io_backend (dict): 输入/输出后端类型及其他配置。
        - filename_tmpl (str): 每个文件名的模板。注意模板不包括文件扩展名，默认为 '{}'。
        - gt_size (int): GT 图像裁剪后的大小。
        - use_hflip (bool): 是否使用水平翻转。
        - use_rot (bool): 是否使用旋转（包括垂直翻转和宽高转换）。
        - scale (bool): 缩放，通常会自动添加。
        - phase (str): 数据集的使用阶段，'train' 或 'val'。
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt  # 配置选项
        # 文件客户端（IO 后端）
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # 输入输出后端配置
        self.mean = opt['mean'] if 'mean' in opt else None  # 均值，用于归一化
        self.std = opt['std'] if 'std' in opt else None  # 方差，用于归一化

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']  # GT 和 LQ 图像的文件夹路径
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']  # 文件名模板
        else:
            self.filename_tmpl = '{}'  # 默认模板

        # 根据输入/输出后端配置加载路径
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]  # 使用 LMDB 时，配置数据库路径
            self.io_backend_opt['client_keys'] = ['lq', 'gt']  # 客户端键
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])  # 从 LMDB 获取路径
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            # 使用元信息文件获取路径
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # 默认情况下，从文件夹获取路径
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        """
        获取指定索引的图像对，执行必要的转换和数据增强。
        """

        # 如果文件客户端未初始化，创建一个新的文件客户端
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']  # 缩放因子

        # 加载 GT 和 LQ 图像。维度顺序：HWC，通道顺序：BGR；图像范围：[0, 1]，数据类型：float32。
        gt_path = self.paths[index]['gt_path']  # 获取 GT 图像路径
        img_bytes = self.file_client.get(gt_path, 'gt')  # 从文件客户端获取图像字节数据
        img_gt = imfrombytes(img_bytes, float32=True)  # 将字节数据转换为图像

        lq_path = self.paths[index]['lq_path']  # 获取 LQ 图像路径
        img_bytes = self.file_client.get(lq_path, 'lq')  # 获取 LQ 图像字节数据
        img_lq = imfrombytes(img_bytes, float32=True)  # 将字节数据转换为图像

        # 训练阶段的图像增强
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']  # GT 图像裁剪后的大小
            # 随机裁剪
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # 随机翻转和旋转
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # 颜色空间转换
        if 'color' in self.opt and self.opt['color'] == 'y':
            # 转换为 YCbCr 色彩空间，并只保留 Y 通道
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # 在验证或测试阶段，裁剪与 LQ 图像尺寸不匹配的 GT 图像（特别是对于超分辨率基准数据集）
        # TODO: 更新数据集，避免强制裁剪
        if self.opt['phase'] != 'train':
            h, w, _ = img_lq.shape
            # 根据缩放比例调整 GT 图像的尺寸，使其与 LQ 图像一致
            img_gt = img_gt[0:h * scale, 0:w * scale, :]  # 使用 LQ 图像的尺寸对 GT 图像进行裁剪

        # BGR 转 RGB，HWC 转 CHW，numpy 转 tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)  # 对 LQ 图像进行归一化
            normalize(img_gt, self.mean, self.std, inplace=True)  # 对 GT 图像进行归一化

        # 返回一个字典，包含 LQ 图像、GT 图像及其路径
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        """
        返回数据集的大小，即图像对的数量。
        """
        return len(self.paths)