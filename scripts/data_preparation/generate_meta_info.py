from os import path as osp  # 导入 os.path 并重命名为 osp，便于文件路径处理
from PIL import Image  # 导入 PIL 库用于图像处理
from utils import scandir  # 自定义工具函数，用于扫描文件夹中的文件

def generate_meta_info_aid():
    """
    为 AID 数据集生成元信息文件。

    元信息文件内容包括每个图像的文件名、分辨率（宽度和高度）以及通道数。
    """
    gt_folder = 'datasets/AID/AID_train_HR/'  # 高分辨率子图像文件夹路径
    meta_info_txt = 'datasets/meta_info/meta_info_AIDp600_GT.txt'  # 输出的元信息文件路径

    # 获取文件夹中的所有图像文件路径，按名称排序
    img_list = sorted(list(scandir(gt_folder)))

    # 打开元信息文件，准备写入
    with open(meta_info_txt, 'w') as f:
        # 遍历图像列表
        for idx, img_path in enumerate(img_list):
            # 使用 PIL 的 lazy load 功能仅加载图像头信息而不完全加载图像数据
            img = Image.open(osp.join(gt_folder, img_path))
            width, height = img.size  # 获取图像宽度和高度
            mode = img.mode  # 获取图像的模式（例如：RGB、L 等）

            # 根据图像模式判断通道数
            if mode == 'RGB':
                n_channel = 3  # RGB 模式有 3 个通道
            elif mode == 'L':
                n_channel = 1  # L 模式为单通道（灰度图）
            else:
                # 如果遇到不支持的模式，抛出异常
                raise ValueError(f'Unsupported mode {mode}.')

            # 生成当前图像的元信息字符串
            info = f'{img_path} ({height},{width},{n_channel})'
            # 打印元信息（带编号）到控制台
            print(idx + 1, info)
            # 将元信息写入文件
            f.write(f'{info}\n')

# 主函数入口
if __name__ == '__main__':
    generate_meta_info_aid()  # 调用生成元信息函数
