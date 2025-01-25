import logging
import torch
from os import path as osp

from data import build_dataloader, build_dataset
from models import build_model
from utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # 解析选项，设置分布式设置，设置随机种子
    opt, _ = parse_options(root_path, is_train=False)  # 解析配置文件，获取测试选项

    # 设置PyTorch的CuDNN后端以优化性能
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True  # 如果需要确定性结果，可以取消注释此行

    # 创建实验目录并初始化日志记录器
    make_exp_dirs(opt)  # 根据选项创建实验目录
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")  # 设置日志文件路径
    logger = get_root_logger(logger_name='srworkflow', log_level=logging.INFO, log_file=log_file)  # 获取根日志记录器
    logger.info(get_env_info())  # 记录环境信息
    logger.info(dict2str(opt))  # 记录配置选项

    # 创建测试数据集和数据加载器
    test_loaders = []  # 初始化测试数据加载器列表
    for _, dataset_opt in sorted(opt['datasets'].items()):  # 遍历所有数据集选项
        test_set = build_dataset(dataset_opt)  # 构建测试数据集
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])  # 构建测试数据加载器
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")  # 记录测试数据集中的图像数量
        test_loaders.append(test_loader)  # 将数据加载器添加到列表中

    # 创建模型
    model = build_model(opt)  # 根据选项构建模型

    # 对每个测试数据加载器进行测试
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']  # 获取测试数据集的名称
        logger.info(f'Testing {test_set_name}...')  # 记录正在测试的数据集名称
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])  # 使用模型进行验证


if __name__ == '__main__':
    # 获取当前文件的根路径
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # 调用测试流程函数
    test_pipeline(root_path)