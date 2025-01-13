import datetime
import logging
import math
import time
import torch
from os import path as osp

from data import build_dataset, build_dataloader
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import build_model
from utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    # 如果配置中启用了 W&B 日志，并且没有处于 "debug" 模式，初始化 W&B 日志
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        # 如果启用了 W&B 日志，则要求启用 TensorBoard 日志，否则报错
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)  # 初始化 W&B 日志记录器

    tb_logger = None
    # 如果启用了 TensorBoard 日志，并且没有处于 "debug" 模式，则初始化 TensorBoard 日志记录器
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # 将 TensorBoard 日志目录设置为 'tb_logger' 文件夹下，文件夹名称为实验名称
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger  # 返回 TensorBoard 日志记录器对象


def create_train_val_dataloader(opt, logger):
    """Create train and validation dataloaders.

    创建训练和验证数据加载器

    Args:
        opt (dict): 配置字典，包含训练参数、数据集配置等信息
        logger (logger): 用于记录日志的日志记录器

    Returns:
        train_loader: 训练数据加载器
        train_sampler: 训练数据的采样器
        val_loaders: 验证数据加载器列表
        total_epochs: 总训练轮次
        total_iters: 总迭代次数
    """
    train_loader, val_loaders = None, []

    # 遍历数据集配置，创建对应的数据加载器
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':  # 处理训练集数据
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)  # 数据集扩展比例，默认为1
            train_set = build_dataset(dataset_opt)  # 构建训练集
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)  # 扩展采样器
            # 构建训练数据加载器，考虑分布式训练的设置
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            # 计算每个 epoch 所需的迭代次数和总迭代次数
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])  # 获取总迭代次数
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))  # 根据总迭代次数计算总轮次

            # 打印训练集的统计信息
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        # 处理验证集数据
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)  # 构建验证集
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)  # 将验证集加载器加入列表

        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')  # 如果数据集阶段不合法，抛出错误

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters  # 返回训练集、训练采样器、验证集、总轮次和总迭代次数


def load_resume_state(opt):
    """Load resume state to resume training from checkpoint.

    加载训练的恢复状态，用于恢复训练。

    Args:
        opt (dict): 配置字典，包含恢复训练的参数

    Returns:
        resume_state: 恢复状态（字典）
    """
    resume_state_path = None
    if opt['auto_resume']:  # 如果启用了自动恢复功能
        # 获取保存训练状态的路径
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):  # 如果状态文件夹存在
            # 查找目录下的所有 '.state' 文件，获取其路径
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                # 获取所有状态文件的迭代号，并选择最大迭代号的文件进行恢复
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path  # 将恢复路径设置到配置中
    else:
        # 如果没有启用自动恢复，则直接从配置文件中获取恢复路径
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        # 如果有恢复状态路径，则加载恢复状态
        device_id = torch.cuda.current_device()  # 获取当前 GPU 设备
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))  # 加载恢复状态
        check_resume(opt, resume_state['iter'])  # 检查恢复状态
    return resume_state  # 返回恢复状态


def train_pipeline(root_path):
    """Train pipeline for model training.

    训练管道，负责从配置解析到训练过程的执行。

    Args:
        root_path (str): 项目根目录路径
    """
    # 解析配置文件并设置分布式环境
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path  # 将根路径添加到配置字典中

    torch.backends.cudnn.benchmark = True  # 启用 cudnn 的自动调优，以提高训练速度
    # torch.backends.cudnn.deterministic = True  # 是否启用 cudnn 的确定性模式（非必需）

    # 加载恢复状态（如果需要）
    resume_state = load_resume_state(opt)
    # 如果没有恢复状态，则创建实验目录
    if resume_state is None:
        make_exp_dirs(opt)
        # 如果启用了 TensorBoard 并且不是调试模式，创建 TensorBoard 日志目录
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # 将配置文件复制到实验根目录
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # 注意：不要在上述代码中使用 get_root_logger 否则日志初始化将不正确
    # 初始化日志文件，记录训练过程
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='srworkflow', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())  # 记录环境信息
    logger.info(dict2str(opt))  # 记录配置参数

    # 初始化 W&B 和 TensorBoard 日志记录器
    tb_logger = init_tb_loggers(opt)

    # 创建训练和验证数据加载器
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # 创建模型
    model = build_model(opt)
    if resume_state:  # 如果有恢复状态，恢复训练
        model.resume_training(resume_state)  # 恢复模型的训练状态（包括优化器等）
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']  # 恢复的起始 epoch
        current_iter = resume_state['iter']  # 恢复的起始迭代次数
    else:
        start_epoch = 0  # 没有恢复状态，从第 0 epoch 开始
        current_iter = 0  # 从第 0 迭代开始

    # 创建消息日志记录器
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # 配置数据加载器的预取模式（如果有）
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)  # 使用 CPU 预取器
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)  # 使用 CUDA 预取器
        logger.info(f'Use {prefetch_mode} prefetch dataloader')  # 输出使用的预取模式
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # 开始训练
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()  # 计时器，用于统计数据加载时间和迭代时间
    start_time = time.time()

    # 训练每个 epoch
    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)  # 设置当前 epoch
        prefetcher.reset()  # 重置预取器
        train_data = prefetcher.next()  # 获取下一批训练数据

        while train_data is not None:  # 如果还有数据
            data_timer.record()  # 记录数据加载时间

            current_iter += 1  # 增加当前迭代次数
            if current_iter > total_iters:
                break  # 如果迭代次数超过总迭代次数，则结束训练

            # 更新学习率
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # 进行训练
            model.feed_data(train_data)  # 输入数据到模型
            model.optimize_parameters(current_iter)  # 优化模型参数
            iter_timer.record()  # 记录迭代时间

            if current_iter == 1:
                # 如果是第一次迭代，重置消息日志记录器的开始时间
                msg_logger.reset_start_time()

            # 每 `print_freq` 次迭代输出一次日志
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})  # 获取当前学习率
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})  # 记录时间
                log_vars.update(model.get_current_log())  # 获取当前模型日志
                msg_logger(log_vars)  # 输出日志

            # 每 `save_checkpoint_freq` 次保存模型和训练状态
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)  # 保存模型和训练状态

            # 每 `val_freq` 次进行验证
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])  # 验证模型

            data_timer.start()  # 开始记录数据加载时间
            iter_timer.start()  # 开始记录迭代时间
            train_data = prefetcher.next()  # 获取下一批数据
        # end of iter

    # end of epoch
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))  # 计算总训练时间
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')  # 保存最终模型
    model.save(epoch=-1, current_iter=-1)  # 保存最新模型

    # 如果配置了验证，进行最终验证
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    # 关闭 TensorBoard 日志记录器
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)