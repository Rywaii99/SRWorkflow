import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from models import lr_scheduler as lr_scheduler
from utils import get_root_logger
from utils.dist_util import master_only


class BaseModel():
    """基础模型类，定义了训练、验证、保存和加载模型等操作。

    Attributes:
        opt (dict): 配置字典，包含训练和模型设置。
        device (torch.device): 当前设备，决定使用 GPU 还是 CPU。
        is_train (bool): 是否在训练模式。
        schedulers (list): 学习率调度器的列表。
        optimizers (list): 优化器的列表。
    """

    def __init__(self, opt):
        """
        初始化模型参数。

        Args:
            opt (dict): 配置字典，包含训练配置、优化器配置等。
        """
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')  # 根据 GPU 配置选择设备
        self.is_train = opt['is_train']  # 是否在训练模式
        self.schedulers = []  # 存储学习率调度器
        self.optimizers = []  # 存储优化器

    def feed_data(self, data):
        """传入训练数据给模型（在具体子类中实现）。

        Args:
            data (dict): 输入数据，通常包含图像和标签。
        """
        pass

    def optimize_parameters(self):
        """执行一次优化步骤（在具体子类中实现）。
        """
        pass

    def get_current_visuals(self):
        """返回当前视觉效果的图像（在具体子类中实现）。

        Returns:
            dict: 包含图像的字典。
        """
        pass

    def save(self, epoch, current_iter):
        """保存模型和训练状态。

        Args:
            epoch (int): 当前的 epoch 数。
            current_iter (int): 当前的迭代数。
        """
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """验证模型的性能。

        Args:
            dataloader (torch.utils.data.DataLoader): 验证数据加载器。
            current_iter (int): 当前的迭代数。
            tb_logger (tensorboard logger): 用于记录日志的 TensorBoard。
            save_img (bool): 是否保存预测图像。默认：False。
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)  # 分布式验证
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)  # 非分布式验证

    def _initialize_best_metric_results(self, dataset_name):
        """初始化最佳性能指标记录。

        Args:
            dataset_name (str): 数据集的名称。
        """
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # 为数据集添加记录
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        """更新最佳指标结果。

        Args:
            dataset_name (str): 数据集名称。
            metric (str): 指标名称。
            val (float): 当前指标值。
            current_iter (int): 当前迭代次数。
        """
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def model_ema(self, decay=0.999):
        """执行模型的指数移动平均（EMA）。

        Args:
            decay (float): 衰减因子，决定移动平均的更新速度。
        """
        net_g = self.get_bare_model(self.net_g)  # 获取裸模型（去除 DataParallel/DistributedDataParallel）

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        """获取当前的日志字典。

        Returns:
            dict: 当前的日志信息。
        """
        return self.log_dict

    def model_to_device(self, net):
        """将模型转移到指定设备，并处理分布式或数据并行模式。

        Args:
            net (nn.Module): 需要转移到设备的模型。

        Returns:
            nn.Module: 转移到指定设备后的模型。
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        """根据优化器类型返回对应的优化器。

        Args:
            optim_type (str): 优化器类型，如 'Adam'、'SGD' 等。
            params (iterable): 优化器需要的参数。
            lr (float): 学习率。

        Returns:
            torch.optim.Optimizer: 对应的优化器实例。
        """
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        elif optim_type == 'Adamax':
            optimizer = torch.optim.Adamax(params, lr, **kwargs)
        elif optim_type == 'SGD':
            optimizer = torch.optim.SGD(params, lr, **kwargs)
        elif optim_type == 'ASGD':
            optimizer = torch.optim.ASGD(params, lr, **kwargs)
        elif optim_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(params, lr, **kwargs)
        elif optim_type == 'Rprop':
            optimizer = torch.optim.Rprop(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supported yet.')
        return optimizer

    def setup_schedulers(self):
        """设置学习率调度器。

        依据配置文件中的学习率调度器类型进行初始化。
        """
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **train_opt['scheduler']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """获取裸模型，去除 DataParallel 或 DistributedDataParallel 的包装。

        Args:
            net (nn.Module): 包装后的网络。

        Returns:
            nn.Module: 裸模型（去掉并行包装的部分）。
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """打印网络结构和参数数量。

        Args:
            net (nn.Module): 需要打印的网络模型。
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """设置学习率，通常用于 warm-up 阶段。

        Args:
            lr_groups_l (list): 每个优化器的学习率列表。
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """获取初始学习率，通常用于 warm-up 设置。

        Returns:
            list: 每个优化器的初始学习率列表。
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """更新学习率，支持学习率调度和 warm-up。

        Args:
            current_iter (int): 当前迭代次数。
            warmup_iter (int): warm-up 的迭代次数，默认为 -1 表示不使用 warm-up。
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

        # 如果当前迭代数小于 warmup_iter，则进行 warm-up
        if current_iter < warmup_iter:
            init_lr_g_l = self._get_init_lr()  # 获取初始学习率
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        """获取当前的学习率。

        Returns:
            list: 当前每个优化器的学习率。
        """
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """保存网络参数。

        Args:
            net (nn.Module | list[nn.Module]): 网络模型，或网络模型列表。
            net_label (str): 网络标签，通常是模型的名称。
            current_iter (int): 当前的迭代次数。
            param_key (str | list[str]): 网络参数的关键字，默认为 'params'。
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        # 确保是列表形式
        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # 去除不必要的 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # 尝试保存模型，最多重试三次
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """打印加载模型时键名不同或尺寸不同的情况。

        Args:
            crt_net (torch model): 当前网络模型。
            load_net (dict): 加载的网络参数。
            strict (bool): 是否严格加载。如果为 False，则允许尺寸不同的键。
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # 检查同名键的尺寸是否一致
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """加载模型参数。

        Args:
            load_path (str): 模型文件路径。
            net (nn.Module): 需要加载的网络模型。
            strict (bool): 是否严格加载。
            param_key (str): 加载参数时的键，默认为 'params'。
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')

        # 去掉不必要的 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """保存训练状态，用于恢复训练。

        Args:
            epoch (int): 当前 epoch。
            current_iter (int): 当前迭代数。
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # 遇到错误时重试 3 次
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def resume_training(self, resume_state):
        """恢复训练，加载优化器和调度器状态。

        Args:
            resume_state (dict): 恢复状态字典。
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), '优化器数量不匹配'
        assert len(resume_schedulers) == len(self.schedulers), '调度器数量不匹配'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def reduce_loss_dict(self, loss_dict):
        """减少损失字典中的损失值（支持分布式训练时进行平均）。

        Args:
            loss_dict (OrderedDict): 包含损失值的字典。

        Returns:
            OrderedDict: 平均后的损失值字典。
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)  # 在分布式训练中聚合损失
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']  # 对所有 GPU 平均损失
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
