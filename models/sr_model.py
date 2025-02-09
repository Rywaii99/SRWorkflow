import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from archs import build_network
from losses import build_loss
from metrics import calculate_metric
from utils import get_root_logger, imwrite, tensor2img
from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """用于单图像超分辨率的基本SR模型"""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # 定义网络
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # 如果是训练模式（is_train 为 True），则初始化训练设置
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        # 设置 EMA（指数移动平均）参数
        """
            如果配置中启用了 EMA（ema_decay > 0），
            则使用 EMA 技术创建一个 net_g_ema 网络副本，
            用于测试时的推理。这有助于提高模型的稳定性和性能。
        """
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 使用网络权重初始化 EMA 模型
            self.net_g_ema.eval()

        # 定义损失函数
        """
            根据训练配置，定义像素损失（cri_pix）和感知损失（cri_perceptual），
            如果没有配置损失函数，则抛出异常。
        """
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # 设置优化器和学习率调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        # 遍历网络的所有参数，筛选出需要优化的参数。
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        # 根据配置的优化器类型，创建优化器 optimizer_g，并将其添加到优化器列表中。
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        # 将输入数据（低分辨率图像 lq）和目标数据（高分辨率图像 gt）传入网络。
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # 计算网络的输出，计算损失并进行反向传播。
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()

        # 像素损失
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # 感知损失
        if self.cri_perceptual:
            # 计算感知损失和风格损失。
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        # 计算总损失并反向传播，更新优化器。
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # 如果启用了 EMA，更新 EMA 模型。
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        # 根据是否使用 EMA 模型，选择合适的网络进行推理。
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 如果是主进程（rank=0），执行非分布式验证
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # 获取数据集名称
        dataset_name = dataloader.dataset.opt['name']
        # 判断是否需要计算度量指标
        with_metrics = self.opt['val'].get('metrics') is not None
        # 判断是否使用进度条（pbar）
        use_pbar = self.opt['val'].get('pbar', False)
        # 判断是否使用分块测试(block-by-block)
        block_by_block = self.opt['val'].get('block_by_block', False)

        # 如果需要计算度量指标，并且是第一次执行，则初始化指标结果
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # 只在第一次执行时初始化
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # 初始化每个数据集的最佳度量指标结果（支持多个验证数据集）
            self._initialize_best_metric_results(dataset_name)

        # 如果需要计算度量指标，清空之前的指标结果
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        # 用来存储度量指标数据
        metric_data = dict()

        # 如果需要使用进度条，初始化进度条
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # 遍历验证数据集
        for idx, val_data in enumerate(dataloader):
            # 获取当前图像的文件名（去除扩展名）
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            # 输入数据，执行测试
            self.feed_data(val_data)

            if block_by_block:
                # 获取图像尺寸
                b, c, h, w = self.lq.shape
                self.origin_lq = self.lq
                factor = self.opt['scale']
                tp = self.opt['val']['patch_size']
                ip = tp // factor

                # 初始化输出图像
                sr = torch.zeros((b, c, h * factor, w * factor))

                # 按块处理图像
                for iy in range(0, h, ip):
                    if iy + ip > h:
                        iy = h - ip
                    ty = factor * iy

                    for ix in range(0, w, ip):
                        if ix + ip > w:
                            ix = w - ip
                        tx = factor * ix

                        # 提取当前块
                        lr_p = self.origin_lq[:, :, iy:iy + ip, ix:ix + ip]
                        self.lq = lr_p.to(self.device)  # 把当前块传到GPU
                        self.test()

                        # 获取超分辨率结果
                        visuals = self.get_current_visuals()
                        sr_p = visuals['result']

                        # 将每个块的结果放回到全图的位置
                        sr[:, :, ty:ty + tp, tx:tx + tp] = sr_p

                # 恢复拼接后的图像
                self.output = sr
                self.lq = self.origin_lq
            else:
                self.test()

            # 获取当前的视觉输出（低质量图像，超分辨率图像，地面真值图像）
            visuals = self.get_current_visuals()

            # 将超分辨率图像转换为图像格式
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img

            # 如果有地面真值图像（gt），将其转换为图像格式
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt  # 删除地面真值图像，释放内存

            # 为了避免 GPU 内存溢出，手动删除低质量图像和输出，清理缓存
            del self.lq
            if hasattr(self, 'origin_lq'):
                del self.origin_lq
            del self.output
            torch.cuda.empty_cache()

            # 如果需要保存图像，将超分辨率图像保存到指定路径
            if save_img:
                if self.opt['is_train']:  # 如果是训练模式
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:  # 如果是验证模式
                    if self.opt['val']['suffix']:  # 如果有后缀
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            # 如果需要计算度量指标
            if with_metrics:
                # 遍历所有指标名称，计算每个指标的值
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            # 如果使用进度条，更新进度条
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        # 如果使用进度条，关闭进度条
        if use_pbar:
            pbar.close()

        # 如果需要计算度量指标
        if with_metrics:
            # 计算每个度量指标的平均值
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # 更新当前最佳度量指标结果
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            # 将验证的度量指标值记录到日志中
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        # 构建日志字符串
        log_str = f'Validation {dataset_name}\n'

        # 遍历所有度量指标，将其值记录到日志中
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            # 如果有最佳指标结果，则输出最佳值和对应的迭代次数
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        # 输出日志信息
        logger = get_root_logger()
        logger.info(log_str)

        # 如果有 TensorBoard logger，则将度量指标记录到 TensorBoard
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        # 获取当前的视觉输出，包括低质量图像（lq）、超分辨率图像（result）和地面真值图像（gt）
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()  # 低质量图像
        out_dict['result'] = self.output.detach().cpu()  # 超分辨率图像
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()  # 地面真值图像
        return out_dict

    def save(self, epoch, current_iter):
        # 如果有 ema（指数移动平均）的网络参数，保存网络和 ema 参数
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            # 否则，只保存网络参数
            self.save_network(self.net_g, 'net_g', current_iter)
        # 保存训练状态（epoch 和当前迭代次数）
        self.save_training_state(epoch, current_iter)







