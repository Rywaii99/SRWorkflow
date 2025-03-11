import torch
from collections import OrderedDict

from archs import build_network
from losses import build_loss
from utils import get_root_logger
from utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """
    SRGAN模型，用于单张图像的超分辨率。
    """

    def init_training_settings(self):
        """
        初始化训练设置的方法。
        该方法用于设置网络、损失函数、优化器等训练相关的参数和配置。
        """
        train_opt = self.opt['train']  # 获取训练配置选项

        self.ema_decay = train_opt.get('ema_decay', 0)  # 获取指数移动平均（EMA）的衰减率，默认值为0
        if self.ema_decay > 0:
            logger = get_root_logger()  # 获取根日志记录器
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # 定义带有指数移动平均（EMA）的生成器网络 net_g_ema
            # net_g_ema 仅用于在单个GPU上进行测试和保存
            # 无需使用 DistributedDataParallel 进行包装
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # 加载预训练模型
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # 复制 net_g 的权重
            self.net_g_ema.eval()  # 将 net_g_ema 设置为评估模式

        # 定义判别器网络 net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)  # 将网络移动到指定设备（如GPU）
        self.print_network(self.net_d)  # 打印网络结构信息

        # 加载预训练模型
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        self.net_g.train()  # 将生成器网络 net_g 设置为训练模式
        self.net_d.train()  # 将判别器网络 net_d 设置为训练模式

        # 定义损失函数
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)  # 构建像素损失函数并移动到指定设备
        else:
            self.cri_pix = None  # 没有像素损失配置则设为None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)  # 构建 ldl 损失函数并移动到指定设备
        else:
            self.cri_ldl = None  # 没有 ldl 损失配置则设为None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)  # 构建感知损失函数并移动到指定设备
        else:
            self.cri_perceptual = None  # 没有感知损失配置则设为None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)  # 构建生成对抗损失函数并移动到指定设备

        self.net_d_iters = train_opt.get('net_d_iters', 1)  # 获取判别器训练间隔，默认值为1
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)  # 获取判别器初始化迭代次数，默认值为0

        # 设置优化器和调度器
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """
        设置优化器的方法。
        该方法用于根据配置为生成器网络和判别器网络分别设置优化器。
        """
        train_opt = self.opt['train']  # 获取训练配置选项
        # 生成器网络的优化器
        optim_type = train_opt['optim_g'].pop('type')  # 获取优化器类型
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)  # 将生成器优化器添加到优化器列表中

        # 判别器网络的优化器
        optim_type = train_opt['optim_d'].pop('type')  # 获取优化器类型
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)  # 将判别器优化器添加到优化器列表中

    def optimize_parameters(self, current_iter):
        """
        优化网络参数的方法。

        参数:
            current_iter (int): 当前的迭代次数。
        """
        # 优化生成器网络（net_g）
        for p in self.net_d.parameters():
            p.requires_grad = False  # 冻结判别器网络的参数，不计算其梯度

        self.optimizer_g.zero_grad()  # 清空生成器网络的梯度
        self.output = self.net_g(self.lq)  # 使用生成器网络对低质量图像（lq）进行处理，得到输出

        l_g_total = 0  # 初始化生成器的总损失为0
        loss_dict = OrderedDict()  # 用于存储不同损失项的有序字典

        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 像素损失
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output, self.gt)  # 计算生成器输出与真实图像（gt）之间的像素损失
                l_g_total += l_g_pix  # 将像素损失加到总损失中
                loss_dict['l_g_pix'] = l_g_pix  # 将像素损失添加到损失字典中

            # 感知损失
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)  # 计算感知损失和风格损失
                if l_g_percep is not None:
                    l_g_total += l_g_percep  # 将感知损失加到总损失中
                    loss_dict['l_g_percep'] = l_g_percep  # 将感知损失添加到损失字典中
                if l_g_style is not None:
                    l_g_total += l_g_style  # 将风格损失加到总损失中
                    loss_dict['l_g_style'] = l_g_style  # 将风格损失添加到损失字典中

            # 生成对抗网络损失（GAN loss）
            fake_g_pred = self.net_d(self.output)  # 判别器对生成器生成图像的预测结果
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)  # 计算生成器的生成对抗损失
            l_g_total += l_g_gan  # 将生成对抗损失加到总损失中
            loss_dict['l_g_gan'] = l_g_gan  # 将生成对抗损失添加到损失字典中

            l_g_total.backward()  # 反向传播计算生成器的梯度
            self.optimizer_g.step()  # 更新生成器的参数

        # 优化判别器网络（net_d）
        for p in self.net_d.parameters():
            p.requires_grad = True  # 解冻判别器网络的参数，计算其梯度

        self.optimizer_d.zero_grad()  # 清空判别器网络的梯度

        # 真实图像
        real_d_pred = self.net_d(self.gt)  # 判别器对真实图像的预测结果
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)  # 计算判别器对真实图像的损失
        loss_dict['l_d_real'] = l_d_real  # 将判别器对真实图像的损失添加到损失字典中
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())  # 将判别器对真实图像预测结果的均值添加到损失字典中
        l_d_real.backward()  # 反向传播计算判别器对真实图像的梯度

        # 虚假图像
        fake_d_pred = self.net_d(self.output.detach())  # 判别器对生成器生成图像（分离梯度）的预测结果
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)  # 计算判别器对虚假图像的损失
        loss_dict['l_d_fake'] = l_d_fake  # 将判别器对虚假图像的损失添加到损失字典中
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())  # 将判别器对虚假图像预测结果的均值添加到损失字典中
        l_d_fake.backward()  # 反向传播计算判别器对虚假图像的梯度
        self.optimizer_d.step()  # 更新判别器的参数

        self.log_dict = self.reduce_loss_dict(loss_dict)  # 对损失字典进行处理

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)  # 执行指数移动平均（EMA）更新模型参数

    def save(self, epoch, current_iter):
        """
        保存模型的方法。

        参数:
            epoch (int): 当前的训练轮数。
            current_iter (int): 当前的迭代次数。
        """
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)  # 保存生成器网络
        self.save_network(self.net_d, 'net_d', current_iter)  # 保存判别器网络
        self.save_training_state(epoch, current_iter)  # 保存训练状态（如优化器状态、调度器状态等）