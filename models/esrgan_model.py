import torch
from collections import OrderedDict

from utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel


@MODEL_REGISTRY.register()
class ESRGANModel(SRGANModel):
    """
    ESRGAN模型，用于单张图像的超分辨率任务。
    """

    def optimize_parameters(self, current_iter):
        """
        优化网络参数的方法。

        参数:
            current_iter (int): 当前的迭代次数。
        """
        # 冻结判别器网络（net_d）的参数，不计算其梯度
        for p in self.net_d.parameters():
            p.requires_grad = False

        # 清空生成器网络（net_g）的梯度
        self.optimizer_g.zero_grad()
        # 使用生成器网络（net_g）对低质量图像（lq）进行处理，得到输出
        self.output = self.net_g(self.lq)

        l_g_total = 0  # 初始化生成器的总损失为0
        loss_dict = OrderedDict()  # 用于存储不同损失项的有序字典

        # 如果当前迭代次数是判别器训练间隔（net_d_iters）的倍数，并且大于判别器初始化迭代次数（net_d_init_iters）
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 像素损失
            if self.cri_pix:
                # 计算生成器输出与真实图像（gt）之间的像素损失
                l_g_pix = self.cri_pix(self.output, self.gt)
                l_g_total += l_g_pix  # 将像素损失加到总损失中
                loss_dict['l_g_pix'] = l_g_pix  # 将像素损失添加到损失字典中

            # 感知损失
            if self.cri_perceptual:
                # 计算生成器输出与真实图像之间的感知损失和风格损失
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep  # 将感知损失加到总损失中
                    loss_dict['l_g_percep'] = l_g_percep  # 将感知损失添加到损失字典中
                if l_g_style is not None:
                    l_g_total += l_g_style  # 将风格损失加到总损失中
                    loss_dict['l_g_style'] = l_g_style  # 将风格损失添加到损失字典中

            # 生成对抗网络损失（相对论性生成对抗网络）
            # 判别器对真实图像的预测结果， detach() 表示不计算梯度
            real_d_pred = self.net_d(self.gt).detach()
            # 判别器对生成器生成图像的预测结果
            fake_g_pred = self.net_d(self.output)
            # 计算生成器的真实图像损失
            l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            # 计算生成器的虚假图像损失
            l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            l_g_gan = (l_g_real + l_g_fake) / 2  # 计算生成器的生成对抗网络损失

            l_g_total += l_g_gan  # 将生成对抗网络损失加到总损失中
            loss_dict['l_g_gan'] = l_g_gan  # 将生成对抗网络损失添加到损失字典中

            l_g_total.backward()  # 反向传播计算生成器的梯度
            self.optimizer_g.step()  # 更新生成器的参数

        # 解冻判别器网络（net_d）的参数，计算其梯度
        for p in self.net_d.parameters():
            p.requires_grad = True

        # 清空判别器网络（net_d）的梯度
        self.optimizer_d.zero_grad()
        # 生成对抗网络损失（相对论性生成对抗网络）

        # 为了避免在分布式训练中出现错误：
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # 我们将真实图像和虚假图像的反向传播分开，并且在计算均值时分离张量。

        # 真实图像
        # 判别器对生成器生成图像的预测结果， detach() 表示不计算梯度
        fake_d_pred = self.net_d(self.output).detach()
        # 判别器对真实图像的预测结果
        real_d_pred = self.net_d(self.gt)
        # 计算判别器的真实图像损失
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()  # 反向传播计算判别器的真实图像梯度

        # 虚假图像
        # 判别器对生成器生成图像的预测结果， detach() 表示不计算梯度
        fake_d_pred = self.net_d(self.output.detach())
        # 计算判别器的虚假图像损失
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()  # 反向传播计算判别器的虚假图像梯度
        self.optimizer_d.step()  # 更新判别器的参数

        loss_dict['l_d_real'] = l_d_real  # 将判别器的真实图像损失添加到损失字典中
        loss_dict['l_d_fake'] = l_d_fake  # 将判别器的虚假图像损失添加到损失字典中
        # 将判别器对真实图像预测结果的均值添加到损失字典中
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        # 将判别器对虚假图像预测结果的均值添加到损失字典中
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)  # 对损失字典进行处理（可能是为了分布式训练等）

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)  # 执行指数移动平均（EMA）更新模型参数