import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler


class MultiStepRestartLR(_LRScheduler):
    """
    结合了MultiStep学习率衰减和重启机制的学习率调度器。

    该调度器在达到指定的里程碑时会按照衰减因子（gamma）降低学习率，并在指定的重启点重置学习率为初始值。

    Args:
        optimizer (torch.nn.optimizer): 用于训练的PyTorch优化器。
        milestones (list): 学习率衰减的迭代次数（epoch）。例如 [30, 80] 表示在第30、80个epoch时进行衰减。
        gamma (float): 学习率衰减因子。默认值是 0.1。每当达到一个里程碑时，学习率会乘以 `gamma`。
        restarts (list): 指定学习率重启的迭代次数。默认值是 [0]，即不重启。
        restart_weights (list): 每次重启时的学习率缩放因子。重启时的学习率会乘以对应的 `restart_weights` 中的权重。
        last_epoch (int): 用于初始化调度器的最后一个epoch，默认值为 -1。
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0, ), restart_weights=(1, ), last_epoch=-1):
        # 将里程碑转化为Counter，用于快速查询
        self.milestones = Counter(milestones)
        self.gamma = gamma  # 学习率衰减因子
        self.restarts = restarts  # 重启点列表
        self.restart_weights = restart_weights  # 重启时的学习率缩放因子

        # 确保重启点和重启权重的长度一致
        assert len(self.restarts) == len(self.restart_weights), 'restarts 和 restart_weights 长度不匹配。'

        # 调用父类初始化
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        返回当前epoch的学习率。

        如果当前epoch是一个重启点，则将学习率重置为初始值并根据重启权重进行缩放。
        如果当前epoch是一个里程碑，学习率会按衰减因子（gamma）降低。
        否则，保持当前的学习率不变。

        Returns:
            list: 当前学习率的列表，对应每个参数组。
        """
        # 如果当前epoch是重启点，则根据重启权重调整学习率
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]

        # 如果当前epoch不是里程碑，则返回当前学习率
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]

        # 如果当前epoch是里程碑，则按gamma衰减学习率
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    """
    根据当前的迭代次数，获取所在周期的索引。

    例如，给定累计周期 [100, 200, 300, 400] 和当前迭代次数，
    如果迭代次数是50，返回0；如果迭代次数是210，返回2；如果迭代次数是300，返回2。

    Args:
        iteration (int): 当前迭代次数。
        cumulative_period (list[int]): 累积周期列表，每个元素表示一个周期的结束时间（迭代次数）。

    Returns:
        int: 当前迭代所在周期的索引。
    """
    # 遍历所有周期，找到当前迭代所在的周期
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """
    结合了余弦退火（Cosine Annealing）和重启机制的学习率调度器。

    每个周期内，学习率会根据余弦函数衰减。每当达到指定的周期结束时（由 periods 列表指定），
    学习率会重启并根据 restart_weights 缩放。

    Args:
        optimizer (torch.nn.optimizer): 用于训练的PyTorch优化器。
        periods (list): 每个周期的长度（单位：epoch）。例如 [10, 10, 10, 10] 表示4个周期，每个周期10个epoch。
        restart_weights (list): 每个周期结束时学习率的缩放因子。默认值是 [1]。
        eta_min (float): 学习率衰减到的最小值，默认是 0。
        last_epoch (int): 用于初始化调度器的最后一个epoch，默认值为 -1。
    """

    def __init__(self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1):
        # 初始化周期和重启权重
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min  # 最小学习率
        assert (len(self.periods) == len(self.restart_weights)), 'periods 和 restart_weights 长度必须相同。'

        # 计算每个周期的累计结束时间
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]

        # 调用父类初始化
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        返回当前epoch的学习率，使用余弦退火衰减，并在周期结束时重启学习率。

        学习率的计算基于当前的周期，并且每个周期结束时会重启学习率，根据重启权重进行缩放。

        Returns:
            list: 当前学习率的列表，对应每个参数组。
        """
        # 获取当前迭代所在的周期索引
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)

        # 获取当前周期的学习率缩放因子
        current_weight = self.restart_weights[idx]

        # 获取当前周期的起始迭代次数（上一个周期的结束时间）
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]

        # 获取当前周期的长度
        current_period = self.periods[idx]

        # 计算余弦退火衰减后的学习率
        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """
    Cosine annealing with restarts learning rate scheme, where each cycle can have a different minimum learning rate.

    例如，配置如下：
    periods = [10, 10, 10, 10]  # 每个周期的长度
    restart_weights = [1, 0.5, 0.5, 0.5]  # 每次重启时的学习率权重
    eta_mins = [1e-7, 1e-6, 1e-5, 1e-4]  # 每个周期的最小学习率

    这个调度器有四个周期，每个周期长度为 10 次迭代。在每个周期结束时（即10、20、30次迭代），
    调度器会使用 `restart_weights` 中对应的权重重新启动，并且每个周期的最小学习率由 `eta_mins` 列表指定。

    Args:
        optimizer (torch.nn.optimizer): 用于优化的 PyTorch 优化器。
        periods (list): 每个余弦退火周期的长度。
        restart_weights (list): 每次重启时的学习率权重。默认值为 [1]。
        eta_mins (list): 每个周期的最小学习率，允许每个周期有不同的最小学习率。默认值为 [0]。
        last_epoch (int): 用于 _LRScheduler 的最后一个 epoch，默认为 -1。

    说明：
        - `periods` 和 `restart_weights` 的长度应该一致，且每个周期都有一个对应的最小学习率 `eta_mins`。
        - 在每个周期内，学习率会按余弦函数衰减，直到到达最小学习率（`eta_min`）。
        - 每当一个周期结束时（`last_epoch` 达到当前周期的末尾），学习率会根据 `restart_weights` 重新启动。
        - `eta_mins` 允许每个周期的最小学习率不同，从而支持更加灵活的调度策略。
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1,),
                 eta_mins=(0,),
                 last_epoch=-1):
        # 保存初始化参数
        self.periods = periods
        self.init_preiods = periods  # 备份初始的 periods 列表
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins  # 每个周期对应不同的最小学习率

        # 确保 periods 和 restart_weights 的长度一致
        assert (len(self.periods) == len(self.restart_weights)), \
            'periods 和 restart_weights 长度必须相同。'

        # 计算每个周期的累积周期
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        # 初始化父类
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        根据当前 epoch 返回调整后的学习率。

        在每个周期内，学习率根据余弦退火衰减，并且周期结束时会根据 restart_weights 重启。
        每个周期的最小学习率可以不同，由 eta_mins 控制。

        Returns:
            list: 每个参数组的学习率。
        """
        # 找到当前 epoch 所在的周期索引
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)

        # 获取当前周期的重启权重
        current_weight = self.restart_weights[idx]

        # 找到上一个周期的末尾 epoch，用于计算距离重启的步数
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]

        # 当前周期的长度
        current_period = self.periods[idx]

        # 获取当前周期的最小学习率
        eta_min = self.eta_mins[idx]

        # 计算每个参数组的学习率
        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
