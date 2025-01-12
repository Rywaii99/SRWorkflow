import importlib  # 用于动态导入模块
from copy import deepcopy  # 用于深拷贝对象
from os import path as osp  # 用于文件路径操作，常用别名 osp

from utils import get_root_logger, scandir
from utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# 自动扫描并导入所有架构模块到注册表
# 扫描 'archs' 文件夹下的所有文件，收集以 '_arch.py' 结尾的文件
arch_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件（此文件）的绝对路径所在的目录
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]  # 获取该目录下所有以 '_arch.py' 结尾的文件名（不包括扩展名）
# 导入所有的架构模块
_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]  # 动态导入每个架构模块


def build_network(opt):
    opt = deepcopy(opt)  # 对输入的配置选项 opt 进行深拷贝，防止修改原始配置
    network_type = opt.pop('type')  # 从 opt 中弹出 'type' 键，获取网络类型（即网络架构的名称）
    net = ARCH_REGISTRY.get(network_type)(**opt)  # 从 ARCH_REGISTRY 中获取网络架构构造函数，并传入剩余的配置参数创建网络实例
    logger = get_root_logger()  # 获取根日志器（用于日志记录）
    logger.info(f'Network [{net.__class__.__name__}] is created.')  # 输出日志，表示网络创建完成
    return net  # 返回创建的网络实例

