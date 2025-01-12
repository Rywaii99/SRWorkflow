import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir
from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# 自动扫描并导入模型模块
# 扫描 'models' 文件夹下所有以 '_model.py' 结尾的文件
model_folder = osp.dirname(osp.abspath(__file__))  # 获取当前文件所在目录的绝对路径
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]  # 获取所有以 '_model.py' 结尾的文件名（不包括扩展名）
# 导入所有模型模块
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]  # 动态导入每个模型模块


def build_model(opt):
    """根据配置文件选项构建模型。

        Args：
            opt (dict): 配置字典，必须包含以下内容：
                model_type (str): 模型类型的名称，用于从模型注册器中查找并实例化模型。

        Return：
            model (nn.Module): 构建好的模型。
        """
    opt = deepcopy(opt)  # 深拷贝配置字典，避免修改原始字典
    # 从模型注册器中获取并实例化相应的模型
    model = MODEL_REGISTRY.get(opt['model_type'])(opt)

    # 获取日志记录器
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')  # 输出模型创建信息

    return model  # 返回构建好的模型