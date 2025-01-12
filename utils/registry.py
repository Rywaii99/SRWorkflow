# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501


class Registry():
    """
    提供名称 -> 对象 映射的注册表，用于支持第三方用户的自定义模块。

    创建一个注册表（例如，用于存放骨干网络）：

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    注册一个对象：

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    或者：

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        初始化注册表。

        参数:
            name (str): 注册表的名称
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        """
        将对象注册到注册表中。

        参数:
            name (str): 对象的名称
            obj (object): 要注册的对象
            suffix (str, 可选): 可选的后缀，用于生成唯一名称
        """
        if isinstance(suffix, str):
            name = name + '_' + suffix

        # 检查是否已经注册过该名称
        assert (name not in self._obj_map), (f"对象 '{name}' 已经在 '{self._name}' 注册表中注册！")
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        注册对象。可以作为装饰器使用，也可以直接调用。

        参数:
            obj (object): 要注册的对象，默认为 None。
            suffix (str, 可选): 对象名称的后缀
        """
        if obj is None:
            # 作为装饰器使用
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # 作为函数调用使用
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='srworkflow'):
        """
        获取注册表中的对象。

        参数:
            name (str): 要查找的对象名称
            suffix (str, 可选): 后缀，默认 'srworkflow'
        """
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'未找到名称 {name}，使用名称: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"在 '{self._name}' 注册表中没有找到名为 '{name}' 的对象!")
        return ret

    def __contains__(self, name):
        """检查名称是否在注册表中"""
        return name in self._obj_map

    def __iter__(self):
        """返回注册表的迭代器"""
        return iter(self._obj_map.items())

    def keys(self):
        """返回所有注册表的键"""
        return self._obj_map.keys()


# 定义多个具体的注册表
DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
