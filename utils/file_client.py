from abc import ABCMeta, abstractmethod


from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """存储后端的抽象基类。

    所有的存储后端都需要实现两个方法：`get()` 和 `get_text()`。
    `get()` 用于读取文件作为字节流，`get_text()` 用于读取文件的文本内容。
    """

    @abstractmethod
    def get(self, filepath):
        """获取指定路径的文件内容，返回字节流。"""
        pass

    @abstractmethod
    def get_text(self, filepath):
        """获取指定路径的文件内容，返回文本数据。"""
        pass


class MemcachedBackend(BaseStorageBackend):
    """Memcached 存储后端。

    Attributes:
        server_list_cfg (str): Memcached 服务器列表的配置文件。
        client_cfg (str): Memcached 客户端的配置文件。
        sys_path (str | None): 额外的路径，默认值为 `None`，若提供将会添加到 `sys.path` 中。
    """

    def __init__(self, server_list_cfg, client_cfg, sys_path=None):
        if sys_path is not None:
            import sys
            sys.path.append(sys_path)  # 如果提供了 sys_path，则将其添加到 sys.path 中
        try:
            import mc  # 尝试导入 Memcached 客户端库
        except ImportError:
            raise ImportError('Please install memcached to enable MemcachedBackend.')

        self.server_list_cfg = server_list_cfg
        self.client_cfg = client_cfg
        self._client = mc.MemcachedClient.GetInstance(self.server_list_cfg, self.client_cfg)
        # mc.pyvector 是指向内存缓存的指针
        self._mc_buffer = mc.pyvector()

    def get(self, filepath):
        """从 Memcached 获取文件内容。

        Args:
            filepath (str): 文件路径。

        Returns:
            value_buf: 文件内容的字节流。
        """
        filepath = str(filepath)  # 转换为字符串类型
        import mc
        self._client.Get(filepath, self._mc_buffer)  # 从 Memcached 获取数据
        value_buf = mc.ConvertBuffer(self._mc_buffer)  # 将获取到的数据转换为字节流
        return value_buf

    def get_text(self, filepath):
        """Memcached 后端不支持获取文本数据，抛出异常。"""
        raise NotImplementedError


class HardDiskBackend(BaseStorageBackend):
    """硬盘存储后端。

    从硬盘读取文件并返回字节流或文本数据。
    """

    def get(self, filepath):
        """从硬盘读取文件内容。

        Args:
            filepath (str): 文件路径。

        Returns:
            value_buf: 文件内容的字节流。
        """
        filepath = str(filepath)  # 转换为字符串类型
        with open(filepath, 'rb') as f:  # 以二进制方式打开文件
            value_buf = f.read()  # 读取文件内容
        return value_buf

    def get_text(self, filepath):
        """从硬盘读取文件内容并返回文本数据。

        Args:
            filepath (str): 文件路径。

        Returns:
            value_buf: 文件内容的文本数据。
        """
        filepath = str(filepath)  # 转换为字符串类型
        with open(filepath, 'r') as f:  # 以文本方式打开文件
            value_buf = f.read()  # 读取文件内容
        return value_buf


class LmdbBackend(BaseStorageBackend):
    """LMDB 存储后端。

    适用于从 LMDB 数据库读取文件。

    Args:
        db_paths (str | list[str]): LMDB 数据库路径。
        client_keys (str | list[str]): LMDB 客户端键。默认值为 'default'。
        readonly (bool, optional): LMDB 环境参数，若为 True，则禁止写操作。默认值为 True。
        lock (bool, optional): LMDB 环境参数，若为 False，则在并发访问时不锁定数据库。默认值为 False。
        readahead (bool, optional): LMDB 环境参数，若为 False，则禁用操作系统的文件预读取机制。默认值为 False。

    Attributes:
        db_paths (list): LMDB 数据库路径列表。
        _client (list): LMDB 环境的客户端列表。
    """

    def __init__(self, db_paths, client_keys='default', readonly=True, lock=False, readahead=False, **kwargs):
        try:
            import lmdb  # 尝试导入 LMDB 库
        except ImportError:
            raise ImportError('Please install lmdb to enable LmdbBackend.')

        if isinstance(client_keys, str):
            client_keys = [client_keys]  # 如果是字符串，则转换为列表
        if isinstance(db_paths, list):
            self.db_paths = [str(v) for v in db_paths]  # 转换为字符串列表
        elif isinstance(db_paths, str):
            self.db_paths = [str(db_paths)]  # 转换为字符串列表
        assert len(client_keys) == len(self.db_paths), ('client_keys 和 db_paths 的长度必须一致，'
                                                        f'但接收到的是 {len(client_keys)} 和 {len(self.db_paths)}。')

        self._client = {}
        for client, path in zip(client_keys, self.db_paths):
            self._client[client] = lmdb.open(path, readonly=readonly, lock=lock, readahead=readahead, **kwargs)

    def get(self, filepath, client_key):
        """从指定的 LMDB 环境中获取文件内容。

        Args:
            filepath (str | obj:`Path`): LMDB 数据库中的键。
            client_key (str): 用于区分不同 LMDB 环境的键。

        Returns:
            value_buf: 文件内容的字节流。
        """
        filepath = str(filepath)  # 转换为字符串类型
        assert client_key in self._client, (f'client_key {client_key} 不在 LMDB 客户端列表中。')
        client = self._client[client_key]
        with client.begin(write=False) as txn:  # 以只读方式打开事务
            value_buf = txn.get(filepath.encode('ascii'))  # 获取文件内容
        return value_buf

    def get_text(self, filepath):
        """LMDB 后端不支持获取文本数据，抛出异常。"""
        raise NotImplementedError


class FileClient(object):
    """通用的文件客户端，用于访问不同后端的文件。

    该客户端从指定后端的路径加载文件或文本，并返回其二进制内容。它还可以注册其他后端访问器。

    Attributes:
        backend (str): 存储后端类型，支持 "disk"（硬盘）、"memcached"（Memcached）和 "lmdb"（LMDB）。
        client (:obj:`BaseStorageBackend`): 存储后端对象。
    """

    _backends = {
        'disk': HardDiskBackend,
        'memcached': MemcachedBackend,
        'lmdb': LmdbBackend,
    }

    def __init__(self, backend='disk', **kwargs):
        """初始化文件客户端。

        Args:
            backend (str): 存储后端类型，支持 "disk"、"memcached" 和 "lmdb"。
            **kwargs: 传递给存储后端的其他参数。

        Raises:
            ValueError: 如果提供的 `backend` 不在已支持的后端列表中。
        """
        if backend not in self._backends:
            raise ValueError(f'Backend {backend} is not supported. Currently supported ones'
                             f' are {list(self._backends.keys())}')
        self.backend = backend
        self.client = self._backends[backend](**kwargs)

    def get(self, filepath, client_key='default'):
        """获取指定路径的文件内容。

        Args:
            filepath (str): 文件路径。
            client_key (str): 仅在 LMDB 后端中使用，用于区分不同的 LMDB 环境。

        Returns:
            value_buf: 文件内容的字节流。
        """
        if self.backend == 'lmdb':
            return self.client.get(filepath, client_key)  # LMDB 后端需要提供 client_key
        else:
            return self.client.get(filepath)

    def get_text(self, filepath):
        """获取指定路径的文件内容（文本）。

        Args:
            filepath (str): 文件路径。

        Returns:
            value_buf: 文件内容的文本数据。
        """
        return self.client.get_text(filepath)
