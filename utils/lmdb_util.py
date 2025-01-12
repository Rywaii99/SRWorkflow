import cv2  # 用于读取和写入图像文件
import lmdb  # 用于创建和操作LMDB数据库
import sys  # 用于系统操作，如退出程序等
from multiprocessing import Pool  # 用于并行处理
from os import path as osp  # 用于处理路径
from tqdm import tqdm  # 用于进度条显示


def make_lmdb_from_imgs(data_path,
                        lmdb_path,
                        img_path_list,
                        keys,
                        batch=5000,
                        compress_level=1,
                        multiprocessing_read=False,
                        n_thread=40,
                        map_size=None):
    """
    从图像文件创建 LMDB 数据库。

    参数：
        data_path (str): 存放图像的文件夹路径。
        lmdb_path (str): LMDB 文件保存的路径。
        img_path_list (list): 图像文件的路径列表。
        keys (list): 用于图像的 LMDB 键值列表。
        batch (int): 每处理一批图像后提交 LMDB 数据库。默认 5000。
        compress_level (int): 图像压缩级别。默认 1。
        multiprocessing_read (bool): 是否使用多进程读取图像。默认 False。
        n_thread (int): 多进程的线程数。
        map_size (int | None): LMDB 数据库的映射大小。如果未提供，则根据图像大小估算。

    返回：
        无
    """

    assert len(img_path_list) == len(keys), ('img_path_list 和 keys 长度必须相同, '
                                             f'但分别为 {len(img_path_list)} 和 {len(keys)}')
    print(f'为 {data_path} 创建 LMDB 数据库，保存到 {lmdb_path}...')
    print(f'总图像数: {len(img_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path 必须以 '.lmdb' 结尾。")
    if osp.exists(lmdb_path):
        print(f'文件夹 {lmdb_path} 已存在，退出。')
        sys.exit(1)


    '''多进程读取图像'''

    if multiprocessing_read:
        # 使用多进程读取所有图像到内存（多线程）
        dataset = {}  # 使用字典保持顺序
        shapes = {}  # 存储图像形状信息
        print(f'使用多线程读取图像，线程数: {n_thread} ...')
        pbar = tqdm(total=len(img_path_list), unit='image')

        def callback(arg):
            """回调函数，获取图像数据并更新进度条。"""
            key, dataset[key], shapes[key] = arg
            pbar.update(1)
            pbar.set_description(f'读取 {key}')

        pool = Pool(n_thread)
        for path, key in zip(img_path_list, keys):
            pool.apply_async(read_img_worker, args=(osp.join(data_path, path), key, compress_level), callback=callback)
        pool.close()
        pool.join()
        pbar.close()
        print(f'完成读取 {len(img_path_list)} 张图像。')

    '''创建 LMDB 环境'''

    if map_size is None:
        # 如果没有指定 map_size，则根据第一张图像的大小估算
        img = cv2.imread(osp.join(data_path, img_path_list[0]), cv2.IMREAD_UNCHANGED)
        _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
        data_size_per_img = img_byte.nbytes
        print('每张图像的大小为: ', data_size_per_img)
        data_size = data_size_per_img * len(img_path_list)
        map_size = data_size * 10

    env = lmdb.open(lmdb_path, map_size=map_size)

    '''将图像数据写入 LMDB 数据库'''

    pbar = tqdm(total=len(img_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(img_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'写入 {key}')
        key_byte = key.encode('ascii')
        if multiprocessing_read:
            img_byte = dataset[key]
            h, w, c = shapes[key]
        else:
            _, img_byte, img_shape = read_img_worker(osp.join(data_path, path), key, compress_level)
            h, w, c = img_shape

        txn.put(key_byte, img_byte)
        # 写入元数据
        txt_file.write(f'{key}.png ({h},{w},{c}) {compress_level}\n')
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\n完成写入 LMDB。')


def read_img_worker(path, key, compress_level):
    """读取图像的辅助函数。

    参数：
        path (str): 图像路径。
        key (str): 图像的键值。
        compress_level (int): 压缩级别。

    返回：
        key (str): 图像的键值。
        img_byte (byte): 图像的字节数据。
        img_shape (tuple[int]): 图像的形状 (height, width, channels)。
    """

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:  # 如果是灰度图像
        h, w = img.shape
        c = 1
    else:  # 如果是彩色图像
        h, w, c = img.shape
    _, img_byte = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level])
    return (key, img_byte, (h, w, c))


class LmdbMaker():
    """LMDB 创建器。

    参数：
        lmdb_path (str): LMDB 文件保存路径。
        map_size (int): LMDB 映射大小，默认 1TB。
        batch (int): 每处理一批图像后提交 LMDB。默认 5000。
        compress_level (int): 图像压缩级别。默认 1。
    """

    def __init__(self, lmdb_path, map_size=1024**4, batch=5000, compress_level=1):
        if not lmdb_path.endswith('.lmdb'):
            raise ValueError("lmdb_path 必须以 '.lmdb' 结尾。")
        if osp.exists(lmdb_path):
            print(f'文件夹 {lmdb_path} 已存在，退出。')
            sys.exit(1)

        self.lmdb_path = lmdb_path
        self.batch = batch
        self.compress_level = compress_level
        self.env = lmdb.open(lmdb_path, map_size=map_size)
        self.txn = self.env.begin(write=True)
        self.txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
        self.counter = 0

    def put(self, img_byte, key, img_shape):
        """将图像数据插入 LMDB。

        参数：
            img_byte (byte): 图像字节数据。
            key (str): 图像的键值。
            img_shape (tuple[int]): 图像形状 (height, width, channels)。
        """
        self.counter += 1
        key_byte = key.encode('ascii')
        self.txn.put(key_byte, img_byte)
        # 写入元数据
        h, w, c = img_shape
        self.txt_file.write(f'{key}.png ({h},{w},{c}) {self.compress_level}\n')
        if self.counter % self.batch == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        """关闭 LMDB 数据库并提交事务。"""
        self.txn.commit()
        self.env.close()
        self.txt_file.close()





