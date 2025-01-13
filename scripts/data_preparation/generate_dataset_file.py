import os
import random
import shutil
from copy import deepcopy
from tqdm import tqdm


def exclude_train_file(dataset_folder):
    """
    获取训练集中所有图片的文件名，用于排除。

    Args:
        dataset_folder (str): 训练集输出文件夹路径。

    Returns:
        set: 包含所有训练集图片文件名的集合。
    """
    # 获取训练集中所有图片的文件名，用于排除
    train_exclude_list = set()
    for category in os.listdir(dataset_folder):
        category_path = os.path.join(dataset_folder, category)
        if os.path.isdir(category_path):
            train_exclude_list.update(os.listdir(category_path))

    return train_exclude_list


def exclude_train_and_test_file(train_folder, test_folder):
    """
    获取训练集和测试集中所有图片的文件名，用于排除（不考虑分类子文件夹结构）。

    Args:
        - train_folder (str): 训练集文件夹路径。
        - test_folder (str): 测试集文件夹路径。

    Returns:
        set: 包含所有训练集和测试集图片文件名的集合。
    """
    def get_file_names(folder):
        """
        获取指定文件夹下所有文件的文件名。

        Args:
            - folder (str): 文件夹路径。

        Returns:
            set: 包含文件夹中所有文件名的集合。
        """
        file_names = set()
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):  # 确保是文件
                file_names.add(filename)
        return file_names

    # 获取训练集和测试集的文件名
    train_files = get_file_names(train_folder)
    test_files = get_file_names(test_folder)

    # 合并训练集和测试集文件名
    all_files = train_files.union(test_files)

    return all_files

def generate_dataset_aid(sample_size=100, is_train=True, is_val=False, exclude_list=None):
    """
    从每个类别文件夹中随机抽取指定数量的图片，并保存到目标文件夹。

    Args:
        opt:
        - source_folder (str): 源数据集文件夹路径。
        - output_folder (str): 输出文件夹路径。
        - sample_size (int): 每个类别随机抽取的图片数量，默认为 100。
    """
    # 如果目标文件夹不存在，创建它
    dataset = 'AID'
    source_folder = '../../datasets/origin/' + dataset
    output_folder = '../../datasets/processed/' + dataset
    if is_train:
        output_folder = output_folder + '/train_HR'
    else:
        if is_val:
            output_folder = output_folder + '/val_HR'
        else:
            output_folder = output_folder + '/test_HR'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 获取所有类别
        categories = os.listdir(source_folder)

        # 在所有类别图片上加进度条显示
        all_images = []  # 保存所有待处理图片的信息
        for category in categories:
            category_path = os.path.join(source_folder, category)
            if os.path.isdir(category_path):
                images = [img for img in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, img))]

                if not is_train:
                    # 测试集不包含训练集内容
                    # 去除排除列表中的图片
                    available_images = [img for img in images if img not in exclude_list]
                    sampled_images = random.sample(available_images, min(len(available_images), sample_size))
                else:
                    # 训练集
                    sampled_images = random.sample(images, min(len(images), sample_size))

                # 添加到待处理的总列表中
                for image in sampled_images:
                    all_images.append((category, image, category_path))

        # 进度条显示处理
        for category, image, category_path in tqdm(all_images, desc="处理图片", unit="图片", ncols=100):
            # 直接将所有图片保存到train文件夹下，不区分类别
            src_path = os.path.join(category_path, image)
            dst_path = os.path.join(output_folder, image)

            # 处理文件名冲突情况（若存在同名图片时）
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(image)
                counter = 1
                new_dst_path = dst_path
                while os.path.exists(new_dst_path):
                    new_dst_path = os.path.join(output_folder, f"{name}_{counter}{ext}")
                    counter += 1
                dst_path = new_dst_path

            # 复制图片到目标文件夹
            shutil.copy(src_path, dst_path)

        print(f"图片抽取完成，已保存到文件夹: {output_folder}")


def generate_dataset_dota(sample_size=700, is_train=False):
    """
    从 DOTA 数据集中随机抽取指定数量的图片，并保存到目标文件夹。

    Args:
        opt (dict): 配置字典，包含以下键值：
            - source_folder (str): 源数据集文件夹路径（例如：'DOTA/Test/images'）。
            - output_folder (str): 输出文件夹路径（例如：'DOTA/Selected'）。
        sample_size (int): 随机抽取的图片数量，默认为 300。
    """
    # 如果目标文件夹不存在，创建它
    dataset = 'DOTA'

    if is_train:
        source_folder = '../../datasets/origin/' + dataset + '/Train/images'
        output_folder = '../../datasets/processed/' + dataset + '/train_HR'
    else:
        # source_folder = '../../datasets/origin/' + dataset + '/Test/images'
        source_folder = '../../datasets/processed/' + dataset + '/test_crop'
        output_folder = '../../datasets/processed/' + dataset + '/test_HR'

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图片文件
    images = [img for img in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, img))]

    # 确保图片数量足够
    if len(images) < sample_size:
        raise ValueError(f"源文件夹中的图片数量不足，只有 {len(images)} 张，但需要抽取 {sample_size} 张。")

    # 随机抽取指定数量的图片
    sampled_images = random.sample(images, sample_size)

    # 复制图片到目标文件夹
    for image in tqdm(sampled_images, desc="处理图片", unit="图片", ncols=100):
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(output_folder, image)
        shutil.copy(src_path, dst_path)

    print(f"图片抽取完成，共抽取 {sample_size} 张图片，已保存到文件夹: {output_folder}")


def generate_dataset_dior(sample_size=1000, is_train=False):
    """
    从 DIOR 数据集中随机抽取指定数量的图片，并保存到目标文件夹。

    Args:
        opt (dict): 配置字典，包含以下键值：
            - source_folder (str): 源数据集文件夹路径（例如：'DOTA/Test/images'）。
            - output_folder (str): 输出文件夹路径（例如：'DOTA/Selected'）。
        sample_size (int): 随机抽取的图片数量，默认为 300。
    """
    # 如果目标文件夹不存在，创建它
    dataset = 'DIOR'

    if is_train:
        source_folder = '../../datasets/origin/' + dataset + '/Train/JPEGImages-trainval'
        output_folder = '../../datasets/processed/' + dataset + '/train_HR'
    else:
        source_folder = '../../datasets/origin/' + dataset + '/Test/JPEGImages-test'
        output_folder = '../../datasets/processed/' + dataset + '/test_HR'

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有图片文件
    images = [img for img in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, img))]

    # 确保图片数量足够
    if len(images) < sample_size:
        raise ValueError(f"源文件夹中的图片数量不足，只有 {len(images)} 张，但需要抽取 {sample_size} 张。")

    # 随机抽取指定数量的图片
    sampled_images = random.sample(images, sample_size)

    # 复制图片到目标文件夹
    for image in tqdm(sampled_images, desc="处理图片", unit="图片", ncols=100):
        src_path = os.path.join(source_folder, image)
        dst_path = os.path.join(output_folder, image)
        shutil.copy(src_path, dst_path)

    print(f"图片抽取完成，共抽取 {sample_size} 张图片，已保存到文件夹: {output_folder}")


if __name__ == '__main__':
    PRE_DIR = '../../'
    DATASET = 'AID'
    train_folder = PRE_DIR + 'datasets/processed/' + DATASET + '/train_HR'
    test_folder = PRE_DIR + 'datasets/processed/' + DATASET + '/test_HR'

    # # 创建AID训练数据集，每个类别随机抽100张
    # generate_dataset_aid(sample_size=100)
    # # 排除训练集内图片
    # train_exclude_list = exclude_train_file(train_folder)
    # # 创建AID测试数据集，每个类别随机抽10张与训练集不重复的
    # generate_dataset_aid(is_train=False, sample_size=10, exclude_list=train_exclude_list)
    # 排除训练集和测试集内图片
    train_and_test_exclude_list = exclude_train_and_test_file(train_folder=train_folder, test_folder=test_folder)
    # 创建验证集，每个类别随机抽5张不重复的
    generate_dataset_aid(is_train=False, is_val=True, sample_size=5, exclude_list=train_and_test_exclude_list)

    # generate_dataset_dota()
    # generate_dataset_dior()


