# SRWorkflow

**SRWorkflow** 是一个基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 项目的自用图像超分辨率（SR）框架，经过简化和定制化设计。本项目为图像超分辨率任务提供了一个简单高效的工作流，支持多种 SR 模型和图像处理操作。

## 特性

- **简化设计**：对 BasicSR 项目进行部分简化，移除了不必要的组件，使框架轻量化并易于使用。
- **高效工作流**：为常见的图像超分辨率任务（如数据准备、模型训练和评估）提供了简洁流畅的工作流。
- **可定制配置**：支持通过配置文件快速调整训练参数、模型架构和优化策略。
- **模块化**：框架的各个组件（数据加载、模型定义、损失函数等）均模块化设计，可灵活组合以满足不同需求。
- **支持多种模型**：支持常见的超分辨率模型，如 **EDSR**，以及更多模型（**待完善...**）。
- **图像超分辨率训练**：可使用指定架构训练超分辨率模型。
- **图像超分辨率测试**：利用训练好的模型对低分辨率图像进行推理。
- **数据处理与预处理**：提供常见的数据操作，如数据集加载、裁剪、缩放等。

## 安装

待补充...

## 使用方法

### 数据准备

1. 需要下载训练所需数据集，AID、DOTA和DIOR。
2. 执行 `scripts/data_preparation/generate_dataset_file.py` 将从原始数据集中随机筛选数据组成训练集、测试集和验证集。
3. 执行 `scripts/matlab_scripts/generate_bicubic_img.m` 处理图片，生成倍数补全图和对应的下采样倍数图片。
4. 执行 `scripts/matlab_scripts/create_lmdb.py` 构建 lmdb 格式的训练集。

### 训练参数





## 贡献

这是一个**自用**框架，如果有任何问题或改进建议，请随时提交 Issue。

## 致谢

- 本项目基于 [BasicSR](https://github.com/xinntao/BasicSR) 项目开发。
- 感谢开源社区对本项目的支持与启发。

## 许可证

SRWorkflow 遵循 [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) 许可协议。更多详细信息请参阅 `LICENSE` 文件。