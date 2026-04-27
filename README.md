# DataPipeline

一个灵活、模块化的数据增强框架，使用像 **DDPM**（去噪扩散概率模型）这样的生成模型。旨在支持表格数据和图像处理流，包含小样本微调功能，以使模型适应目标分布。

## 特性

- **DDPM 核心**: 去噪扩散概率模型的模块化实现。
- **数据适配器**: 在 `TabularAdapter`（表格适配器）和 `ImageAdapter`（图像适配器）之间轻松切换。
- **小样本微调**: 使用极少量数据使预训练生成模型适应新分布。
- **统一流**: 针对不同数据模态的一致 API (`augment`, `finetune`)。

## 项目结构

```text
/pipeline
├── /adapters    # 数据准备逻辑 (归一化等)
├── /models      # 模型定义 (DDPM, 网络结构)
├── core.py      # 主要的数据增强流 (AugmentationPipeline) 逻辑
└── main.py      # 测试和使用的入口点
```

## 快速开始

### 前置条件
- Python 3.14+
- `torch`, `numpy`, `fsspec`, `jinja2`, `networkx`, `sympy` (见 `requirements.txt`)

### 运行测试
执行提供的测试套件以验证安装：

```bash
python main.py
```

## 使用示例

```python
from pipeline import AugmentationPipeline, TabularAdapter, DDPM

# 设置 Pipeline
adapter = TabularAdapter(normalize=True)
model = DDPM(data_shape=(5,), timesteps=50) 
pipeline = AugmentationPipeline(adapter=adapter, model=model)

# 在目标数据上微调
pipeline.finetune(target_data, epochs=10)

# 增强
augmented_data = pipeline.augment(seed_data, num_samples=5)
```

## 许可证
MIT
