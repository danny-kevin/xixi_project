# 注意力增强M-TCN-LSTM混合神经网络

## 📋 项目简介

本项目实现了一种**注意力增强的M-TCN-LSTM混合神经网络模型**，用于传染病疫情预测。该模型融合了时间卷积网络(TCN)的局部特征提取能力与长短期记忆网络(LSTM)的序列建模优势，并通过注意力机制增强模型的预测精度和可解释性。

### 🎯 核心创新点

1. **混合架构设计**: 融合M-TCN的局部特征提取与LSTM的序列建模
2. **注意力机制增强**: 变量间注意力机制 + 随机注意力正则化
3. **传染病特性优化**: 针对滞后效应和非周期性冲击的专门优化
4. **可解释性框架**: 多级解释框架支持决策透明

---

## 🏗️ 模型架构

```
输入层 (多变量时间序列窗口)
    ↓
M-TCN模块 (并行TCN子网络处理各变量)
    ↓
特征拼接层
    ↓
注意力增强层 (Self-Attention + Dropout正则化)
    ↓
双层双向LSTM模块
    ↓
全连接输出层
    ↓
预测结果 (未来7/14天病例数)
```

---

## 📁 项目结构

```
xixi_project/
├── data/
│   ├── raw/                    # 原始数据
│   ├── processed/              # 预处理后数据
│   └── external/               # 外部数据源
├── src/
│   ├── data/                   # 数据处理模块
│   │   ├── data_loader.py      # 数据加载器
│   │   ├── preprocessor.py     # 预处理模块
│   │   └── dataset.py          # PyTorch Dataset类
│   ├── models/                 # 模型模块
│   │   ├── tcn.py              # TCN子网络
│   │   ├── mtcn.py             # M-TCN模块
│   │   ├── attention.py        # 注意力机制
│   │   ├── lstm_module.py      # LSTM模块
│   │   └── hybrid_model.py     # 完整混合模型
│   ├── training/               # 训练模块
│   │   ├── trainer.py          # 训练器
│   │   ├── loss.py             # 损失函数
│   │   └── scheduler.py        # 学习率调度
│   ├── evaluation/             # 评估模块
│   │   ├── metrics.py          # 评估指标
│   │   └── interpretability.py # 可解释性分析
│   └── utils/                  # 工具模块
│       ├── config.py           # 配置管理
│       └── visualization.py    # 可视化工具
├── configs/
│   └── default_config.yaml     # 默认配置文件
├── notebooks/
│   └── experiments.ipynb       # 实验笔记本
├── tests/
│   └── test_models.py          # 单元测试
├── agents/                     # Agent prompts
├── Docs/                       # 文档
├── requirements.txt
└── README.md
```

---

## 🚀 快速开始

### 环境安装

```bash
# 创建虚拟环境
conda create -n xixi_project python=3.10
conda activate xixi_project

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```python
from src.data import DataLoader, DataPreprocessor, EpidemicDataset
from src.models import AttentionMTCNLSTM
from src.training import Trainer, TrainingConfig
from src.utils import load_config

# 加载配置
config = load_config("configs/default_config.yaml")

# 准备数据
# ... (由 01_data_preparation_agent 实现)

# 创建模型
model = AttentionMTCNLSTM(
    num_variables=config.model.num_variables,
    tcn_channels=config.model.tcn_channels,
    # ... 其他参数
)

# 训练
trainer = Trainer(model, TrainingConfig())
trainer.train(train_loader, val_loader)
```

---

## 📊 数据说明

本项目使用四类多源异构数据：

| 数据类型 | 具体内容 | 时间粒度 |
|---------|---------|---------|
| 疫情数据 | 每日新增确诊、死亡、康复病例 | 日度 |
| 人口流动数据 | 手机定位、交通枢纽人流指数 | 日度 |
| 环境数据 | 温度、湿度、紫外线强度 | 日度 |
| 干预政策数据 | 封城等级、社交距离、疫苗接种率 | 日度 |

---

## 🔧 技术栈

- **深度学习框架**: PyTorch
- **数据处理**: Pandas, NumPy, Scikit-learn
- **可视化**: Matplotlib, Seaborn, Plotly
- **可解释性**: SHAP, Captum
- **实验跟踪**: TensorBoard, Weights & Biases

---

## 📝 开发Agent列表

| Agent编号 | Agent名称 | 主要职责 | 状态 |
|-----------|----------|---------|------|
| 00 | 项目总览Agent | 项目协调与架构设计 | ✅ 已完成 |
| 01 | 数据准备Agent | 数据收集、清洗、预处理 | ⏳ 待实现 |
| 02 | M-TCN模块Agent | M-TCN架构设计与实现 | ⏳ 待实现 |
| 03 | 注意力机制Agent | 注意力层设计与优化 | ⏳ 待实现 |
| 04 | LSTM模块Agent | LSTM模块设计与实现 | ⏳ 待实现 |
| 05 | 模型整合Agent | 模型集成与训练流程 | ⏳ 待实现 |
| 06 | 评估解释Agent | 性能评估与可解释性分析 | ⏳ 待实现 |

---

## ⚠️ 关键技术要点

1. **时序因果性**: 确保模型不使用未来信息（防止信息泄露）
2. **感受野覆盖**: TCN扩张系数需覆盖14-21天的滞后周期
3. **注意力正则化**: 训练时随机丢弃注意力权重（概率0.1）
4. **多阶段训练**: 先预训练子模块，后端到端微调
5. **时序交叉验证**: 严格时序划分，避免未来信息泄露

---

## 📄 许可证

本项目仅供学术研究使用。

---

## 📧 联系方式

如有问题，请通过项目Issues提交。
