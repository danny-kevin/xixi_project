# 项目运行流程指南

## 🚀 快速开始

### 前提条件

1. **环境准备**
```bash
# 创建虚拟环境
conda create -n xixi_project python=3.10
conda activate xixi_project

# 安装依赖
pip install -r requirements.txt
```

2. **数据准备**
- 将原始数据放入 `data/raw/` 目录
- 确保数据格式符合要求（见下文）

---

## 📋 完整运行流程

### 阶段0: 验证环境

```bash
# 测试PyTorch安装
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# 测试CUDA（如果有GPU）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 测试项目导入
python -c "from src.utils import setup_logger; print('✅ 项目环境正常')"
```

---

### 阶段1: 数据准备 (需要01_data_preparation_agent完成)

#### 1.1 准备原始数据

将以下数据文件放入 `data/raw/`:
- `epidemic_data.csv` - 疫情数据
- `mobility_data.csv` - 人口流动数据
- `environment_data.csv` - 环境数据
- `intervention_data.csv` - 干预政策数据

#### 1.2 数据格式要求

**疫情数据** (`epidemic_data.csv`):
```csv
date,new_cases,new_deaths,new_recovered
2020-01-01,100,5,80
2020-01-02,120,6,95
...
```

**人口流动数据** (`mobility_data.csv`):
```csv
date,mobility_index,transport_flow
2020-01-01,0.8,1000
2020-01-02,0.75,950
...
```

**环境数据** (`environment_data.csv`):
```csv
date,temperature,humidity,uv_index
2020-01-01,25.5,60,5
2020-01-02,26.0,58,6
...
```

**干预政策数据** (`intervention_data.csv`):
```csv
date,lockdown_level,social_distance_policy,vaccination_rate
2020-01-01,0,0,0
2020-01-02,1,0.5,0
...
```

#### 1.3 运行数据预处理

```bash
# 方式1: 使用Python脚本
python -c "
from src.data import DataLoader, DataPreprocessor
from src.utils.config import load_config

config = load_config('configs/default_config.yaml')
loader = DataLoader(config.data.data_dir)
preprocessor = DataPreprocessor(config.data)

# 加载数据
raw_data = loader.load_all_data()
print(f'✅ 加载了 {len(raw_data)} 类数据')

# 预处理
processed_data = preprocessor.preprocess(raw_data)
print(f'✅ 预处理完成，数据形状: {processed_data.shape}')
"

# 方式2: 使用提供的脚本（如果有）
# python scripts/prepare_data.py --config configs/default_config.yaml
```

---

### 阶段2: 模型训练 (需要02-05_agent完成)

#### 2.1 快速训练（测试用）

```bash
# 训练1个epoch，快速验证流程
python train.py \
    --config configs/default_config.yaml \
    --epochs 1 \
    --batch-size 16
```

预期输出:
```
2025-12-29 17:00:00 - INFO - 开始训练...
2025-12-29 17:00:01 - INFO - Epoch 1/1
2025-12-29 17:00:05 - INFO - Train Loss: 0.1234
2025-12-29 17:00:06 - INFO - Val Loss: 0.1456
2025-12-29 17:00:06 - INFO - ✅ 训练完成
```

#### 2.2 完整训练

```bash
# 使用默认配置训练100个epoch
python train.py --config configs/default_config.yaml
```

#### 2.3 自定义训练参数

```bash
python train.py \
    --config configs/default_config.yaml \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --device cuda \
    --checkpoint-dir checkpoints/exp1
```

#### 2.4 使用多阶段训练

```bash
# 启用预训练 + 微调
python train.py \
    --config configs/default_config.yaml \
    --use-pretrain \
    --pretrain-epochs 20 \
    --finetune-epochs 80
```

#### 2.5 断点续训

```bash
# 从检查点恢复训练
python train.py \
    --config configs/default_config.yaml \
    --resume checkpoints/best_model.pth
```

---

### 阶段3: 模型评估 (需要06_evaluation_interpretation_agent完成)

#### 3.1 评估训练好的模型

```bash
python -c "
from src.models import AttentionMTCNLSTM
from src.evaluation import ModelEvaluator
from src.utils.checkpoint import load_checkpoint
import torch

# 加载模型
model = AttentionMTCNLSTM(...)
load_checkpoint(model, 'checkpoints/best_model.pth')

# 评估
evaluator = ModelEvaluator(model)
results = evaluator.evaluate(test_loader)

print('评估结果:')
print(f'  MSE: {results[\"mse\"]:.4f}')
print(f'  RMSE: {results[\"rmse\"]:.4f}')
print(f'  MAE: {results[\"mae\"]:.4f}')
"
```

#### 3.2 生成评估报告

```bash
# 生成完整的评估报告
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --test-data data/processed/test.npz \
    --output results/evaluation_report.txt
```

---

### 阶段4: 可解释性分析 (需要06_evaluation_interpretation_agent完成)

#### 4.1 注意力权重可视化

```bash
python -c "
from src.evaluation import AttentionVisualizer
from src.utils.checkpoint import load_checkpoint

# 加载模型
model = load_checkpoint('checkpoints/best_model.pth')

# 可视化注意力
visualizer = AttentionVisualizer(model)
attention_weights = visualizer.extract_attention_weights(sample_input)

# 绘制热力图
visualizer.plot_temporal_attention(
    attention_weights,
    save_path='results/figures/attention_heatmap.png'
)
"
```

#### 4.2 特征重要性分析

```bash
python -c "
from src.evaluation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model)
importance = analyzer.permutation_importance(X_test, y_test, feature_names)

# 绘制重要性图
analyzer.plot_feature_importance(
    importance,
    save_path='results/figures/feature_importance.png'
)
"
```

---

### 阶段5: 完整实验流程

#### 5.1 运行完整实验

```bash
# 一键运行完整流程：数据准备 → 训练 → 评估 → 可解释性分析
python run_experiment.py --config configs/default_config.yaml
```

这个脚本会：
1. ✅ 加载和预处理数据
2. ✅ 创建和训练模型
3. ✅ 评估模型性能
4. ✅ 生成可解释性分析
5. ✅ 保存所有结果到 `results/` 目录

#### 5.2 使用Jupyter Notebook

```bash
# 启动Jupyter
jupyter notebook

# 打开 notebooks/01_quick_start.ipynb
# 按照notebook中的步骤逐步执行
```

---

## 🔧 常见问题排查

### 问题1: 导入错误

```
ImportError: cannot import name 'DataLoader' from 'src.data'
```

**原因**: 对应的Agent还未实现该模块

**解决**: 
1. 检查 `Docs/completion_summary.md` 确认哪些模块已实现
2. 等待对应Agent完成实现
3. 或使用模拟数据测试框架

### 问题2: CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决**:
```bash
# 减小batch size
python train.py --batch-size 16

# 或使用CPU
python train.py --device cpu
```

### 问题3: 数据格式错误

```
ValueError: Data shape mismatch
```

**解决**:
1. 检查数据格式是否符合要求
2. 运行形状验证:
```python
from src.utils.shape_validator import ShapeValidator
validator = ShapeValidator()
validator.validate_data(X, y)
```

### 问题4: 配置文件错误

```
KeyError: 'num_variables'
```

**解决**:
```bash
# 使用默认配置
cp configs/default_config.yaml configs/my_config.yaml

# 编辑配置文件
# 确保所有必需字段都存在
```

---

## 📊 监控训练过程

### 使用TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 在浏览器打开 http://localhost:6006
```

### 使用Weights & Biases (可选)

```bash
# 登录WandB
wandb login

# 训练时会自动上传到WandB
python train.py --config configs/default_config.yaml --use-wandb
```

---

## 📁 输出文件说明

训练完成后，会生成以下文件：

```
checkpoints/
├── best_model.pth          # 最佳模型（验证集loss最低）
├── last_model.pth          # 最后一个epoch的模型
└── checkpoint_epoch_50.pth # 中间检查点

logs/
├── train.log               # 训练日志
└── tensorboard/            # TensorBoard日志

results/
├── evaluation_report.txt   # 评估报告
├── config.yaml             # 使用的配置
├── predictions.npy         # 预测结果
└── figures/                # 可视化图表
    ├── training_history.png
    ├── predictions.png
    ├── attention_heatmap.png
    └── feature_importance.png
```

---

## 🎯 Agent实现检查清单

在运行完整流程前，确保以下Agent已完成：

### 必需（核心功能）
- [ ] 01_data_preparation_agent
  - [ ] DataLoader
  - [ ] DataPreprocessor
  - [ ] EpidemicDataset

- [ ] 02_mtcn_module_agent
  - [ ] TCN
  - [ ] MTCN

- [ ] 03_attention_mechanism_agent
  - [ ] SelfAttention
  - [ ] VariableAttention

- [ ] 04_lstm_module_agent
  - [ ] BiLSTMModule

- [ ] 05_model_integration_agent
  - [ ] AttentionMTCNLSTM
  - [ ] Trainer
  - [ ] Loss functions

### 可选（增强功能）
- [ ] 06_evaluation_interpretation_agent
  - [ ] Metrics
  - [ ] AttentionVisualizer
  - [ ] FeatureImportanceAnalyzer

---

## 🚦 验证流程

### 步骤1: 验证数据模块
```bash
python -c "from src.data import DataLoader, DataPreprocessor, EpidemicDataset; print('✅ 数据模块OK')"
```

### 步骤2: 验证模型模块
```bash
python -c "from src.models import AttentionMTCNLSTM; print('✅ 模型模块OK')"
```

### 步骤3: 验证训练模块
```bash
python -c "from src.training import Trainer; print('✅ 训练模块OK')"
```

### 步骤4: 验证评估模块
```bash
python -c "from src.evaluation import ModelEvaluator; print('✅ 评估模块OK')"
```

### 步骤5: 端到端测试
```bash
python train.py --epochs 1 --batch-size 8
```

---

## 📚 进一步学习

1. **修改模型架构**: 编辑 `configs/default_config.yaml`
2. **添加新的损失函数**: 在 `src/training/loss.py` 中实现
3. **自定义评估指标**: 在 `src/evaluation/metrics.py` 中添加
4. **实验不同配置**: 复制配置文件并修改参数

---

## 💡 最佳实践

1. **先小规模测试**: 用少量数据和少量epoch验证流程
2. **使用版本控制**: 记录每次实验的配置和结果
3. **定期保存检查点**: 避免训练中断导致损失
4. **监控资源使用**: 注意GPU内存和训练时间
5. **记录实验日志**: 便于后续分析和复现

---

## 🎉 完成标志

当你看到以下输出时，说明一切正常：

```
2025-12-29 17:00:00 - INFO - ✅ 数据加载完成
2025-12-29 17:00:01 - INFO - ✅ 模型创建完成
2025-12-29 17:00:02 - INFO - ✅ 开始训练...
...
2025-12-29 18:00:00 - INFO - ✅ 训练完成
2025-12-29 18:00:01 - INFO - ✅ 评估完成
2025-12-29 18:00:02 - INFO - ✅ 结果已保存到 results/
```

祝你训练顺利！🚀
