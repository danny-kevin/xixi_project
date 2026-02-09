# 美国COVID-19数据使用说明

## 📊 数据概况

**文件名**: `data/raw/dataset_US_final.csv`

**数据列** (7列):
| 列名 | 说明 | 类型 |
|------|------|------|
| Date | 日期 | 时间序列 |
| Confirmed | 累计确诊病例数 | 目标变量 |
| Deaths | 死亡病例数 | 输入特征 |
| Stringency | 政策严格程度指数 | 输入特征 |
| Mobility_Work | 工作场所流动性 | 输入特征 |
| Mobility_Transit | 交通流动性 | 输入特征 |
| Mobility_Home | 居家流动性 | 输入特征 |

**数据规模**: 976行 (2020-02-15 到 2022-04-23)

**输入特征数**: 5个 (除Confirmed外的其他列)

---

## 🚀 快速开始

### 1. 测试数据加载

```bash
python load_us_data.py
```

这将:
- ✓ 加载CSV文件
- ✓ 检查缺失值并处理
- ✓ 归一化数据
- ✓ 创建时间窗口
- ✓ 划分训练/验证/测试集
- ✓ 创建PyTorch DataLoaders
- ✓ 显示数据统计信息

### 2. 使用专用配置文件训练

```bash
python main.py --mode train --config configs/default_config.yaml
```

---

## 📐 模型配置调整

由于实际数据只有**5个输入特征**，已对模型配置进行了以下调整:

### `configs/default_config.yaml`

```yaml
model:
  num_variables: 5          # 5个输入特征
  attention_num_heads: 4    # 降低注意力头数（从8→4）
  
data:
  target_column: "Confirmed"
  feature_columns:
    - Deaths
    - Stringency
    - Mobility_Work
    - Mobility_Transit
    - Mobility_Home
```

---

## 💡 代码示例

### 方式1: 使用专用数据加载器

```python
from load_us_data import USCovidDataLoader

# 创建加载器
loader = USCovidDataLoader()

# 准备数据
data_dict = loader.prepare_data(
    target_column='Confirmed',
    window_size=21,     # 使用过去21天预测
    horizon=7,          # 预测未来7天
    train_ratio=0.7,
    val_ratio=0.15
)

# 创建DataLoaders
train_loader, val_loader, test_loader = loader.create_dataloaders(
    data_dict,
    batch_size=32
)
```

### 方式2: 使用原有数据管道

```python
from pathlib import Path
import pandas as pd
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import EpidemicDataset

# 1. 加载数据
df = pd.read_csv("data/raw/dataset_US_final.csv", parse_dates=['Date'])
df = df.set_index('Date').sort_index()

# 2. 分离特征和目标
target = df[['Confirmed']]
features = df.drop('Confirmed', axis=1)

# 3. 预处理
preprocessor = DataPreprocessor()
features_norm = preprocessor.normalize(features)
target_norm = preprocessor.normalize(target)

# 4. 合并并创建窗口
data = pd.concat([target_norm, features_norm], axis=1).values
X, y = preprocessor.create_time_windows(data, window_size=21, horizon=7)

# 5. 创建数据集
dataset = EpidemicDataset(X, y)
```

---

## ⚙️ 关键配置参数

### 数据参数
- `window_size: 21` - 使用过去21天的数据
- `prediction_horizon: 7` - 预测未来7天
- `train_ratio: 0.7` - 70%训练集
- `val_ratio: 0.15` - 15%验证集
- `test_ratio: 0.15` - 15%测试集

### 模型参数
- `num_variables: 5` - 5个输入变量
- `input_size: 1` - 每个变量1维
- `tcn_channels: [32, 64, 64]` - TCN通道数
- `lstm_hidden_size: 128` - LSTM隐藏层大小
- `attention_num_heads: 4` - 4个注意力头

---

## 📝 注意事项

1. **目标变量选择**:
   - 当前配置: 预测 `Confirmed` (累计确诊)
   - 可选: 改为预测 `Deaths` (死亡病例)

2. **数据预处理**:
   - 缺失值: 使用前向/后向填充
   - 归一化: StandardScaler (zero mean, unit variance)
   - 时序划分: 严格按时间顺序，避免数据泄漏

3. **输入形状**:
   - X: `(samples, window_size=21, num_features=6)`
   - y: `(samples, horizon=7)` 或 `(samples,)`

---

## 🔧 定制修改

### 更改预测目标为Deaths

修改 `configs/default_config.yaml`:

```yaml
data:
  target_column: "Deaths"  # 改为预测死亡数
  feature_columns:
    - Confirmed            # 将Confirmed改为输入特征
    - Stringency
    - Mobility_Work
    - Mobility_Transit
    - Mobility_Home

model:
  num_variables: 5         # 保持不变
```

### 调整时间窗口

```yaml
data:
  window_size: 14          # 使用过去14天
  prediction_horizon: 3    # 预测未来3天
```

---

## 📊 数据统计

运行以下命令查看数据统计:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/dataset_US_final.csv', parse_dates=['Date'])
print(df.describe())
print('\n缺失值统计:')
print(df.isnull().sum())
"
```

---

## ✅ 验证清单

使用数据前请确认:

- [ ] CSV文件存在: `data/raw/dataset_US_final.csv`
- [ ] 配置文件正确: `configs/default_config.yaml`
- [ ] 数据加载器可运行: `python load_us_data.py`
- [ ] 目标变量已确认: `Confirmed` 或 `Deaths`
- [ ] 特征数量匹配: 模型 `num_variables = 5`

---

## 🆘 常见问题

**Q: 如何查看数据内容?**
```bash
python -c "import pandas as pd; print(pd.read_csv('data/raw/dataset_US_final.csv').head(10))"
```

**Q: 模型报错 shape mismatch?**
- 检查 `num_variables` 是否为 5
- 检查 `feature_columns` 是否列出了5个特征

**Q: 如何使用GPU训练?**
```yaml
training:
  device: "cuda"  # 或 "cpu"
```

---

生成时间: 2026-01-05
配置版本: v1.0
