# 数据准备模块使用指南

## 📋 概述

数据准备模块提供了完整的端到端数据处理流程，包括：
- 多源异构数据加载
- 数据清洗和预处理
- 时间序列窗口构建
- PyTorch Dataset 和 DataLoader 创建

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy torch scikit-learn openpyxl
```

### 2. 准备数据

#### 方式1: 使用示例数据

```bash
# 生成示例数据用于测试
python scripts/generate_sample_data.py
```

这将在 `data/raw/` 目录下生成以下文件：
- `epidemic.csv` - 疫情数据
- `mobility.csv` - 人口流动数据
- `environmental.csv` - 环境数据
- `intervention.csv` - 干预政策数据

#### 方式2: 使用真实数据

将您的数据文件放在 `data/raw/` 目录下，确保：
- 所有文件都包含 `date` 列（日期格式）
- 疫情数据至少包含 `new_cases` 列
- 文件格式为 CSV 或 Excel

### 3. 使用数据管道

#### 最简单的方式 - 一键准备数据

```python
from src.data import prepare_data

# 一键完成所有数据准备工作
train_loader, val_loader, test_loader, preprocessor = prepare_data(
    data_dir='data',
    window_size=21,      # 输入窗口：21天
    horizon=7,           # 预测范围：7天
    batch_size=32,
    num_workers=0        # Windows上设置为0
)

# 现在可以直接用于训练
for batch_x, batch_y in train_loader:
    # batch_x: (batch_size, 21, num_features)
    # batch_y: (batch_size, 7)
    print(f"输入形状: {batch_x.shape}, 目标形状: {batch_y.shape}")
    break
```

#### 更灵活的方式 - 使用 DataPipeline

```python
from src.data import DataPipeline

# 创建数据管道
pipeline = DataPipeline(
    data_dir='data',
    window_size=21,
    horizon=7,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=32,
    num_workers=0
)

# 运行完整流程
train_loader, val_loader, test_loader = pipeline.run()

# 获取预处理器（用于反归一化）
preprocessor = pipeline.get_preprocessor()

# 获取特征名称
feature_names = pipeline.get_feature_names()
print(f"特征列表: {feature_names}")
```

#### 分步骤使用 - 完全自定义

```python
from src.data import DataLoader, DataPreprocessor, EpidemicDataset, create_data_loaders

# 1. 加载数据
loader = DataLoader('data')
data_dict = loader.load_all_data()
merged_data = loader.merge_data_sources(data_dict)

# 2. 预处理
preprocessor = DataPreprocessor()

# 处理缺失值
cleaned_data = preprocessor.handle_missing_values(merged_data, method='interpolate')

# 检测异常值
outliers = preprocessor.detect_outliers(cleaned_data, method='iqr', threshold=1.5)

# 归一化
normalized_data = preprocessor.normalize(cleaned_data, method='minmax')

# 3. 创建时间窗口
data_array = normalized_data.values
X, y = preprocessor.create_time_windows(
    data_array,
    window_size=21,
    horizon=7,
    stride=1
)

# 4. 划分数据集
n_samples = len(X)
train_end = int(n_samples * 0.7)
val_end = int(n_samples * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 5. 创建 Dataset
train_dataset = EpidemicDataset(X_train, y_train)
val_dataset = EpidemicDataset(X_val, y_val)
test_dataset = EpidemicDataset(X_test, y_test)

# 6. 创建 DataLoader
train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset, val_dataset, test_dataset,
    batch_size=32,
    num_workers=0
)
```

## 📊 数据格式说明

### 输入数据格式

每个数据文件应包含以下列：

#### 疫情数据 (epidemic.csv)
```
date,new_cases,new_deaths,new_recovered,cumulative_cases,cumulative_deaths,active_cases
2020-01-01,100,2,80,100,2,18
2020-01-02,120,3,90,220,5,45
...
```

#### 人口流动数据 (mobility.csv)
```
date,intra_city_flow,inter_city_flow,public_transport,retail_mobility,workplace_mobility
2020-01-01,100,80,90,70,85
2020-01-02,95,75,85,65,80
...
```

#### 环境数据 (environmental.csv)
```
date,temperature,humidity,uv_index,precipitation,wind_speed
2020-01-01,15.5,60,5,0,3.2
2020-01-02,16.2,58,6,0,2.8
...
```

#### 干预政策数据 (intervention.csv)
```
date,lockdown_level,social_distance,mask_mandate,vaccination_rate,testing_rate
2020-01-01,0,0,0,0,5.2
2020-01-02,0,0,0,0,5.5
...
```

### 输出数据格式

```python
# DataLoader 输出
for batch_x, batch_y in train_loader:
    # batch_x: torch.Tensor, shape=(batch_size, window_size, num_features)
    #   - batch_size: 批次大小（如32）
    #   - window_size: 输入窗口大小（如21天）
    #   - num_features: 特征数量（所有数据源的特征总和）
    
    # batch_y: torch.Tensor, shape=(batch_size, horizon) 或 (batch_size,)
    #   - horizon: 预测范围（如7天）
    #   - 目标变量通常是 new_cases
    pass
```

## 🔧 高级功能

### 1. 自定义缺失值处理

```python
preprocessor = DataPreprocessor()

# 方法1: 时间序列插值（推荐）
df = preprocessor.handle_missing_values(df, method='interpolate')

# 方法2: 前向填充
df = preprocessor.handle_missing_values(df, method='ffill')

# 方法3: 后向填充
df = preprocessor.handle_missing_values(df, method='bfill')

# 方法4: 均值填充
df = preprocessor.handle_missing_values(df, method='mean')
```

### 2. 异常值检测

```python
# IQR 方法（四分位距）
outliers = preprocessor.detect_outliers(df, method='iqr', threshold=1.5)

# Z-score 方法
outliers = preprocessor.detect_outliers(df, method='zscore', threshold=3.0)

# 查看异常值
print(f"异常值数量: {outliers.sum().sum()}")
```

### 3. 数据归一化

```python
# MinMax 归一化 [0, 1]
df_normalized = preprocessor.normalize(df, method='minmax')

# 标准化 (均值=0, 标准差=1)
df_normalized = preprocessor.normalize(df, method='standard')

# 指定特定列进行归一化
df_normalized = preprocessor.normalize(
    df, 
    method='minmax',
    columns=['new_cases', 'temperature']
)
```

### 4. 反归一化

```python
# 训练后，将预测结果转换回原始尺度
predictions_normalized = model(input_data)  # 归一化的预测结果

# 反归一化
predictions_original = preprocessor.inverse_transform(
    predictions_normalized.cpu().numpy(),
    column='epidemic_new_cases'  # 使用合并后的列名
)
```

### 5. 自定义时间窗口

```python
# 创建不同大小的窗口
X_14, y_14 = preprocessor.create_time_windows(
    data, 
    window_size=14,   # 14天输入
    horizon=3,        # 3天预测
    stride=1          # 每次滑动1天
)

# 使用更大的步长（减少样本数量）
X_sparse, y_sparse = preprocessor.create_time_windows(
    data,
    window_size=21,
    horizon=7,
    stride=7          # 每次滑动7天
)
```

## 🧪 测试和验证

### 运行完整测试

```bash
# 测试所有组件
python scripts/test_data_pipeline.py
```

### 验证数据形状

```python
from src.data import prepare_data

train_loader, val_loader, test_loader, preprocessor = prepare_data(
    data_dir='data',
    window_size=21,
    horizon=7,
    batch_size=32
)

# 检查数据形状
for batch_x, batch_y in train_loader:
    print(f"✓ 输入形状: {batch_x.shape}")  # 应该是 (32, 21, num_features)
    print(f"✓ 目标形状: {batch_y.shape}")  # 应该是 (32, 7) 或 (32,)
    print(f"✓ 数据类型: {batch_x.dtype}")  # 应该是 torch.float32
    break
```

## ⚠️ 常见问题

### 1. ModuleNotFoundError: No module named 'torch'

**解决方案**: 安装 PyTorch
```bash
pip install torch
```

### 2. 数据文件未找到

**解决方案**: 确保数据文件在正确的位置
```bash
# 检查文件是否存在
ls data/raw/

# 如果没有，生成示例数据
python scripts/generate_sample_data.py
```

### 3. 日期解析错误

**解决方案**: 确保日期列名为 `date`，格式为标准日期格式
```python
# 如果日期格式特殊，可以手动指定
df = pd.read_csv('data.csv', parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format='%Y/%m/%d'))
```

### 4. 内存不足

**解决方案**: 减小批次大小或使用更少的特征
```python
train_loader, val_loader, test_loader, preprocessor = prepare_data(
    data_dir='data',
    batch_size=16,    # 减小批次大小
    num_workers=0     # 减少工作进程
)
```

### 5. Windows 上 DataLoader 报错

**解决方案**: 设置 `num_workers=0`
```python
train_loader, val_loader, test_loader, preprocessor = prepare_data(
    data_dir='data',
    num_workers=0  # Windows 上必须设置为 0
)
```

## 📚 API 参考

### DataLoader

```python
DataLoader(data_dir: str)
```
- `load_epidemic_data(filename)` - 加载疫情数据
- `load_mobility_data(filename)` - 加载人口流动数据
- `load_environmental_data(filename)` - 加载环境数据
- `load_intervention_data(filename)` - 加载干预政策数据
- `load_all_data()` - 加载所有数据
- `merge_data_sources(data_dict)` - 合并多源数据

### DataPreprocessor

```python
DataPreprocessor(config: Optional[Dict] = None)
```
- `handle_missing_values(df, method)` - 处理缺失值
- `detect_outliers(df, method, threshold)` - 检测异常值
- `normalize(df, method, columns)` - 数据归一化
- `create_time_windows(data, window_size, horizon, stride)` - 创建时间窗口
- `temporal_train_test_split(data, train_ratio, val_ratio)` - 时序数据划分
- `inverse_transform(data, column)` - 反归一化

### EpidemicDataset

```python
EpidemicDataset(X, y, feature_names=None, transform=None)
```
- `__len__()` - 返回数据集大小
- `__getitem__(idx)` - 获取单个样本
- `get_feature_dim()` - 返回特征维度
- `get_window_size()` - 返回时间窗口大小

### DataPipeline

```python
DataPipeline(
    data_dir,
    window_size=21,
    horizon=7,
    train_ratio=0.7,
    val_ratio=0.15,
    batch_size=32,
    num_workers=4
)
```
- `load_data()` - 加载数据
- `preprocess_data(df, handle_missing, detect_outliers, normalize)` - 预处理数据
- `create_datasets(df)` - 创建数据集
- `create_dataloaders(train_dataset, val_dataset, test_dataset)` - 创建DataLoader
- `run()` - 运行完整管道
- `get_preprocessor()` - 获取预处理器
- `get_feature_names()` - 获取特征名称

## 🎯 最佳实践

1. **始终使用时序划分**: 避免数据泄露
2. **保存预处理器**: 用于预测时的数据转换
3. **验证数据形状**: 确保与模型输入匹配
4. **处理缺失值**: 在归一化之前
5. **记录预处理步骤**: 便于复现和调试

## 📞 支持

如有问题，请查看：
- `Docs/data_flow.md` - 数据流文档
- `scripts/test_data_pipeline.py` - 测试示例
- `agents/01_data_preparation_agent.md` - Agent 文档
