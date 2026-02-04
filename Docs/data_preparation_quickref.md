# 数据准备模块 - 快速参考

## 🚀 一键使用

```python
from src.data import prepare_data

train_loader, val_loader, test_loader, preprocessor = prepare_data(
    data_dir='data',
    window_size=21,  # 输入21天
    horizon=7,       # 预测7天
    batch_size=32
)
```

## 📦 主要组件

| 组件 | 用途 | 导入方式 |
|------|------|----------|
| `DataLoader` | 加载多源数据 | `from src.data import DataLoader` |
| `DataPreprocessor` | 数据预处理 | `from src.data import DataPreprocessor` |
| `EpidemicDataset` | PyTorch数据集 | `from src.data import EpidemicDataset` |
| `DataPipeline` | 完整管道 | `from src.data import DataPipeline` |
| `prepare_data` | 便捷函数 | `from src.data import prepare_data` |

## 📊 数据格式

### 输入
- 疫情数据: `epidemic.csv` (必须包含 `date`, `new_cases`)
- 人口流动: `mobility.csv`
- 环境数据: `environmental.csv`
- 干预政策: `intervention.csv`

### 输出
```python
batch_x: (batch_size, window_size, num_features)  # 如 (32, 21, 20)
batch_y: (batch_size, horizon)                     # 如 (32, 7)
```

## 🔧 常用操作

### 生成示例数据
```bash
python scripts/generate_sample_data.py
```

### 测试管道
```bash
python scripts/test_data_pipeline.py
```

### 快速验证
```bash
python -c "from src.data import prepare_data; print('OK')"
```

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `window_size` | 21 | 输入时间窗口（天） |
| `horizon` | 7 | 预测范围（天） |
| `train_ratio` | 0.7 | 训练集比例 |
| `val_ratio` | 0.15 | 验证集比例 |
| `batch_size` | 32 | 批次大小 |
| `num_workers` | 4 | 数据加载进程数（Windows设为0） |

## 🎯 关键特性

✅ 时序因果性保证（防止数据泄露）  
✅ 可逆归一化（支持结果还原）  
✅ 多种预处理方法  
✅ 一键式便捷函数  
✅ 详细的进度输出

## 📚 文档

- 📖 完整指南: `Docs/data_preparation_guide.md`
- 📋 完成总结: `Docs/01_data_preparation_completion.md`
- 🎯 Agent文档: `agents/01_data_preparation_agent.md`

## ⚠️ 注意

- Windows用户: 设置 `num_workers=0`
- 需要安装: `pandas`, `numpy`, `torch`, `scikit-learn`
- 数据文件必须包含 `date` 列
