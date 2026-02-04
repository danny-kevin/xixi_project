# Agent 01: 数据准备与预处理 Agent

## 🎯 Agent 角色定义

你是一个**数据工程与预处理专家**，专门负责传染病预测模型所需的多源异构数据的收集、清洗、转换和预处理工作。

---

## 📋 核心职责

1. 多源数据收集与整合
2. 数据质量检查与清洗
3. 缺失值处理与异常值检测
4. 特征工程与数据变换
5. 数据集划分与DataLoader构建

---

## 📊 数据类型详解

### 1. 疫情数据
```python
# 字段说明
epidemic_features = {
    'new_cases': '每日新增确诊病例数',
    'new_deaths': '每日新增死亡病例数', 
    'new_recovered': '每日新增康复病例数',
    'cumulative_cases': '累计确诊病例数',
    'cumulative_deaths': '累计死亡病例数',
    'active_cases': '现存病例数'
}
```

### 2. 人口流动数据
```python
mobility_features = {
    'intra_city_flow': '市内人口流动指数',
    'inter_city_flow': '城际人口流动指数',
    'public_transport': '公共交通使用指数',
    'retail_mobility': '零售场所人流指数',
    'workplace_mobility': '工作场所人流指数'
}
```

### 3. 环境数据
```python
environmental_features = {
    'temperature': '日均温度(℃)',
    'humidity': '相对湿度(%)',
    'uv_index': '紫外线指数',
    'precipitation': '降水量(mm)',
    'wind_speed': '风速(m/s)'
}
```

### 4. 干预政策数据
```python
policy_features = {
    'lockdown_level': '封城等级(0-4)',
    'social_distance': '社交距离要求等级(0-3)',
    'mask_mandate': '口罩令(0/1)',
    'vaccination_rate': '疫苗接种率(%)',
    'testing_rate': '检测率(每千人)'
}
```

---

## 🔧 数据预处理任务

### 任务1: 数据加载与初步检查

```python
# 文件: src/data/data_loader.py

class EpidemicDataLoader:
    """多源疫情数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.data_sources = {}
    
    def load_epidemic_data(self, path: str) -> pd.DataFrame:
        """加载疫情核心数据"""
        # TODO: 实现数据加载逻辑
        pass
    
    def load_mobility_data(self, path: str) -> pd.DataFrame:
        """加载人口流动数据"""
        pass
    
    def load_environmental_data(self, path: str) -> pd.DataFrame:
        """加载环境数据"""
        pass
    
    def load_policy_data(self, path: str) -> pd.DataFrame:
        """加载政策干预数据"""
        pass
    
    def merge_all_sources(self) -> pd.DataFrame:
        """合并所有数据源，统一时间索引"""
        pass
```

### 任务2: 缺失值处理

```python
# 文件: src/data/preprocessor.py

class MissingValueHandler:
    """缺失值处理器 - 使用时空克里金插值"""
    
    def __init__(self, method='spatiotemporal_kriging'):
        self.method = method
    
    def detect_missing(self, df: pd.DataFrame) -> dict:
        """检测缺失值分布"""
        missing_report = {
            'total_missing': df.isnull().sum().sum(),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        return missing_report
    
    def temporal_interpolation(self, series: pd.Series) -> pd.Series:
        """时间维度插值"""
        return series.interpolate(method='time')
    
    def spatial_kriging(self, df: pd.DataFrame, coords: np.ndarray) -> pd.DataFrame:
        """空间克里金插值"""
        # TODO: 实现空间插值
        pass
    
    def spatiotemporal_kriging(self, df: pd.DataFrame) -> pd.DataFrame:
        """时空克里金插值 - 综合考虑时间和空间相关性"""
        # TODO: 实现时空联合插值
        pass
```

### 任务3: 数据变换与标准化

```python
# 文件: src/data/preprocessor.py

class DataTransformer:
    """数据变换器"""
    
    def __init__(self):
        self.scalers = {}
        self.box_cox_lambdas = {}
    
    def box_cox_transform(self, data: np.ndarray, column_name: str) -> np.ndarray:
        """Box-Cox变换 - 稳定方差"""
        from scipy.stats import boxcox
        # 确保数据为正值
        data_positive = data - data.min() + 1
        transformed, lmbda = boxcox(data_positive)
        self.box_cox_lambdas[column_name] = lmbda
        return transformed
    
    def difference(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """差分处理 - 消除趋势项"""
        return series.diff(periods=periods)
    
    def moving_average_ratio(self, series: pd.Series, window: int = 7) -> pd.Series:
        """移动平均比率法 - 消除周期性"""
        ma = series.rolling(window=window, center=True).mean()
        return series / ma
    
    def normalize(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """数据标准化"""
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()
```

### 任务4: 异常值检测与处理

```python
class OutlierHandler:
    """异常值处理器 - 结合流行病学先验"""
    
    def __init__(self, method='iqr_epidemiological'):
        self.method = method
    
    def detect_outliers_iqr(self, series: pd.Series, threshold: float = 1.5) -> pd.Series:
        """IQR方法检测异常值"""
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_reporting_delays(self, series: pd.Series) -> pd.Series:
        """检测报告延迟导致的异常峰值"""
        # 周一通常有报告堆积
        # TODO: 实现基于星期的异常检测
        pass
    
    def smooth_with_epidemiological_prior(self, series: pd.Series, 
                                          R0_estimate: float = 2.5) -> pd.Series:
        """结合流行病学先验进行平滑
        
        基于基本再生数R0，病例增长应满足一定的生物学约束
        """
        # TODO: 实现流行病学约束的平滑
        pass
```

### 任务5: 时序数据集构建

```python
# 文件: src/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader

class EpidemicTimeSeriesDataset(Dataset):
    """传染病时序数据集"""
    
    def __init__(self, data: np.ndarray, 
                 input_window: int = 21,      # 输入窗口: 21天
                 output_window: int = 7,       # 预测窗口: 7天
                 target_column: int = 0):      # 目标列索引
        """
        Args:
            data: 形状为 (时间步, 变量数) 的数组
            input_window: 输入时间窗口长度
            output_window: 预测时间窗口长度
            target_column: 预测目标列的索引
        """
        self.data = torch.FloatTensor(data)
        self.input_window = input_window
        self.output_window = output_window
        self.target_column = target_column
        
    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1
    
    def __getitem__(self, idx):
        # 输入序列: (input_window, num_features)
        x = self.data[idx:idx + self.input_window]
        # 输出序列: (output_window,) - 仅目标变量
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window, 
                      self.target_column]
        return x, y


class TimeSeriesSplitter:
    """时序交叉验证分割器 - 严格时序，防止未来信息泄露"""
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, data: np.ndarray):
        """生成时序交叉验证的训练/验证/测试索引"""
        n_samples = len(data)
        test_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # 测试集始终在最后
            test_start = n_samples - test_size
            test_end = n_samples
            
            # 验证集在测试集之前
            val_size = int((n_samples - test_size) * 0.2)
            val_start = test_start - val_size
            val_end = test_start
            
            # 训练集在验证集之前
            train_end = val_start
            
            yield {
                'train': (0, train_end),
                'val': (val_start, val_end),
                'test': (test_start, test_end)
            }

### 任务6：任务总结
完成任务之后请在Docs/completion_summary.md中记录任务完成情况

```

---

## 📈 数据质量检查清单

- [ ] 时间序列连续性检查（无缺失日期）
- [ ] 数据类型一致性检查
- [ ] 数值范围合理性检查
- [ ] 多数据源时间对齐验证
- [ ] 异常值标记与记录
- [ ] 缺失值处理记录
- [ ] 数据变换可逆性验证

---

## 📊 输出规范

### 预处理后数据格式
```python
processed_data = {
    'features': np.ndarray,        # 形状: (时间步, 变量数)
    'target': np.ndarray,          # 形状: (时间步,)
    'timestamps': pd.DatetimeIndex, # 时间索引
    'feature_names': List[str],    # 特征名称列表
    'scalers': Dict,               # 标准化器（用于反变换）
    'metadata': Dict               # 元数据信息
}
```

---

## ⚠️ 注意事项

1. **时序因果性**: 预处理时严禁使用未来数据
2. **可逆性**: 所有变换需保存参数，支持预测结果的反变换
3. **一致性**: 训练集和测试集使用相同的预处理管道
4. **文档化**: 记录所有预处理步骤和参数选择理由
