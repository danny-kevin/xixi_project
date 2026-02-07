"""
数据预处理模块
Data Preprocessor Module

负责数据清洗、归一化、特征工程等预处理操作
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# scikit-learn is optional. The project can run without it (e.g. in minimal envs)
# as long as we provide compatible scalers for normalize().
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class StandardScaler:  # minimal sklearn-like API
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X):
            arr = np.asarray(X, dtype=float)
            self.mean_ = arr.mean(axis=0)
            self.scale_ = arr.std(axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
            return self

        def transform(self, X):
            arr = np.asarray(X, dtype=float)
            return (arr - self.mean_) / self.scale_

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class MinMaxScaler:  # minimal sklearn-like API
        def __init__(self, feature_range=(0.0, 1.0)):
            self.feature_range = feature_range
            self.data_min_ = None
            self.data_max_ = None
            self.scale_ = None
            self.min_ = None

        def fit(self, X):
            arr = np.asarray(X, dtype=float)
            self.data_min_ = arr.min(axis=0)
            self.data_max_ = arr.max(axis=0)
            data_range = self.data_max_ - self.data_min_
            data_range = np.where(data_range == 0, 1.0, data_range)

            fr_min, fr_max = self.feature_range
            self.scale_ = (fr_max - fr_min) / data_range
            self.min_ = fr_min - self.data_min_ * self.scale_
            return self

        def transform(self, X):
            arr = np.asarray(X, dtype=float)
            return arr * self.scale_ + self.min_

        def fit_transform(self, X):
            return self.fit(X).transform(X)


class DataPreprocessor:
    """
    数据预处理器
    
    功能:
    - 缺失值处理
    - 异常值检测与处理
    - 数据归一化/标准化
    - 时间序列窗口构建
    - 特征工程
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置参数
        """
        self.config = config or {}
        self.scalers: Dict[str, Union[StandardScaler, MinMaxScaler]] = {}
        self.feature_names: List[str] = []
        
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入DataFrame
            method: 填充方法 ('interpolate', 'ffill', 'bfill', 'mean')
            
        Returns:
            处理后的DataFrame
        """
        df_filled = df.copy()
        
        if method == 'interpolate':
            # 时间序列插值（线性插值）
            df_filled = df_filled.interpolate(method='time', limit_direction='both')
        elif method == 'ffill':
            # 前向填充
            df_filled = df_filled.fillna(method='ffill')
        elif method == 'bfill':
            # 后向填充
            df_filled = df_filled.fillna(method='bfill')
        elif method == 'mean':
            # 均值填充
            df_filled = df_filled.fillna(df_filled.mean())
        else:
            raise ValueError(f"不支持的填充方法: {method}")
        
        # 如果仍有缺失值，使用前向填充
        if df_filled.isnull().any().any():
            df_filled = df_filled.fillna(method='ffill').fillna(method='bfill')
        
        return df_filled
    
    def detect_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        检测异常值
        
        Args:
            df: 输入DataFrame
            method: 检测方法 ('iqr', 'zscore')
            threshold: 阈值
            
        Returns:
            异常值标记的DataFrame (True表示异常值)
        """
        outliers = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        if method == 'iqr':
            # IQR方法
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            # Z-score方法
            for col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                z_scores = np.abs((df[col] - mean) / std)
                outliers[col] = z_scores > threshold
        
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        return outliers
    
    def normalize(
        self, 
        df: pd.DataFrame, 
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        数据归一化
        
        Args:
            df: 输入DataFrame
            method: 归一化方法 ('standard', 'minmax')
            columns: 需要归一化的列
            
        Returns:
            归一化后的DataFrame
        """
        df_normalized = df.copy()
        
        # 如果未指定列，则对所有数值列进行归一化
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # 创建并保存scaler
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"不支持的归一化方法: {method}")
            
            # 拟合并转换
            df_normalized[col] = scaler.fit_transform(df[[col]])
            
            # 保存scaler以便后续反归一化
            self.scalers[col] = scaler
        
        # 保存特征名称
        self.feature_names = columns
        
        return df_normalized
    
    def create_time_windows(
        self, 
        data: np.ndarray,
        window_size: int,
        horizon: int,
        stride: int = 1,
        target_col_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列滑动窗口
        
        Args:
            data: 输入数据数组, shape: (timesteps, features)
            window_size: 窗口大小(历史天数)
            horizon: 预测范围(预测天数)
            stride: 滑动步长
            
        Returns:
            (X, y) - 输入窗口和目标值
            X shape: (samples, window_size, features)
            y shape: (samples, horizon) 或 (samples,) if horizon=1
        """
        X, y = [], []
        
        # 确保数据是2D的
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_samples = len(data)

        if target_col_idx < 0 or target_col_idx >= data.shape[1]:
            raise ValueError(f"target_col_idx out of range: {target_col_idx}")
        
        for i in range(0, n_samples - window_size - horizon + 1, stride):
            # 输入窗口: 过去window_size个时间步的所有特征
            X.append(data[i:i + window_size])
            
            # 目标值: 未来horizon个时间步的第一个特征（通常是new_cases）
            if horizon == 1:
                y.append(data[i + window_size, target_col_idx])
            else:
                y.append(data[i + window_size:i + window_size + horizon, target_col_idx])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def temporal_train_test_split(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        时序数据划分 (严格按时间顺序，避免信息泄露)
        
        Args:
            data: 输入数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (train, val, test) 数据集元组
        """
        n_samples = len(data)

        # 计算分割点
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 严格按时间顺序划分
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        return train_data, val_data, test_data
    
    def inverse_transform(
        self, 
        data: np.ndarray, 
        column: str
    ) -> np.ndarray:
        """
        反归一化 (用于预测结果还原)
        
        Args:
            data: 归一化后的数据
            column: 列名
            
        Returns:
            反归一化后的数据
        """
        if column not in self.scalers:
            raise ValueError(f"列 '{column}' 没有对应的scaler，无法反归一化")
        
        scaler = self.scalers[column]
        
        # 确保数据形状正确
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # 反归一化
        original_data = scaler.inverse_transform(data)
        
        return original_data.flatten()
