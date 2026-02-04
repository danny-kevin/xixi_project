"""
美国COVID-19数据加载器
US COVID-19 Data Loader

专门用于加载和预处理 dataset_US_final.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import sys

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.dataset import EpidemicDataset, create_data_loaders


class USCovidDataLoader:
    """
    美国COVID-19数据加载器
    
    处理dataset_US_final.csv文件
    列: Date, Confirmed, Deaths, Stringency, Mobility_Work, Mobility_Transit, Mobility_Home
    """
    
    def __init__(self, data_path: str = "data/raw/dataset_US_final.csv"):
        """
        初始化数据加载器
        
        Args:
            data_path: CSV文件路径
        """
        self.data_path = Path(data_path)
        
    def load_data(self) -> pd.DataFrame:
        """
        加载CSV文件
        
        Returns:
            DataFrame with parsed dates
        """
        df = pd.read_csv(self.data_path, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        
        print(f"✓ 加载数据: {len(df)} 行")
        print(f"✓ 日期范围: {df.index.min()} 到 {df.index.max()}")
        print(f"✓ 列名: {list(df.columns)}")
        
        return df
    
    def prepare_data(
        self,
        target_column: str = 'Confirmed',
        window_size: int = 21,
        horizon: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: str = 'standard'
    ) -> Dict:
        """
        准备训练数据
        
        Args:
            target_column: 目标列名（要预测的变量）
            window_size: 历史窗口大小
            horizon: 预测范围
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            scaler_type: 归一化方法
            
        Returns:
            包含X_train, y_train等的字典
        """
        # 1. 加载数据
        df = self.load_data()
        
        # 2. 处理缺失值
        print("\n检查缺失值:")
        missing = df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
            print("  使用前向填充处理缺失值...")
            df = df.fillna(method='ffill').fillna(method='bfill')
        else:
            print("  ✓ 无缺失值")
        
        # 3. 分离特征和目标
        feature_columns = [col for col in df.columns if col != target_column]
        print(f"\n特征列 ({len(feature_columns)}): {feature_columns}")
        print(f"目标列: {target_column}")
        
        # 4. 归一化
        preprocessor = DataPreprocessor()
        
        # 分别归一化特征和目标
        df_normalized = df.copy()
        df_normalized[feature_columns] = preprocessor.normalize(
            df[feature_columns],
            method=scaler_type
        )
        df_normalized[target_column] = preprocessor.normalize(
            df[[target_column]],
            method=scaler_type
        )[target_column]
        
        print(f"✓ 归一化完成 ({scaler_type})")
        
        # 5. 创建时间窗口
        # 合并所有列（特征+目标）用于窗口创建
        all_data = df_normalized.values  # shape: (时间步, 特征数)
        
        X, y = preprocessor.create_time_windows(
            all_data,
            window_size=window_size,
            horizon=horizon
        )
        
        print(f"\n创建时间窗口:")
        print(f"  X shape: {X.shape}  # (samples, window_size, num_features)")
        print(f"  y shape: {y.shape}  # (samples, horizon) or (samples,)")
        
        # 6. 时序划分数据集
        X_train, X_val, X_test = preprocessor.temporal_train_test_split(
            X, train_ratio=train_ratio, val_ratio=val_ratio
        )
        y_train, y_val, y_test = preprocessor.temporal_train_test_split(
            y, train_ratio=train_ratio, val_ratio=val_ratio
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'preprocessor': preprocessor,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'original_df': df
        }
    
    def create_dataloaders(
        self,
        data_dict: Dict,
        batch_size: int = 32,
        num_workers: int = 4
    ) ->Tuple:
        """
        创建PyTorch DataLoaders
        
        Args:
            data_dict: prepare_data()返回的字典
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # 创建Datasets
        train_dataset = EpidemicDataset(
            data_dict['X_train'],
            data_dict['y_train'],
            feature_names=data_dict['feature_columns']
        )
        
        val_dataset = EpidemicDataset(
            data_dict['X_val'],
            data_dict['y_val'],
            feature_names=data_dict['feature_columns']
        )
        
        test_dataset = EpidemicDataset(
            data_dict['X_test'],
            data_dict['y_test'],
            feature_names=data_dict['feature_columns']
        )
        
        # 创建DataLoaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        print(f"\nDataLoader创建完成:")
        print(f"  Batch size: {batch_size}")
        print(f"  训练批次数: {len(train_loader)}")
        print(f"  验证批次数: {len(val_loader)}")
        print(f"  测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader


def main():
    """测试数据加载"""
    print("=" * 60)
    print("美国COVID-19数据加载测试")
    print("=" * 60)
    
    loader = USCovidDataLoader()
    
    # 准备数据
    data_dict = loader.prepare_data(
        target_column='Confirmed',
        window_size=21,
        horizon=7
    )
    
    # 创建DataLoaders
    train_loader, val_loader, test_loader = loader.create_dataloaders(
        data_dict,
        batch_size=32
    )
    
    # 测试一个批次
    print("\n" + "=" *  60)
    print("测试第一个批次:")
    print("=" * 60)
    for batch_X, batch_y in train_loader:
        print(f"  batch_X shape: {batch_X.shape}")  # (batch, window, features)
        print(f"  batch_y shape: {batch_y.shape}")  # (batch, horizon) or (batch,)
        print(f"  batch_X dtype: {batch_X.dtype}")
        print(f"  batch_y dtype: {batch_y.dtype}")
        break
    
    print("\n✅ 数据加载测试成功！")


if __name__ == '__main__':
    main()
