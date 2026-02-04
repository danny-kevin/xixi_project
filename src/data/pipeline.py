"""
数据预处理管道
Data Preprocessing Pipeline

整合数据加载、预处理、数据集创建的完整流程
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .dataset import EpidemicDataset, create_data_loaders


class DataPipeline:
    """
    端到端数据预处理管道
    
    整合了数据加载、清洗、预处理、数据集构建的完整流程
    """
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 21,
        horizon: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        初始化数据管道
        
        Args:
            data_dir: 数据目录路径
            window_size: 输入时间窗口大小（天）
            horizon: 预测时间范围（天）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 初始化组件
        self.loader = DataLoader(data_dir)
        self.preprocessor = DataPreprocessor()
        
        # 存储处理后的数据
        self.merged_data: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        
    def load_data(self) -> pd.DataFrame:
        """
        加载所有数据源
        
        Returns:
            合并后的DataFrame
        """
        print("📊 加载数据...")
        data_dict = self.loader.load_all_data()
        
        if not data_dict:
            raise ValueError("未能加载任何数据源")
        
        print(f"✅ 成功加载 {len(data_dict)} 个数据源")
        
        # 合并数据
        print("🔗 合并数据源...")
        merged_data = self.loader.merge_data_sources(data_dict)
        print(f"✅ 合并完成，数据形状: {merged_data.shape}")
        
        self.merged_data = merged_data
        return merged_data
    
    def preprocess_data(
        self,
        df: Optional[pd.DataFrame] = None,
        handle_missing: bool = True,
        detect_outliers: bool = True,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 输入DataFrame（如果为None，使用self.merged_data）
            handle_missing: 是否处理缺失值
            detect_outliers: 是否检测异常值
            normalize: 是否归一化
            
        Returns:
            预处理后的DataFrame
        """
        if df is None:
            if self.merged_data is None:
                raise ValueError("请先调用load_data()加载数据")
            df = self.merged_data
        
        df_processed = df.copy()
        
        # 1. 处理缺失值
        if handle_missing:
            print("🔧 处理缺失值...")
            missing_count = df_processed.isnull().sum().sum()
            if missing_count > 0:
                print(f"   发现 {missing_count} 个缺失值")
                df_processed = self.preprocessor.handle_missing_values(
                    df_processed, 
                    method='interpolate'
                )
                print(f"   ✅ 缺失值处理完成")
        
        # 2. 检测异常值
        if detect_outliers:
            print("🔍 检测异常值...")
            outliers = self.preprocessor.detect_outliers(
                df_processed, 
                method='iqr', 
                threshold=1.5
            )
            outlier_count = outliers.sum().sum()
            if outlier_count > 0:
                print(f"   发现 {outlier_count} 个异常值")
                # 这里可以选择处理异常值，暂时只记录
        
        # 3. 数据归一化
        if normalize:
            print("📏 数据归一化...")
            df_processed = self.preprocessor.normalize(
                df_processed,
                method='minmax'
            )
            print(f"   ✅ 归一化完成")
        
        self.feature_names = df_processed.columns.tolist()
        return df_processed
    
    def create_datasets(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[EpidemicDataset, EpidemicDataset, EpidemicDataset]:
        """
        创建训练、验证、测试数据集
        
        Args:
            df: 预处理后的DataFrame
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        if df is None:
            raise ValueError("请提供预处理后的数据")
        
        print("🔨 创建时间序列窗口...")
        
        # 转换为numpy数组
        data_array = df.values
        
        # 创建时间窗口
        X, y = self.preprocessor.create_time_windows(
            data_array,
            window_size=self.window_size,
            horizon=self.horizon,
            stride=1
        )
        
        print(f"   输入形状: {X.shape}, 目标形状: {y.shape}")
        
        # 时序划分
        print("✂️ 划分数据集...")
        n_samples = len(X)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        print(f"   训练集: {len(X_train)} 样本")
        print(f"   验证集: {len(X_val)} 样本")
        print(f"   测试集: {len(X_test)} 样本")
        
        # 创建Dataset对象
        train_dataset = EpidemicDataset(X_train, y_train, self.feature_names)
        val_dataset = EpidemicDataset(X_val, y_val, self.feature_names)
        test_dataset = EpidemicDataset(X_test, y_test, self.feature_names)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: EpidemicDataset,
        val_dataset: EpidemicDataset,
        test_dataset: EpidemicDataset
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        创建DataLoader
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            test_dataset: 测试数据集
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print("🚀 创建DataLoader...")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        print(f"   批次大小: {self.batch_size}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(val_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def run(self) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        运行完整的数据准备管道
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print("=" * 60)
        print("🎯 开始数据预处理管道")
        print("=" * 60)
        
        # 1. 加载数据
        merged_data = self.load_data()
        
        # 2. 预处理
        processed_data = self.preprocess_data(
            merged_data,
            handle_missing=True,
            detect_outliers=True,
            normalize=True
        )
        
        # 3. 创建数据集
        train_dataset, val_dataset, test_dataset = self.create_datasets(processed_data)
        
        # 4. 创建DataLoader
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset
        )
        
        print("=" * 60)
        print("✅ 数据预处理管道完成！")
        print("=" * 60)
        
        return train_loader, val_loader, test_loader
    
    def get_preprocessor(self) -> DataPreprocessor:
        """获取预处理器（用于反归一化等操作）"""
        return self.preprocessor
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names


# 便捷函数
def prepare_data(
    data_dir: str,
    window_size: int = 21,
    horizon: int = 7,
    batch_size: int = 32,
    **kwargs
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, DataPreprocessor]:
    """
    一键准备数据的便捷函数
    
    Args:
        data_dir: 数据目录
        window_size: 输入窗口大小
        horizon: 预测范围
        batch_size: 批次大小
        **kwargs: 其他参数
        
    Returns:
        (train_loader, val_loader, test_loader, preprocessor)
    """
    pipeline = DataPipeline(
        data_dir=data_dir,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        **kwargs
    )
    
    train_loader, val_loader, test_loader = pipeline.run()
    preprocessor = pipeline.get_preprocessor()
    
    return train_loader, val_loader, test_loader, preprocessor
