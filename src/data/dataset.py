"""
PyTorch Dataset类定义
PyTorch Dataset Module

用于构建可供DataLoader使用的数据集
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class EpidemicDataset(Dataset):
    """
    传染病预测数据集
    
    将预处理后的时间序列数据封装为PyTorch Dataset,
    支持多变量输入和多步预测输出
    """
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        """
        初始化数据集
        
        Args:
            X: 输入特征数组, shape: (samples, window_size, num_features)
            y: 目标值数组, shape: (samples, horizon) 或 (samples,)
            feature_names: 特征名称列表
            transform: 可选的数据变换函数
        """
        # TODO: 由 01_data_preparation_agent 实现
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.transform = transform
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (input_tensor, target_tensor) 元组
        """
        x = self.X[idx]
        y = self.y[idx]
        
        # 转换为torch张量
        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y) if isinstance(y, np.ndarray) else torch.FloatTensor([y])
        
        # 如果有transform，应用它
        if self.transform is not None:
            x_tensor = self.transform(x_tensor)
        
        return x_tensor, y_tensor
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        if self.X.ndim == 3:
            # shape: (samples, window_size, features)
            return self.X.shape[2]
        elif self.X.ndim == 2:
            # shape: (samples, features)
            return self.X.shape[1]
        else:
            return 1
    
    def get_window_size(self) -> int:
        """返回时间窗口大小"""
        if self.X.ndim == 3:
            # shape: (samples, window_size, features)
            return self.X.shape[1]
        else:
            # 如果是2D数据，窗口大小为1
            return 1


def create_data_loaders(
    train_dataset: EpidemicDataset,
    val_dataset: EpidemicDataset,
    test_dataset: EpidemicDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        
    Returns:
        (train_loader, val_loader, test_loader) 元组
    """
    # 训练集使用shuffle
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True  # 加速GPU数据传输
    )
    
    # 验证集和测试集不需要shuffle
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
