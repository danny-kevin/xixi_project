"""
接口协议定义
Protocol Definitions

使用Python的Protocol定义各模块的接口契约
"""

from typing import Protocol, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# ==================== 数据模块协议 ====================

class DataLoaderProtocol(Protocol):
    """数据加载器接口协议"""
    
    def load_epidemic_data(self, filename: str) -> pd.DataFrame:
        """加载疫情数据"""
        ...
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """加载所有类型数据"""
        ...
    
    def merge_data_sources(
        self,
        data_dict: Dict[str, pd.DataFrame],
        on: str = 'date'
    ) -> pd.DataFrame:
        """合并多源数据"""
        ...


class DataPreprocessorProtocol(Protocol):
    """数据预处理器接口协议"""
    
    def normalize(
        self,
        df: pd.DataFrame,
        method: str = 'standard'
    ) -> pd.DataFrame:
        """数据归一化"""
        ...
    
    def create_time_windows(
        self,
        data: np.ndarray,
        window_size: int,
        horizon: int,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列窗口"""
        ...
    
    def temporal_train_test_split(
        self,
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """时序数据划分"""
        ...


class EpidemicDatasetProtocol(Protocol):
    """数据集接口协议"""
    
    def __len__(self) -> int:
        """返回数据集大小"""
        ...
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        ...
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        ...


# ==================== 模型模块协议 ====================

class TCNProtocol(Protocol):
    """TCN模块接口协议"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        契约要求:
        - 输入: (batch, seq_len, input_size)
        - 输出: (batch, seq_len, output_channels)
        - 序列长度保持不变
        """
        ...
    
    def get_receptive_field(self) -> int:
        """
        获取感受野大小
        
        契约要求:
        - 返回值 >= 14 (覆盖14天滞后)
        """
        ...


class AttentionProtocol(Protocol):
    """注意力机制接口协议"""
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        契约要求:
        - 输入: (batch, seq_len, embed_dim)
        - 输出: (batch, seq_len, embed_dim)
        - 注意力权重每行和为1
        """
        ...


class LSTMProtocol(Protocol):
    """LSTM模块接口协议"""
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        契约要求:
        - 输入: (batch, seq_len, input_size)
        - 输出: (batch, seq_len, hidden_size * num_directions)
        """
        ...


class HybridModelProtocol(Protocol):
    """混合模型接口协议"""
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        契约要求:
        - 输入: (batch, seq_len, num_variables)
        - 输出: (batch, prediction_horizon, output_size)
        """
        ...
    
    def predict(
        self,
        x: torch.Tensor,
        forecast_steps: int = 7
    ) -> torch.Tensor:
        """多步预测"""
        ...


# ==================== 训练模块协议 ====================

class TrainerProtocol(Protocol):
    """训练器接口协议"""
    
    def train(
        self,
        train_loader,
        val_loader
    ) -> Dict[str, list]:
        """
        执行训练
        
        契约要求:
        - 返回字典包含: 'train_loss', 'val_loss'
        - 列表长度 = epoch数
        - 损失值 >= 0
        """
        ...
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """保存检查点"""
        ...
    
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        ...


class LossFunctionProtocol(Protocol):
    """损失函数接口协议"""
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算损失
        
        契约要求:
        - predictions和targets形状兼容
        - 返回标量张量
        - 返回值 >= 0
        """
        ...


# ==================== 评估模块协议 ====================

class MetricsProtocol(Protocol):
    """评估指标接口协议"""
    
    @staticmethod
    def mse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算MSE
        
        契约要求:
        - 形状相同
        - 返回值 >= 0
        - 无inf/nan
        """
        ...
    
    @staticmethod
    def rmse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """计算RMSE"""
        ...


class EvaluatorProtocol(Protocol):
    """评估器接口协议"""
    
    def evaluate(
        self,
        test_loader,
        return_predictions: bool = False
    ) -> Dict:
        """
        完整评估流程
        
        契约要求:
        - 返回字典包含: mse, rmse, mae, mape, r2
        - 所有指标为有限数
        """
        ...
    
    def generate_report(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """生成评估报告"""
        ...


# 使用示例
if __name__ == '__main__':
    print('接口协议定义')
    print('='*60)
    
    # 列出所有协议
    protocols = [
        'DataLoaderProtocol',
        'DataPreprocessorProtocol',
        'EpidemicDatasetProtocol',
        'TCNProtocol',
        'AttentionProtocol',
        'LSTMProtocol',
        'HybridModelProtocol',
        'TrainerProtocol',
        'LossFunctionProtocol',
        'MetricsProtocol',
        'EvaluatorProtocol'
    ]
    
    for i, protocol in enumerate(protocols, 1):
        print(f'{i}. {protocol}')
    
    print('='*60)
    print('这些协议定义了各模块必须实现的接口')
    print('实现时请确保符合契约要求')
