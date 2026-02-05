"""
配置管理模块
Configuration Module

统一管理项目的所有配置参数
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    processed_csv: str = ""
    
    # 时间窗口设置
    window_size: int = 21  # 历史窗口大小 (天)
    prediction_horizon: int = 7  # 预测范围 (天)
    stride: int = 1  # 滑动步长
    
    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # 预处理
    scaler_type: str = "standard"  # 'standard' 或 'minmax'
    handle_missing: str = "interpolate"
    
    # 变量配置
    target_column: str = "new_cases"
    feature_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """模型配置"""
    # M-TCN配置
    num_variables: int = 10  # 输入变量数量
    input_size: int = 1  # 每个变量的特征维度
    tcn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    tcn_kernel_size: int = 3
    
    # 注意力配置
    attention_embed_dim: int = 128
    attention_num_heads: int = 8
    attention_dropout: float = 0.1
    
    # LSTM配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    
    # 输出配置
    output_size: int = 1
    prediction_horizon: int = 7
    
    # 正则化
    dropout: float = 0.2


@dataclass
class AttentionConfig:
    """注意力配置"""
    type: str = "spatiotemporal"  # variable, temporal, spatiotemporal
    num_heads: int = 4
    dropout: float = 0.1
    stochastic_dropout: float = 0.1
    use_layer_norm: bool = True
    use_residual: bool = True


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 优化器
    optimizer: str = "adamw"
    
    # 学习率调度
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # 早停
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4
    
    # 多阶段训练
    use_pretrain: bool = True
    pretrain_epochs: int = 20
    finetune_lr_ratio: float = 0.1
    
    # 梯度裁剪
    gradient_clip_val: float = 1.0
    
    # 设备
    device: str = "cuda"
    num_workers: int = 4
    
    # 保存
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    log_interval: int = 10


@dataclass
class Config:
    """
    总配置类
    
    整合所有子配置
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    
    # 实验设置
    seed: int = 42
    experiment_name: str = "attention_mtcn_lstm"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置转换为字典
        
        Returns:
            配置字典
        """
        from dataclasses import asdict
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'attention': asdict(self.attention),
            'seed': self.seed,
            'experiment_name': self.experiment_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Config实例
        """
        # 创建子配置
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        attention_config = AttentionConfig(**config_dict.get('attention', {}))
        
        # 创建主配置
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            attention=attention_config,
            seed=config_dict.get('seed', 42),
            experiment_name=config_dict.get('experiment_name', 'attention_mtcn_lstm')
        )


def load_config(config_path: Union[str, Path]) -> Config:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config实例
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """
    保存配置到YAML文件
    
    Args:
        config: 配置实例
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_default_config() -> Config:
    """
    获取默认配置
    
    Returns:
        默认Config实例
    """
    return Config()
