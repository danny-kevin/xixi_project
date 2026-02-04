"""
工具模块
Utils Module

包含配置管理、日志、设备管理、可视化等工具
"""

from .config import AttentionConfig, Config, load_config, save_config, get_default_config
from .logger import setup_logger, get_logger, LoggerContext
from .device_manager import DeviceManager, get_device
from .seed import set_seed, get_random_state, set_random_state, SeedContext
from .checkpoint import (
    save_checkpoint, load_checkpoint,
    save_model_only, load_model_only,
    CheckpointManager
)
from .shape_validator import ShapeValidator, validate_tensor_shape, validate_output_shape
from .type_utils import to_tensor, to_numpy, ensure_tensor, ensure_numpy, check_type, safe_cast
from .experiment_tracker import ExperimentTracker
from .visualization import Visualizer
from .attention_visualization import AttentionVisualizer
from .protocols import (
    DataLoaderProtocol, DataPreprocessorProtocol, EpidemicDatasetProtocol,
    TCNProtocol, AttentionProtocol, LSTMProtocol, HybridModelProtocol,
    TrainerProtocol, LossFunctionProtocol,
    MetricsProtocol, EvaluatorProtocol
)

__all__ = [
    # 配置
    'AttentionConfig', 'Config', 'load_config', 'save_config', 'get_default_config',
    
    # 日志
    'setup_logger', 'get_logger', 'LoggerContext',
    
    # 设备管理
    'DeviceManager', 'get_device',
    
    # 随机种子
    'set_seed', 'get_random_state', 'set_random_state', 'SeedContext',
    
    # 检查点
    'save_checkpoint', 'load_checkpoint',
    'save_model_only', 'load_model_only',
    'CheckpointManager',
    
    # 形状验证
    'ShapeValidator', 'validate_tensor_shape', 'validate_output_shape',
    
    # 类型工具
    'to_tensor', 'to_numpy', 'ensure_tensor', 'ensure_numpy', 'check_type', 'safe_cast',
    
    # 实验追踪
    'ExperimentTracker',
    
    # 可视化
    'Visualizer',
    'AttentionVisualizer',
    
    # 协议
    'DataLoaderProtocol', 'DataPreprocessorProtocol', 'EpidemicDatasetProtocol',
    'TCNProtocol', 'AttentionProtocol', 'LSTMProtocol', 'HybridModelProtocol',
    'TrainerProtocol', 'LossFunctionProtocol',
    'MetricsProtocol', 'EvaluatorProtocol'
]
