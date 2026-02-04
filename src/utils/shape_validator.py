"""
形状验证工具
Shape Validator

提供张量形状的运行时验证和装饰器
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class ShapeValidator:
    """
    张量形状验证器
    
    用于验证张量形状是否符合预期，帮助早期发现数据流问题
    """
    
    @staticmethod
    def validate_shape(
        tensor: Union[torch.Tensor, np.ndarray],
        expected_shape: Tuple[Optional[int], ...],
        name: str = "tensor"
    ) -> None:
        """
        验证张量形状
        
        Args:
            tensor: 待验证的张量
            expected_shape: 期望的形状，None表示任意维度
            name: 张量名称（用于错误信息）
            
        Raises:
            ValueError: 形状不匹配时
            
        Example:
            >>> x = torch.randn(32, 21, 11)
            >>> ShapeValidator.validate_shape(x, (32, 21, 11), "input")
            >>> ShapeValidator.validate_shape(x, (None, 21, 11), "input")  # batch可变
        """
        actual_shape = tensor.shape
        
        # 检查维度数量
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"{name} 维度数量不匹配: "
                f"期望 {len(expected_shape)} 维, 实际 {len(actual_shape)} 维\n"
                f"期望形状: {expected_shape}\n"
                f"实际形状: {actual_shape}"
            )
        
        # 检查每个维度
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{name} 第{i}维大小不匹配: "
                    f"期望 {expected}, 实际 {actual}\n"
                    f"期望形状: {expected_shape}\n"
                    f"实际形状: {actual_shape}"
                )
        
        logger.debug(f"✅ {name} 形状验证通过: {actual_shape}")
    
    @staticmethod
    def validate_range(
        tensor: Union[torch.Tensor, np.ndarray],
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "tensor"
    ) -> None:
        """
        验证张量值范围
        
        Args:
            tensor: 待验证的张量
            min_val: 最小值（None表示不检查）
            max_val: 最大值（None表示不检查）
            name: 张量名称
            
        Raises:
            ValueError: 值超出范围时
        """
        if isinstance(tensor, torch.Tensor):
            actual_min = tensor.min().item()
            actual_max = tensor.max().item()
        else:
            actual_min = tensor.min()
            actual_max = tensor.max()
        
        if min_val is not None and actual_min < min_val:
            raise ValueError(
                f"{name} 包含小于最小值的元素: "
                f"最小值={actual_min}, 期望>={min_val}"
            )
        
        if max_val is not None and actual_max > max_val:
            raise ValueError(
                f"{name} 包含大于最大值的元素: "
                f"最大值={actual_max}, 期望<={max_val}"
            )
        
        logger.debug(f"✅ {name} 值范围验证通过: [{actual_min:.4f}, {actual_max:.4f}]")
    
    @staticmethod
    def validate_no_nan_inf(
        tensor: Union[torch.Tensor, np.ndarray],
        name: str = "tensor"
    ) -> None:
        """
        验证张量不包含NaN或Inf
        
        Args:
            tensor: 待验证的张量
            name: 张量名称
            
        Raises:
            ValueError: 包含NaN或Inf时
        """
        if isinstance(tensor, torch.Tensor):
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
        else:
            has_nan = np.isnan(tensor).any()
            has_inf = np.isinf(tensor).any()
        
        if has_nan:
            raise ValueError(f"{name} 包含NaN值")
        
        if has_inf:
            raise ValueError(f"{name} 包含Inf值")
        
        logger.debug(f"✅ {name} NaN/Inf验证通过")
    
    @staticmethod
    def validate_data_window(
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        window_size: int,
        horizon: int,
        num_features: int
    ) -> None:
        """
        验证时间窗口数据的形状
        
        Args:
            X: 输入窗口
            y: 目标值
            window_size: 窗口大小
            horizon: 预测范围
            num_features: 特征数量
        """
        # 验证X形状
        ShapeValidator.validate_shape(
            X,
            (None, window_size, num_features),
            "X (输入窗口)"
        )
        
        # 验证y形状
        ShapeValidator.validate_shape(
            y,
            (None, horizon),
            "y (目标值)"
        )
        
        # 验证样本数量一致
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X和y的样本数量不一致: X={X.shape[0]}, y={y.shape[0]}"
            )
        
        logger.info(f"✅ 数据窗口验证通过: X{X.shape}, y{y.shape}")
    
    @staticmethod
    def validate_model_pipeline(
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        expected_output_shape: Optional[Tuple] = None
    ) -> torch.Tensor:
        """
        验证模型完整流程
        
        Args:
            model: 模型
            sample_input: 示例输入
            expected_output_shape: 期望的输出形状
            
        Returns:
            模型输出
        """
        logger.info("开始验证模型流程...")
        
        # 验证输入
        ShapeValidator.validate_no_nan_inf(sample_input, "model_input")
        logger.info(f"输入形状: {sample_input.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
            if isinstance(output, tuple):
                output = output[0]  # 如果返回多个值，取第一个
        
        # 验证输出
        ShapeValidator.validate_no_nan_inf(output, "model_output")
        logger.info(f"输出形状: {output.shape}")
        
        # 验证输出形状
        if expected_output_shape is not None:
            ShapeValidator.validate_shape(
                output,
                expected_output_shape,
                "model_output"
            )
        
        logger.info("✅ 模型流程验证通过")
        return output


def validate_tensor_shape(
    arg_name: str,
    expected_shape: Tuple[Optional[int], ...]
) -> Callable:
    """
    装饰器：自动验证函数参数的张量形状
    
    Args:
        arg_name: 参数名称
        expected_shape: 期望的形状
        
    Returns:
        装饰器函数
        
    Example:
        >>> @validate_tensor_shape('x', (None, 21, 11))
        >>> def forward(self, x):
        >>>     return self.process(x)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取参数
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证指定参数
            if arg_name in bound_args.arguments:
                tensor = bound_args.arguments[arg_name]
                if tensor is not None:
                    ShapeValidator.validate_shape(
                        tensor,
                        expected_shape,
                        f"{func.__name__}.{arg_name}"
                    )
            
            # 调用原函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def validate_output_shape(expected_shape: Tuple[Optional[int], ...]) -> Callable:
    """
    装饰器：自动验证函数返回值的张量形状
    
    Args:
        expected_shape: 期望的形状
        
    Returns:
        装饰器函数
        
    Example:
        >>> @validate_output_shape((None, 7, 1))
        >>> def forward(self, x):
        >>>     return self.predict(x)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 调用原函数
            result = func(*args, **kwargs)
            
            # 验证返回值
            if result is not None:
                # 如果返回元组，验证第一个元素
                tensor = result[0] if isinstance(result, tuple) else result
                ShapeValidator.validate_shape(
                    tensor,
                    expected_shape,
                    f"{func.__name__} output"
                )
            
            return result
        
        return wrapper
    return decorator


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.DEBUG)
    
    # 示例1: 基本形状验证
    x = torch.randn(32, 21, 11)
    ShapeValidator.validate_shape(x, (32, 21, 11), "x")
    ShapeValidator.validate_shape(x, (None, 21, 11), "x")  # batch可变
    
    # 示例2: 值范围验证
    normalized_x = torch.randn(32, 21, 11)
    ShapeValidator.validate_range(normalized_x, min_val=-10, max_val=10, name="normalized_x")
    
    # 示例3: NaN/Inf验证
    ShapeValidator.validate_no_nan_inf(x, "x")
    
    # 示例4: 数据窗口验证
    X = torch.randn(100, 21, 11)
    y = torch.randn(100, 7)
    ShapeValidator.validate_data_window(X, y, window_size=21, horizon=7, num_features=11)
    
    # 示例5: 使用装饰器
    class DummyModel(torch.nn.Module):
        @validate_tensor_shape('x', (None, 21, 11))
        @validate_output_shape((None, 7, 1))
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 7, 1)
    
    model = DummyModel()
    output = model(x)
    print(f"✅ 所有验证通过！输出形状: {output.shape}")
