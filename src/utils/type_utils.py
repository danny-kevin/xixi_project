"""
类型工具
Type Utilities

提供类型检查和转换的实用工具
"""

import torch
import numpy as np
from typing import Union, TypeVar, Type, get_args, get_origin
import logging

logger = logging.getLogger(__name__)

# 类型变量
TensorLike = Union[torch.Tensor, np.ndarray]
T = TypeVar('T')


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, list, float],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    将数据转换为PyTorch张量
    
    Args:
        data: 输入数据
        dtype: 目标数据类型
        device: 目标设备
        
    Returns:
        PyTorch张量
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(dtype=dtype, device=device)
    else:
        return torch.tensor(data, dtype=dtype, device=device)


def to_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    将数据转换为NumPy数组
    
    Args:
        data: 输入数据
        
    Returns:
        NumPy数组
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


def ensure_tensor(
    data: TensorLike,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    确保数据是PyTorch张量
    
    Args:
        data: 输入数据
        dtype: 目标数据类型
        
    Returns:
        PyTorch张量
    """
    if not isinstance(data, torch.Tensor):
        data = to_tensor(data, dtype=dtype)
    return data.to(dtype=dtype)


def ensure_numpy(data: TensorLike) -> np.ndarray:
    """
    确保数据是NumPy数组
    
    Args:
        data: 输入数据
        
    Returns:
        NumPy数组
    """
    if not isinstance(data, np.ndarray):
        data = to_numpy(data)
    return data


def check_type(
    value: any,
    expected_type: Type,
    name: str = "value"
) -> None:
    """
    检查值的类型
    
    Args:
        value: 待检查的值
        expected_type: 期望的类型
        name: 值的名称（用于错误信息）
        
    Raises:
        TypeError: 类型不匹配时
    """
    # 处理Union类型
    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        if not isinstance(value, args):
            raise TypeError(
                f"{name} 类型错误: "
                f"期望 {expected_type}, 实际 {type(value)}"
            )
    else:
        if not isinstance(value, expected_type):
            raise TypeError(
                f"{name} 类型错误: "
                f"期望 {expected_type}, 实际 {type(value)}"
            )


def safe_cast(
    value: any,
    target_type: Type[T],
    default: T = None
) -> T:
    """
    安全类型转换
    
    Args:
        value: 待转换的值
        target_type: 目标类型
        default: 转换失败时的默认值
        
    Returns:
        转换后的值或默认值
    """
    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"类型转换失败: {value} -> {target_type}, 使用默认值: {default}")
        return default


# 使用示例
if __name__ == '__main__':
    # 示例1: 转换为张量
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    tensor = to_tensor(arr)
    print(f"NumPy -> Tensor: {tensor.shape}, {tensor.dtype}")
    
    # 示例2: 转换为NumPy
    tensor = torch.randn(3, 4)
    arr = to_numpy(tensor)
    print(f"Tensor -> NumPy: {arr.shape}, {arr.dtype}")
    
    # 示例3: 类型检查
    check_type(tensor, torch.Tensor, "tensor")
    check_type(arr, np.ndarray, "arr")
    
    # 示例4: 安全转换
    result = safe_cast("123", int, default=0)
    print(f"Safe cast: '123' -> {result}")
    
    result = safe_cast("abc", int, default=0)
    print(f"Safe cast (failed): 'abc' -> {result}")
    
    print("[OK] 所有类型工具测试通过")
