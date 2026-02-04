"""
随机种子管理
Seed Manager

确保实验的可复现性
"""

import random
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法（可能会降低性能）
        
    Note:
        - 设置Python、NumPy、PyTorch的随机种子
        - 如果使用CUDA，也会设置CUDA种子
        - deterministic=True时，cuDNN使用确定性算法
    """
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    
    # CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    
    # 确定性算法
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'[OK] 随机种子已设置: {seed} (确定性模式)')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[OK] 随机种子已设置: {seed} (非确定性模式，性能更好)')
    
    logger.debug(f'Python seed: {seed}')
    logger.debug(f'NumPy seed: {seed}')
    logger.debug(f'PyTorch seed: {seed}')


def get_random_state() -> dict:
    """
    获取当前随机状态
    
    Returns:
        包含所有随机状态的字典
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    恢复随机状态
    
    Args:
        state: 随机状态字典（由get_random_state()返回）
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    
    logger.info('[OK] 随机状态已恢复')


class SeedContext:
    """
    随机种子上下文管理器
    
    在with块内使用指定的种子，退出时恢复原状态
    
    Example:
        >>> set_seed(42)
        >>> x1 = torch.randn(5)
        >>> with SeedContext(123):
        >>>     x2 = torch.randn(5)  # 使用种子123
        >>> x3 = torch.randn(5)  # 恢复到种子42的状态
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        self.seed = seed
        self.deterministic = deterministic
        self.old_state = None
    
    def __enter__(self):
        # 保存当前状态
        self.old_state = get_random_state()
        # 设置新种子
        set_seed(self.seed, self.deterministic)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复旧状态
        set_random_state(self.old_state)


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例1: 基本使用
    print('示例1: 基本使用')
    set_seed(42)
    x1 = torch.randn(5)
    print(f'第一次生成: {x1}')
    
    set_seed(42)  # 重新设置相同种子
    x2 = torch.randn(5)
    print(f'第二次生成: {x2}')
    print(f'是否相同: {torch.allclose(x1, x2)}')
    
    # 示例2: 使用上下文管理器
    print('\n示例2: 使用上下文管理器')
    set_seed(42)
    x1 = torch.randn(5)
    print(f'初始: {x1}')
    
    with SeedContext(123):
        x2 = torch.randn(5)
        print(f'上下文内: {x2}')
    
    x3 = torch.randn(5)
    print(f'上下文外: {x3}')
    
    # 示例3: 保存和恢复状态
    print('\n示例3: 保存和恢复状态')
    set_seed(42)
    state = get_random_state()
    
    x1 = torch.randn(5)
    x2 = torch.randn(5)
    print(f'生成两个张量: {x1[0]:.4f}, {x2[0]:.4f}')
    
    set_random_state(state)  # 恢复状态
    x3 = torch.randn(5)
    x4 = torch.randn(5)
    print(f'恢复后生成: {x3[0]:.4f}, {x4[0]:.4f}')
    print(f'是否相同: {torch.allclose(x1, x3)} and {torch.allclose(x2, x4)}')
    
    print('\n[OK] 随机种子管理测试完成')
