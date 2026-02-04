"""
学习率调度器模块
Learning Rate Scheduler Module

包含自定义学习率调度策略
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import List


class WarmupCosineScheduler(_LRScheduler):
    """
    带Warmup的余弦退火学习率调度器
    
    训练初期线性增长学习率 (warmup)，
    然后按余弦曲线逐渐降低
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: warmup阶段epoch数
            total_epochs: 总训练epoch数
            min_lr: 最小学习率
            last_epoch: 上一个epoch (用于恢复训练)
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        获取当前学习率
        
        Returns:
            各参数组的学习率列表
        """
        epoch = self.last_epoch + 1
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            scale = epoch / float(self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]

        if self.total_epochs <= self.warmup_epochs:
            return [self.min_lr for _ in self.base_lrs]

        progress = (epoch - self.warmup_epochs) / float(self.total_epochs - self.warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base_lr - self.min_lr) * cosine for base_lr in self.base_lrs]


class WarmupStepScheduler(_LRScheduler):
    """
    带Warmup的阶梯学习率调度器
    
    训练初期线性增长，之后在指定epoch按比例降低
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: warmup阶段epoch数
            milestones: 学习率降低的epoch列表
            gamma: 学习率衰减因子
            last_epoch: 上一个epoch
        """
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        获取当前学习率
        
        Returns:
            各参数组的学习率列表
        """
        epoch = self.last_epoch + 1
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            scale = epoch / float(self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]

        steps = sum(epoch >= milestone for milestone in self.milestones)
        return [base_lr * (self.gamma ** steps) for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    One Cycle学习率策略
    
    先增后减的单周期学习率策略，
    通常能获得更好的性能
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        初始化One Cycle调度器
        
        Args:
            optimizer: 优化器
            max_lr: 最大学习率
            total_steps: 总训练步数
            pct_start: 上升阶段占比
            div_factor: 初始学习率 = max_lr / div_factor
            final_div_factor: 最终学习率 = max_lr / final_div_factor
            last_epoch: 上一个epoch
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """
        获取当前学习率
        
        Returns:
            各参数组的学习率列表
        """
        step = self.last_epoch + 1
        step = min(step, self.total_steps)

        if isinstance(self.max_lr, (list, tuple)):
            max_lrs = list(self.max_lr)
        else:
            max_lrs = [self.max_lr for _ in self.base_lrs]

        warmup_steps = max(1, int(self.pct_start * self.total_steps))
        lrs = []
        for max_lr in max_lrs:
            initial_lr = max_lr / self.div_factor
            final_lr = max_lr / self.final_div_factor

            if step <= warmup_steps:
                pct = step / float(warmup_steps)
                lr = initial_lr + pct * (max_lr - initial_lr)
            else:
                pct = (step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                lr = max_lr - pct * (max_lr - final_lr)
            lrs.append(lr)

        return lrs


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    **kwargs
) -> _LRScheduler:
    """
    创建学习率调度器的工厂函数
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'onecycle', 'plateau')
        **kwargs: 调度器参数
        
    Returns:
        学习率调度器实例
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == 'cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 0),
            total_epochs=kwargs['total_epochs'],
            min_lr=kwargs.get('min_lr', 1e-6),
        )
    if scheduler_type == 'step':
        return WarmupStepScheduler(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 0),
            milestones=kwargs.get('milestones', []),
            gamma=kwargs.get('gamma', 0.1),
        )
    if scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs['max_lr'],
            total_steps=kwargs['total_steps'],
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4),
        )
    if scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6),
        )

    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
