"""
模型检查点管理
Checkpoint Manager

提供模型保存和加载的统一接口
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict] = None,
    **kwargs
) -> None:
    """
    保存模型检查点
    
    Args:
        model: 模型
        path: 保存路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        epoch: 当前epoch（可选）
        metrics: 评估指标（可选）
        **kwargs: 其他要保存的内容
    """
    # 创建保存目录
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建检查点字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # 添加额外参数
    checkpoint.update(kwargs)
    
    # 保存
    torch.save(checkpoint, path)
    logger.info(f'✅ 检查点已保存: {path}')


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict:
    """
    加载模型检查点
    
    Args:
        model: 模型
        path: 检查点路径
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 目标设备（可选）
        strict: 是否严格匹配模型参数
        
    Returns:
        检查点字典（包含epoch、metrics等信息）
    """
    # 检查文件是否存在
    if not Path(path).exists():
        raise FileNotFoundError(f'检查点文件不存在: {path}')
    
    # 加载检查点
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    logger.info(f'✅ 模型参数已加载: {path}')
    
    # 加载优化器参数
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info('✅ 优化器参数已加载')
    
    # 加载调度器参数
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info('✅ 调度器参数已加载')
    
    # 返回其他信息
    info = {}
    if 'epoch' in checkpoint:
        info['epoch'] = checkpoint['epoch']
        logger.info(f'检查点epoch: {info["epoch"]}')
    
    if 'metrics' in checkpoint:
        info['metrics'] = checkpoint['metrics']
        logger.info(f'检查点指标: {info["metrics"]}')
    
    return info


def save_model_only(model: nn.Module, path: str) -> None:
    """
    仅保存模型参数（不包含优化器等）
    
    Args:
        model: 模型
        path: 保存路径
    """
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), path)
    logger.info(f'✅ 模型参数已保存: {path}')


def load_model_only(
    model: nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> None:
    """
    仅加载模型参数
    
    Args:
        model: 模型
        path: 检查点路径
        device: 目标设备
        strict: 是否严格匹配
    """
    if not Path(path).exists():
        raise FileNotFoundError(f'模型文件不存在: {path}')
    
    if device is None:
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=device)
    
    model.load_state_dict(state_dict, strict=strict)
    logger.info(f'✅ 模型参数已加载: {path}')


class CheckpointManager:
    """
    检查点管理器
    
    自动管理模型检查点的保存和清理
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点目录
            max_checkpoints: 最多保留的检查点数量
            save_best_only: 是否只保存最佳模型
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float('inf')
        
        logger.info(f'检查点目录: {self.checkpoint_dir}')
    
    def save(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        is_best: bool = False
    ) -> None:
        """
        保存检查点
        
        Args:
            model: 模型
            epoch: 当前epoch
            metric: 评估指标（如验证损失）
            optimizer: 优化器
            scheduler: 调度器
            is_best: 是否为最佳模型
        """
        # 如果只保存最佳模型且当前不是最佳，则跳过
        if self.save_best_only and not is_best:
            return
        
        # 保存当前检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        save_checkpoint(
            model, str(checkpoint_path),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={'metric': metric}
        )
        
        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            save_checkpoint(
                model, str(best_path),
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={'metric': metric}
            )
            self.best_metric = metric
            logger.info(f'🏆 新的最佳模型! 指标: {metric:.4f}')
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self) -> None:
        """清理旧的检查点文件"""
        # 获取所有检查点文件
        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_epoch_*.pth'),
            key=lambda p: p.stat().st_mtime
        )
        
        # 保留最新的max_checkpoints个
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                old_checkpoint.unlink()
                logger.debug(f'删除旧检查点: {old_checkpoint.name}')
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None
    ) -> Dict:
        """
        加载最佳模型
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 调度器
            device: 目标设备
            
        Returns:
            检查点信息
        """
        best_path = self.checkpoint_dir / 'best_model.pth'
        return load_checkpoint(
            model, str(best_path),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )


# 使用示例
if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建示例模型
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 示例1: 基本保存和加载
    print('示例1: 基本保存和加载')
    save_checkpoint(
        model, 'checkpoints/test.pth',
        optimizer=optimizer,
        epoch=10,
        metrics={'loss': 0.5}
    )
    
    # 创建新模型并加载
    new_model = nn.Linear(10, 5)
    new_optimizer = torch.optim.Adam(new_model.parameters())
    info = load_checkpoint(new_model, 'checkpoints/test.pth', optimizer=new_optimizer)
    print(f'加载的信息: {info}')
    
    # 示例2: 使用检查点管理器
    print('\n示例2: 使用检查点管理器')
    manager = CheckpointManager('checkpoints/managed', max_checkpoints=3)
    
    for epoch in range(5):
        metric = 1.0 / (epoch + 1)  # 模拟递减的损失
        is_best = metric < manager.best_metric
        manager.save(model, epoch, metric, optimizer, is_best=is_best)
    
    # 加载最佳模型
    info = manager.load_best(new_model, new_optimizer)
    print(f'最佳模型信息: {info}')
    
    print('\n✅ 检查点管理测试完成')
