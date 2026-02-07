"""
训练器模块
Trainer Module

负责模型训练流程的管理和执行
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """
    训练配置
    
    包含所有训练相关的超参数
    """
    # 基础训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # 早停参数
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # 学习率调度
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau'
    warmup_epochs: int = 5
    
    # 多阶段训练
    pretrain_epochs: int = 20  # 预训练epoch数
    finetune_lr_ratio: float = 0.1  # 微调学习率比例
    
    # 保存与日志
    checkpoint_dir: str = 'checkpoints'
    log_interval: int = 10
    save_best_only: bool = True
    
    # 设备配置
    device: str = 'cuda'
    num_workers: int = 4
    
    # 梯度裁剪
    gradient_clip_val: float = 1.0


class Trainer:
    """
    模型训练器
    
    功能:
    - 完整训练循环管理
    - 多阶段训练 (预训练 + 端到端微调)
    - 早停机制
    - 模型检查点保存
    - 训练日志记录
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion: Optional[nn.Module] = None
    ):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            config: 训练配置
            optimizer: 优化器 (可选，默认使用AdamW)
            scheduler: 学习率调度器 (可选)
            criterion: 损失函数 (可选，默认使用MSE)
        """
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self._optimizer_provided = optimizer is not None
        self._scheduler_provided = scheduler is not None
        self.scheduler_type = (config.scheduler_type or '').lower()
        
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)

        if self.criterion is None:
            self.criterion = nn.MSELoss()

        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        if self.scheduler is None and self.scheduler_type and self.scheduler_type != 'onecycle':
            from .scheduler import create_scheduler

            min_lr = getattr(config, 'min_lr', 1e-6)
            if self.scheduler_type == 'step':
                milestones = [int(config.epochs * 0.5), int(config.epochs * 0.8)]
                self.scheduler = create_scheduler(
                    self.optimizer,
                    scheduler_type='step',
                    warmup_epochs=config.warmup_epochs,
                    milestones=milestones,
                    gamma=0.1,
                )
            elif self.scheduler_type == 'plateau':
                self.scheduler = create_scheduler(
                    self.optimizer,
                    scheduler_type='plateau',
                    min_lr=min_lr,
                    patience=config.early_stopping_patience,
                )
            else:
                self.scheduler = create_scheduler(
                    self.optimizer,
                    scheduler_type='cosine',
                    warmup_epochs=config.warmup_epochs,
                    total_epochs=config.epochs,
                    min_lr=min_lr,
                )
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 3:
            inputs, targets, state_ids = batch
            return inputs, targets, state_ids
        inputs, targets = batch
        return inputs, targets, None
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        执行完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            callbacks: 可选的回调函数列表
            
        Returns:
            训练历史记录
        """
        callbacks = callbacks or []
        self._callbacks = callbacks

        if self.config.pretrain_epochs > 0:
            self.pretrain_modules(train_loader, val_loader)

        self.finetune(train_loader, val_loader)
        return self.training_history
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练单个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            epoch平均训练损失
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")
        return self._train_epoch_with_optimizer(train_loader, self.optimizer)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失
        """
        self.model.eval()
        total_loss = 0.0
        
        # 创建验证进度条 (位置3，保留显示)
        pbar = tqdm(val_loader, desc='  Validation', leave=True, position=2, ncols=100)
        
        with torch.no_grad():
            for batch in pbar:
                inputs, targets, state_ids = self._unpack_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if state_ids is not None:
                    state_ids = state_ids.to(self.device)
                    predictions = self.model(inputs, state_ids=state_ids)
                else:
                    predictions = self.model(inputs)
                predictions, targets = self._align_predictions(predictions, targets)
                loss = self.criterion(predictions, targets)
                total_loss += loss.item()
                
                # 更新进度条显示当前损失
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(1, len(val_loader))

    def _train_epoch_with_optimizer(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        
        # 创建训练进度条 (位置2，保留显示)
        pbar = tqdm(train_loader, desc='  Training', leave=True, position=1, ncols=100)

        for batch in pbar:
            inputs, targets, state_ids = self._unpack_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            optimizer.zero_grad()
            if state_ids is not None:
                state_ids = state_ids.to(self.device)
                predictions = self.model(inputs, state_ids=state_ids)
            else:
                predictions = self.model(inputs)
            predictions, targets = self._align_predictions(predictions, targets)
            loss = self.criterion(predictions, targets)
            loss.backward()

            if self.config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val,
                )
            optimizer.step()
            total_loss += loss.item()
            
            # 更新进度条显示当前损失
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(1, len(train_loader))

    @staticmethod
    def _align_predictions(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if predictions.dim() == 3 and predictions.size(-1) == 1 and targets.dim() == 2:
            predictions = predictions.squeeze(-1)
        if targets.dim() == 3 and targets.size(-1) == 1 and predictions.dim() == 2:
            targets = targets.squeeze(-1)
        return predictions, targets

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _run_fixed_epochs(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        stage_name: str = "Pretraining",
    ) -> None:
        # 添加预训练阶段的进度条
        epoch_pbar = tqdm(range(epochs), desc=f'{stage_name}', position=0, ncols=100)
        
        for epoch in epoch_pbar:
            train_loss = self._train_epoch_with_optimizer(train_loader, optimizer)
            val_loss = self.validate(val_loader)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # 更新进度条显示
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })

            if (epoch + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"{stage_name} epoch {epoch + 1}/{epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )
    
    def pretrain_modules(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """
        预训练各子模块
        
        多阶段训练策略: 先单独预训练TCN、LSTM等子模块
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        if self.config.pretrain_epochs <= 0:
            return

        if not hasattr(self.model, 'mtcn') or not hasattr(self.model, 'lstm'):
            self.logger.warning("Model does not expose mtcn/lstm; skipping pretraining.")
            return

        self.logger.info("Pretrain stage 1: M-TCN")
        self._set_requires_grad(self.model.lstm, False)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._run_fixed_epochs(train_loader, val_loader, optimizer, self.config.pretrain_epochs, "Pretrain M-TCN")
        self._set_requires_grad(self.model.lstm, True)

        self.logger.info("Pretrain stage 2: LSTM")
        self._set_requires_grad(self.model.mtcn, False)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._run_fixed_epochs(train_loader, val_loader, optimizer, self.config.pretrain_epochs, "Pretrain LSTM")
        self._set_requires_grad(self.model.mtcn, True)
    
    def finetune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> None:
        """
        端到端微调
        
        在预训练基础上进行端到端微调
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        self._set_requires_grad(self.model, True)

        if not self._optimizer_provided:
            finetune_lr = self.config.learning_rate * self.config.finetune_lr_ratio
            for group in self.optimizer.param_groups:
                group['lr'] = finetune_lr

            if self.scheduler is not None and hasattr(self.scheduler, 'base_lrs'):
                self.scheduler.base_lrs = [finetune_lr for _ in self.scheduler.base_lrs]

        if self.scheduler is None and self.scheduler_type == 'onecycle':
            from .scheduler import create_scheduler

            max_lr = self.optimizer.param_groups[0]['lr']
            total_steps = self.config.epochs * max(1, len(train_loader))
            self.scheduler = create_scheduler(
                self.optimizer,
                scheduler_type='onecycle',
                max_lr=max_lr,
                total_steps=total_steps,
            )

        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
        )
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取设备信息用于显示
        device_name = str(self.device).upper()
        if 'cuda' in device_name.lower():
            device_info = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "GPU"
        else:
            device_info = "CPU"
        
        # 创建 epoch 级别的进度条 (位置1，主进度条)
        epoch_pbar = tqdm(
            range(self.config.epochs), 
            desc=f'Training Progress [Device: {device_info}]', 
            position=0, 
            ncols=120
        )

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # 更新epoch进度条显示当前指标
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self.best_val_loss - self.config.early_stopping_min_delta:
                self.best_val_loss = val_loss
                if self.config.save_best_only:
                    self.save_checkpoint(str(checkpoint_dir / 'best_model.pth'), is_best=True)

            if not self.config.save_best_only:
                self.save_checkpoint(str(checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'))

            if (epoch + 1) % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )

            callbacks = getattr(self, '_callbacks', [])
            for callback in callbacks:
                callback(self, epoch, {'train_loss': train_loss, 'val_loss': val_loss})

            if early_stopping(val_loss):
                self.logger.info("Early stopping triggered.")
                break
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, str(path))

        if is_best and self.config.save_best_only:
            best_path = path.parent / 'best_model.pth'
            if best_path != path:
                torch.save(checkpoint, str(best_path))
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载模型检查点
        
        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)


class EarlyStopping:
    """
    早停机制
    
    当验证损失在指定epoch数内没有改善时停止训练
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        """
        初始化早停机制
        
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该停止训练
        
        Args:
            val_loss: 当前验证损失
            
        Returns:
            是否应该停止
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
