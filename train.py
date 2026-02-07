"""
训练脚本
Training Script

负责模型训练的主要逻辑
"""

import torch
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.experiment_tracker import ExperimentTracker
from src.data.pipeline import DataPipeline


def train(
    config: Config,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    use_wandb: bool = False
):
    """
    训练模型
    
    Args:
        config: 配置对象
        device: 计算设备
        checkpoint_path: 检查点路径（用于恢复训练）
        use_wandb: 是否使用Weights & Biases
    """
    logger = setup_logger('train', log_file='logs/train.log')
    
    logger.info('开始训练流程...')
    logger.info(f'配置: {config.experiment_name}')
    
    # 创建实验追踪器
    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        config=config,
        use_wandb=use_wandb,
        log_dir='logs/tensorboard'
    )
    
    try:
        # ==================== 步骤1: 准备数据 ====================
        logger.info('步骤1: 准备数据...')
        
        logger.info(f'  Data dir: {config.data.data_dir}')
        logger.info(f'  Window: {config.data.window_size}')
        logger.info(f'  Horizon: {config.data.prediction_horizon}')

        pipeline = DataPipeline(
            data_dir=config.data.data_dir,
            window_size=config.data.window_size,
            horizon=config.data.prediction_horizon,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            use_single_file=True,
            single_file_name='dataset_US_final.csv',
            date_column='Date',
            target_column=config.data.target_column,
            group_column='State',
            # Keep per-state windowing, but do NOT leak state identity as features.
            one_hot_columns=[],
            normalize=False,
        )

        train_loader, val_loader, test_loader = pipeline.run()
        feature_names = pipeline.get_feature_names()
        config.data.feature_columns = feature_names
        config.model.num_variables = len(feature_names)
        config.model.prediction_horizon = config.data.prediction_horizon

        logger.info('[OK] data ready')
        logger.info(f'  Train samples: {len(train_loader.dataset)}')
        logger.info(f'  Val samples: {len(val_loader.dataset)}')
        logger.info(f'  Test samples: {len(test_loader.dataset)}')
        # ==================== 步骤2: 创建模型 ====================
        logger.info('步骤2: 创建模型...')
        
        from src.models.hybrid_model import ModelFactory
        
        # 使用ModelFactory创建模型
        model = ModelFactory.create_model(config).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'[OK] 模型创建完成')
        logger.info(f'  模型参数量: {num_params:,}')
        logger.info(f'  输入变量数: {config.model.num_variables}')
        logger.info(f'  预测范围: {config.model.prediction_horizon}天')
        
        # ==================== 步骤3: 创建训练器 ====================
        logger.info('步骤3: 创建训练器...')
        
        from src.training import Trainer, TrainingConfig
        from src.training.loss import HybridLoss
        from src.training.scheduler import WarmupCosineScheduler
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        logger.info(f'  优化器: AdamW (lr={config.training.learning_rate})')
        
        # 创建学习率调度器
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr
        )
        logger.info(f'  调度器: WarmupCosine (warmup={config.training.warmup_epochs} epochs)')
        
        # 创建损失函数
        criterion = HybridLoss()
        logger.info(f'  损失函数: HybridLoss (MSE + 时序一致性正则)')
        
        # 创建训练配置
        training_config = TrainingConfig(
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            device=str(device),
            checkpoint_dir=config.training.checkpoint_dir,
            early_stopping_patience=config.training.early_stopping_patience,
            warmup_epochs=config.training.warmup_epochs,
            gradient_clip_val=config.training.gradient_clip_val,
            pretrain_epochs=config.training.pretrain_epochs if config.training.use_pretrain else 0,
            finetune_lr_ratio=config.training.finetune_lr_ratio
        )
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            config=training_config,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion
        )
        
        logger.info('[OK] 训练器创建完成')
        
        # ==================== 步骤4: 训练模型 ====================
        logger.info('步骤4: 开始训练...')
        
        # 如果有检查点，恢复训练
        if checkpoint_path:
            logger.info(f'从检查点恢复: {checkpoint_path}')
            trainer.load_checkpoint(checkpoint_path)
        
        # 训练
        history = trainer.train(train_loader, val_loader)
        
        # 记录训练历史
        tracker.log_metrics(history)
        
        logger.info('[OK] 训练完成')
        
        # ==================== 步骤5: 保存模型 ====================
        logger.info('步骤5: 保存模型...')
        
        checkpoint_dir = Path(config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        trainer.save_checkpoint(
            str(checkpoint_dir / 'best_model.pth'),
            is_best=True
        )
        
        logger.info(f'[OK] 模型已保存到: {checkpoint_dir}')
        
        # ==================== 步骤6: 测试集评估 ====================
        logger.info('步骤6: 在测试集上评估...')
        
        test_loss = trainer.validate(test_loader)
        logger.info(f'测试集损失: {test_loss:.4f}')
        
        tracker.log_metric('test_loss', test_loss)
        
    except Exception as e:
        logger.error(f'训练过程出错: {e}')
        raise
    
    finally:
        # 关闭追踪器
        tracker.finish()
        logger.info('训练流程结束')


if __name__ == '__main__':
    """
    直接运行此脚本的示例
    
    使用方法:
        python train.py
    
    或使用main.py:
        python main.py --mode train --config configs/default_config.yaml
    """
    from src.utils.config import load_config
    from src.utils.device_manager import DeviceManager
    from src.utils.seed import set_seed
    
    # 加载配置
    config = load_config('configs/default_config.yaml')
    
    # 设置种子
    set_seed(config.seed)
    
    # 获取设备
    device_manager = DeviceManager()
    device = device_manager.get_device()
    
    # 训练
    train(config, device, use_wandb=False)
