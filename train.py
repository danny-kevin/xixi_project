"""
Training script.
"""

import torch
from pathlib import Path
from typing import Optional

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.utils.experiment_tracker import ExperimentTracker


def train(
    config: Config,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    use_wandb: bool = False,
):
    logger = setup_logger('train', log_file='logs/train.log')

    logger.info('Starting training...')
    logger.info(f'Experiment: {config.experiment_name}')

    tracker = ExperimentTracker(
        experiment_name=config.experiment_name,
        config=config,
        use_wandb=use_wandb,
        log_dir='logs/tensorboard',
    )

    try:
        logger.info('Step 1: Prepare data...')
        from load_us_data import USCovidDataLoader

        data_path = config.data.processed_csv or f"{config.data.raw_dir}/dataset_US_final.csv"
        loader = USCovidDataLoader(data_path=data_path)

        data_dict = loader.prepare_data(
            target_column=config.data.target_column,
            feature_columns=config.data.feature_columns,
            window_size=config.data.window_size,
            horizon=config.data.prediction_horizon,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            scaler_type=config.data.scaler_type,
            per_state_normalize=config.data.per_state_normalize,
            target_log1p=config.data.target_log1p,
            state_column=config.data.state_column,
        )

        train_loader, val_loader, test_loader = loader.create_dataloaders(
            data_dict,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
        )

        logger.info('Data prepared')
        logger.info(f'Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches')
        logger.info(f'Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches')
        logger.info(f'Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches')

        logger.info('Step 2: Build model...')
        from src.models.hybrid_model import ModelFactory

        model = ModelFactory.create_model(config).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Model params: {num_params:,}')
        logger.info(f'Num variables: {config.model.num_variables}')
        logger.info(f'Prediction horizon: {config.model.prediction_horizon}')

        logger.info('Step 3: Create trainer...')
        from src.training import Trainer, TrainingConfig
        import torch.nn as nn
        from src.training.scheduler import WarmupCosineScheduler

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        logger.info(f'Optimizer: AdamW (lr={config.training.learning_rate})')

        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr,
        )
        logger.info(f'Scheduler: WarmupCosine (warmup={config.training.warmup_epochs} epochs)')

        criterion = nn.MSELoss()
        logger.info('Loss: MSE')

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
            finetune_lr_ratio=config.training.finetune_lr_ratio,
        )

        trainer = Trainer(
            model=model,
            config=training_config,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )

        logger.info('Step 4: Train...')
        if checkpoint_path:
            logger.info(f'Resume from checkpoint: {checkpoint_path}')
            trainer.load_checkpoint(checkpoint_path)

        history = trainer.train(train_loader, val_loader)
        tracker.log_metrics(history)
        logger.info('Training complete')

        logger.info('Step 5: Save model...')
        checkpoint_dir = Path(config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_dir / 'best_model.pth'), is_best=True)
        logger.info(f'Model saved to: {checkpoint_dir}')

        logger.info('Step 6: Evaluate on test set...')
        test_loss = trainer.validate(test_loader)
        logger.info(f'Test loss: {test_loss:.4f}')
        tracker.log_metric('test_loss', test_loss)

    except Exception as e:
        logger.error(f'Training failed: {e}')
        raise
    finally:
        tracker.finish()
        logger.info('Training finished')


if __name__ == '__main__':
    from src.utils.config import load_config
    from src.utils.device_manager import DeviceManager
    from src.utils.seed import set_seed

    config = load_config('configs/default_config.yaml')
    set_seed(config.seed)

    device_manager = DeviceManager()
    device = device_manager.get_device()

    train(config, device, use_wandb=False)

