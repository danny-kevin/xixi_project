"""
Main Entry Point

Provide CLI interface to run training/evaluation/prediction/experiment.
"""

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.utils.device_manager import DeviceManager


def parse_args():
    parser = argparse.ArgumentParser(
        description='Attention-Enhanced M-TCN-LSTM Epidemic Forecasting Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--config', type=str, default='configs/default_config.yaml', help='Config file path')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict', 'experiment'], default='train')

    parser.add_argument('--epochs', type=int, help='Training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')

    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (resume/eval)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO', help='Log level')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')

    return parser.parse_args()


def _unpack_batch(batch):
    if len(batch) == 3:
        inputs, targets, state_ids = batch
        return inputs, targets, state_ids
    inputs, targets = batch
    return inputs, targets, None


def main():
    args = parse_args()

    logger = setup_logger('main', log_file='logs/main.log', level=args.log_level)

    logger.info('=' * 80)
    logger.info('Attention-Enhanced M-TCN-LSTM Epidemic Forecasting Model')
    logger.info('=' * 80)

    logger.info(f'Loading config: {args.config}')
    config = load_config(args.config)

    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.training.device = args.device
    if args.seed:
        config.seed = args.seed

    logger.info(f'Setting seed: {config.seed}')
    set_seed(config.seed)

    device_manager = DeviceManager()
    device = device_manager.get_device(config.training.device)
    logger.info(f'Using device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Output dir: {output_dir}')

    if args.mode == 'train':
        logger.info('Mode: train')
        from train import train

        train(config, device, args.checkpoint, args.use_wandb)

    elif args.mode == 'eval':
        logger.info('Mode: eval')
        if not args.checkpoint:
            logger.error('Eval mode requires --checkpoint')
            sys.exit(1)

        try:
            import torch
            from src.models.hybrid_model import ModelFactory
            from src.evaluation.metrics import RegressionMetrics
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
            _, _, test_loader = loader.create_dataloaders(
                data_dict,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
            )

            model = ModelFactory.create_model(config).to(device)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Model loaded from {args.checkpoint}')

            model.eval()
            all_predictions = []
            all_targets = []
            all_state_ids = []

            with torch.no_grad():
                for batch in test_loader:
                    inputs, targets, state_ids = _unpack_batch(batch)
                    inputs = inputs.to(device)
                    if state_ids is not None:
                        state_ids = state_ids.to(device)
                        predictions = model(inputs, state_ids=state_ids)
                    else:
                        predictions = model(inputs)

                    if predictions.dim() == 3 and predictions.size(-1) == 1:
                        predictions = predictions.squeeze(-1)
                    if targets.dim() == 3 and targets.size(-1) == 1:
                        targets = targets.squeeze(-1)

                    all_predictions.append(predictions.cpu())
                    all_targets.append(targets.cpu())
                    if state_ids is not None:
                        all_state_ids.append(state_ids.cpu())

            predictions_tensor = torch.cat(all_predictions, dim=0)
            targets_tensor = torch.cat(all_targets, dim=0)

            pred_np = predictions_tensor.numpy()
            target_np = targets_tensor.numpy()

            if config.data.inverse_transform and data_dict.get('state_scalers'):
                state_ids_np = torch.cat(all_state_ids, dim=0).numpy() if all_state_ids else None
                pred_np = loader.inverse_transform_target(
                    pred_np,
                    state_ids_np,
                    data_dict.get('id_to_state'),
                    data_dict.get('state_scalers'),
                    config.data.target_column,
                    data_dict.get('target_log1p', False),
                )
                target_np = loader.inverse_transform_target(
                    target_np,
                    state_ids_np,
                    data_dict.get('id_to_state'),
                    data_dict.get('state_scalers'),
                    config.data.target_column,
                    data_dict.get('target_log1p', False),
                )

            metrics = RegressionMetrics.compute_all(pred_np, target_np)

            logger.info('Evaluation results:')
            logger.info(f'  MSE:  {metrics.mse:.4f}')
            logger.info(f'  RMSE: {metrics.rmse:.4f}')
            logger.info(f'  MAE:  {metrics.mae:.4f}')
            logger.info(f'  MAPE: {metrics.mape:.2f}%')
            logger.info(f'  R2:   {metrics.r2:.4f}')

        except Exception as e:
            logger.error(f'Eval failed: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.mode == 'predict':
        logger.info('Mode: predict')
        if not args.checkpoint:
            logger.error('Predict mode requires --checkpoint')
            sys.exit(1)

        try:
            import torch
            import pandas as pd
            from src.models.hybrid_model import ModelFactory
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
            _, _, test_loader = loader.create_dataloaders(
                data_dict,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
            )

            model = ModelFactory.create_model(config).to(device)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f'Model loaded from {args.checkpoint}')

            model.eval()
            preds = []
            targets = []
            state_ids_all = []

            with torch.no_grad():
                for batch in test_loader:
                    inputs, y, state_ids = _unpack_batch(batch)
                    inputs = inputs.to(device)
                    if state_ids is not None:
                        state_ids = state_ids.to(device)
                        output = model(inputs, state_ids=state_ids)
                    else:
                        output = model(inputs)

                    if output.dim() == 3 and output.size(-1) == 1:
                        output = output.squeeze(-1)
                    if y.dim() == 3 and y.size(-1) == 1:
                        y = y.squeeze(-1)

                    preds.append(output.cpu())
                    targets.append(y.cpu())
                    if state_ids is not None:
                        state_ids_all.append(state_ids.cpu())

            pred_tensor = torch.cat(preds, dim=0)
            target_tensor = torch.cat(targets, dim=0)

            pred_np = pred_tensor.numpy()
            target_np = target_tensor.numpy()

            state_ids_np = torch.cat(state_ids_all, dim=0).numpy() if state_ids_all else None

            if config.data.inverse_transform and data_dict.get('state_scalers'):
                pred_np = loader.inverse_transform_target(
                    pred_np,
                    state_ids_np,
                    data_dict.get('id_to_state'),
                    data_dict.get('state_scalers'),
                    config.data.target_column,
                    data_dict.get('target_log1p', False),
                )
                target_np = loader.inverse_transform_target(
                    target_np,
                    state_ids_np,
                    data_dict.get('id_to_state'),
                    data_dict.get('state_scalers'),
                    config.data.target_column,
                    data_dict.get('target_log1p', False),
                )

            horizon = pred_np.shape[1] if pred_np.ndim > 1 else 1
            data = {}
            if state_ids_np is not None:
                data['state_id'] = state_ids_np
                id_to_state = data_dict.get('id_to_state') or {}
                data['state'] = [id_to_state.get(int(s), '') for s in state_ids_np]

            for i in range(horizon):
                data[f'y_true_t{i+1}'] = target_np[:, i] if target_np.ndim > 1 else target_np
                data[f'y_pred_t{i+1}'] = pred_np[:, i] if pred_np.ndim > 1 else pred_np

            out_path = output_dir / 'predictions.csv'
            pd.DataFrame(data).to_csv(out_path, index=False)
            logger.info(f'Predictions saved: {out_path}')

        except Exception as e:
            logger.error(f'Predict failed: {e}')
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif args.mode == 'experiment':
        logger.info('Mode: experiment')
        from run_experiment import run_experiment
        run_experiment(config, device, args.use_wandb)

    logger.info('=' * 80)
    logger.info('Task completed')
    logger.info('=' * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nUser interrupted')
        sys.exit(0)
    except Exception as e:
        print(f'\nError: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
