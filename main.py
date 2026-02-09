"""
Main Entry Point

Unified CLI for train/eval/predict/experiment.
"""

import argparse
import sys
from pathlib import Path

import torch

from src.utils.config import Config, load_config
from src.utils.device_manager import DeviceManager
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attention-Enhanced M-TCN-LSTM epidemic forecasting model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "predict", "experiment", "tune"],
        default="train",
        help="Run mode",
    )

    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (required for eval/predict, optional for train resume)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (used for logs + some artifacts)",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb tracking")

    # Optuna tuning (mode: tune)
    parser.add_argument("--tune-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--tune-timeout", type=int, default=None, help="Optuna timeout (seconds)")
    parser.add_argument(
        "--tune-study-name",
        type=str,
        default="optuna_study",
        help="Optuna study name",
    )
    parser.add_argument(
        "--tune-storage",
        type=str,
        default=None,
        help="Optuna storage URI (e.g. sqlite:///.../study.db)",
    )
    parser.add_argument(
        "--tune-sampler",
        type=str,
        choices=["tpe", "random"],
        default="tpe",
        help="Optuna sampler",
    )
    parser.add_argument(
        "--tune-pruner",
        type=str,
        choices=["median", "none"],
        default="median",
        help="Optuna pruner",
    )
    parser.add_argument(
        "--tune-num-workers",
        type=int,
        default=0,
        help="Number of parallel Optuna jobs (0/1 is safest on Windows)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "main",
        log_file=str(output_dir / "logs" / "main.log"),
        level=args.log_level,
    )

    logger.info("=" * 80)
    logger.info("Attention-Enhanced M-TCN-LSTM epidemic forecasting model")
    logger.info("=" * 80)

    logger.info(f"Config: {args.config}")
    config: Config = load_config(args.config)

    # CLI overrides: use explicit None checks (0 is a valid value for some args).
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.seed = args.seed

    logger.info(f"Seed: {config.seed}")
    set_seed(config.seed)

    device_manager = DeviceManager()
    device = device_manager.get_device(config.training.device)
    logger.info(f"Device: {device}")
    logger.info(f"Output dir: {output_dir}")

    if args.mode == "train":
        logger.info("Mode: train")
        from train import train

        train(config, device, args.checkpoint, args.use_wandb)
        return

    if args.mode == "eval":
        logger.info("Mode: eval")
        if not args.checkpoint:
            logger.error("Eval mode requires --checkpoint")
            sys.exit(1)

        from src.data.pipeline import DataPipeline
        from src.evaluation.metrics import RegressionMetrics
        from src.models.hybrid_model import ModelFactory

        pipeline = DataPipeline(
            data_dir=config.data.data_dir,
            window_size=config.data.window_size,
            horizon=config.data.prediction_horizon,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            use_single_file=True,
            single_file_name="dataset_US_final.csv",
            date_column="Date",
            target_column=config.data.target_column,
            group_column="State",
            # Keep per-state windowing, but do NOT leak state identity as features.
            one_hot_columns=[],
            normalize=False,
        )

        _, _, test_loader = pipeline.run()
        feature_names = pipeline.get_feature_names()
        config.data.feature_columns = feature_names
        config.model.num_variables = len(feature_names)
        config.model.prediction_horizon = config.data.prediction_horizon

        model = ModelFactory.create_model(config).to(device)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[OK] loaded checkpoint: {args.checkpoint}")

        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                predictions = model(inputs)

                if predictions.dim() == 3 and predictions.size(-1) == 1:
                    predictions = predictions.squeeze(-1)
                if targets.dim() == 3 and targets.size(-1) == 1:
                    targets = targets.squeeze(-1)

                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)

        metrics = RegressionMetrics.compute_all(
            predictions_tensor.numpy(),
            targets_tensor.numpy(),
        )

        logger.info("Eval results:")
        logger.info(f"  MSE:  {metrics.mse:.4f}")
        logger.info(f"  RMSE: {metrics.rmse:.4f}")
        logger.info(f"  MAE:  {metrics.mae:.4f}")
        logger.info(f"  MAPE: {metrics.mape:.2f}%")
        logger.info(f"  R2:   {metrics.r2:.4f}")
        return

    if args.mode == "predict":
        logger.info("Mode: predict")
        if not args.checkpoint:
            logger.error("Predict mode requires --checkpoint")
            sys.exit(1)

        from src.data.pipeline import DataPipeline
        from src.models.hybrid_model import ModelFactory

        pipeline = DataPipeline(
            data_dir=config.data.data_dir,
            window_size=config.data.window_size,
            horizon=config.data.prediction_horizon,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            use_single_file=True,
            single_file_name="dataset_US_final.csv",
            date_column="Date",
            target_column=config.data.target_column,
            group_column="State",
            # Keep per-state windowing, but do NOT leak state identity as features.
            one_hot_columns=[],
            normalize=False,
        )

        df = pipeline.load_data()
        _ = pipeline.preprocess_data(df)
        feature_names = pipeline.get_feature_names()
        config.data.feature_columns = feature_names
        config.model.num_variables = len(feature_names)
        config.model.prediction_horizon = config.data.prediction_horizon

        model = ModelFactory.create_model(config).to(device)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"[OK] loaded checkpoint: {args.checkpoint}")

        model.eval()
        logger.info("[OK] model is ready for prediction")
        logger.info("Use AttentionMTCNLSTM.predict() for forecasts")
        return

    if args.mode == "experiment":
        logger.info("Mode: experiment")
        from run_experiment import run_experiment

        run_experiment(config, device, args.use_wandb)
        return

    if args.mode == "tune":
        logger.info("Mode: tune")
        from src.tuning.optuna_tuner import tune_with_optuna

        result = tune_with_optuna(
            config=config,
            device=device,
            output_dir=output_dir,
            n_trials=args.tune_trials,
            timeout=args.tune_timeout,
            study_name=args.tune_study_name,
            storage=args.tune_storage,
            sampler=args.tune_sampler,
            pruner=args.tune_pruner,
            num_workers=args.tune_num_workers,
        )

        logger.info("[OK] tuning complete")
        logger.info(f"  Best val_loss: {result.get('best_value')}")
        logger.info(f"  Study dir: {result.get('study_dir')}")
        return

    raise RuntimeError(f"Unexpected mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
