"""
Optuna-based hyperparameter tuning utilities.

This module intentionally avoids importing optuna at import-time so that the rest
of the project can be used without the optional dependency installed.
"""

from __future__ import annotations

import copy
import json
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from src.utils.config import Config


def default_storage_uri(output_dir: Path, study_name: str) -> str:
    """
    Build a default Optuna storage URI under:
      <output_dir>/optuna/<study_name>/study.db

    Returns a `sqlite:///...` URI with a POSIX-style path, which Optuna accepts
    on Windows (e.g. `sqlite:///C:/path/to/study.db`).
    """
    base = Path(output_dir) / "optuna" / study_name
    db_path = (base / "study.db").resolve()
    # Optuna expects forward slashes in sqlite URIs on Windows.
    return f"sqlite:///{db_path.as_posix()}"


def apply_training_overrides(config: Config, overrides: Mapping[str, Any]) -> Config:
    """
    Return a copy of `config` with `config.training.<field>` updated from `overrides`.
    Unknown keys are ignored.
    """
    updated = copy.deepcopy(config)
    for key, value in overrides.items():
        if hasattr(updated.training, key):
            setattr(updated.training, key, value)
    return updated


def resolve_storage_uri(
    storage: str | None,
    output_dir: Path,
    study_name: str,
) -> str:
    """
    Resolve Optuna storage URI.

    If `storage` is provided, returns it as-is. Otherwise returns `default_storage_uri(...)`.
    """
    if storage:
        return storage
    return default_storage_uri(output_dir, study_name)


def require_optuna():
    """
    Import Optuna and return the module.

    Raises a friendly RuntimeError if Optuna is not installed.
    """
    try:
        return import_module("optuna")
    except ImportError as e:
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna") from e


def tune_with_optuna(
    *,
    config: Config,
    device,
    output_dir: Path,
    n_trials: int = 50,
    timeout: int | None = None,
    study_name: str = "optuna_study",
    storage: str | None = None,
    sampler: str = "tpe",
    pruner: str = "median",
    num_workers: int = 0,
):
    """
    Run Optuna hyperparameter tuning to minimize validation loss.

    Notes:
    - Optuna is imported lazily via `require_optuna()`.
    - This function is implemented in a way that keeps module import working even
      without Optuna installed.
    """
    optuna = require_optuna()

    # Local imports: keep module import light and keep Optuna optional.
    import torch

    from src.data.dataset import create_data_loaders
    from src.data.pipeline import DataPipeline
    from src.models.hybrid_model import ModelFactory
    from src.training import Trainer, TrainingConfig
    from src.training.loss import HybridLoss
    from src.training.scheduler import WarmupCosineScheduler
    from src.utils.seed import set_seed

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study_dir = output_dir / "optuna" / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    storage_uri = resolve_storage_uri(storage, output_dir, study_name)

    sampler_name = (sampler or "tpe").lower()
    if sampler_name == "tpe":
        sampler_obj = optuna.samplers.TPESampler(seed=int(config.seed))
    elif sampler_name == "random":
        sampler_obj = optuna.samplers.RandomSampler(seed=int(config.seed))
    else:
        raise ValueError(f"Unsupported sampler: {sampler!r}")

    pruner_name = (pruner or "median").lower()
    if pruner_name == "median":
        pruner_obj = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1,
        )
    elif pruner_name == "none":
        pruner_obj = optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unsupported pruner: {pruner!r}")

    # Prepare data once; rebuild DataLoaders per trial (batch_size).
    base_config = copy.deepcopy(config)
    pipeline = DataPipeline(
        data_dir=base_config.data.data_dir,
        window_size=base_config.data.window_size,
        horizon=base_config.data.prediction_horizon,
        train_ratio=base_config.data.train_ratio,
        val_ratio=base_config.data.val_ratio,
        batch_size=base_config.training.batch_size,
        num_workers=base_config.training.num_workers,
        use_single_file=True,
        single_file_name="dataset_US_final.csv",
        date_column="Date",
        target_column=base_config.data.target_column,
        group_column="State",
        # Keep per-state windowing, but do NOT leak state identity as features.
        one_hot_columns=[],
        normalize=False,
    )

    df = pipeline.load_data()
    processed = pipeline.preprocess_data(df, normalize=False)
    train_dataset, val_dataset, test_dataset = pipeline.create_datasets(processed)
    feature_names = pipeline.get_feature_names()

    base_config.data.feature_columns = feature_names
    base_config.model.num_variables = len(feature_names)
    base_config.model.prediction_horizon = base_config.data.prediction_horizon

    def _build_loaders(batch_size: int):
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=base_config.training.num_workers,
        )
        return train_loader, val_loader, test_loader

    def objective(trial):
        # Make trials comparable and reproducible.
        set_seed(int(base_config.seed))

        epochs = int(base_config.training.epochs)
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        warmup_epochs = trial.suggest_int("warmup_epochs", 0, min(10, max(0, epochs)))
        patience = trial.suggest_int("early_stopping_patience", 5, 30)
        clip_val = trial.suggest_float("gradient_clip_val", 0.5, 5.0)

        min_lr_lower = max(1e-8, lr * 1e-4)
        min_lr_upper = max(min_lr_lower, lr * 0.1)
        if min_lr_upper == min_lr_lower:
            min_lr = min_lr_lower
        else:
            min_lr = trial.suggest_float("min_lr", min_lr_lower, min_lr_upper, log=True)

        trial_cfg = apply_training_overrides(
            base_config,
            {
                "learning_rate": float(lr),
                "weight_decay": float(weight_decay),
                "batch_size": int(batch_size),
                "warmup_epochs": int(warmup_epochs),
                "min_lr": float(min_lr),
                "early_stopping_patience": int(patience),
                "gradient_clip_val": float(clip_val),
            },
        )

        trial_ckpt_dir = study_dir / "trials" / f"trial_{trial.number:04d}" / "checkpoints"
        trial_ckpt_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg.training.checkpoint_dir = str(trial_ckpt_dir)

        train_loader, val_loader, _ = _build_loaders(int(batch_size))

        model = ModelFactory.create_model(trial_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(warmup_epochs),
            total_epochs=epochs,
            min_lr=float(min_lr),
        )
        criterion = HybridLoss()

        training_cfg = TrainingConfig(
            epochs=epochs,
            batch_size=int(batch_size),
            learning_rate=float(lr),
            weight_decay=float(weight_decay),
            early_stopping_patience=int(patience),
            early_stopping_min_delta=float(trial_cfg.training.early_stopping_min_delta),
            warmup_epochs=int(warmup_epochs),
            pretrain_epochs=int(trial_cfg.training.pretrain_epochs)
            if bool(trial_cfg.training.use_pretrain)
            else 0,
            finetune_lr_ratio=float(trial_cfg.training.finetune_lr_ratio),
            checkpoint_dir=str(trial_ckpt_dir),
            log_interval=int(trial_cfg.training.log_interval),
            save_best_only=bool(trial_cfg.training.save_best_only),
            device=str(device),
            num_workers=int(trial_cfg.training.num_workers),
            gradient_clip_val=float(clip_val),
        )
        # Not part of the dataclass schema, but used by Trainer if it creates schedulers.
        setattr(training_cfg, "min_lr", float(min_lr))

        def _optuna_callback(trainer: Trainer, epoch: int, metrics: Mapping[str, Any]):
            val_loss = float(metrics.get("val_loss", 0.0))
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trainer = Trainer(
            model=model,
            config=training_cfg,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
        )

        trainer.train(train_loader, val_loader, callbacks=[_optuna_callback])
        best = trainer.best_val_loss
        if best == float("inf") and trainer.training_history.get("val_loss"):
            best = float(min(trainer.training_history["val_loss"]))

        # Try to keep memory usage stable between trials.
        del trainer, model, optimizer, scheduler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return float(best)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_uri,
        load_if_exists=True,
        sampler=sampler_obj,
        pruner=pruner_obj,
    )

    n_jobs = int(num_workers) if int(num_workers) > 1 else 1
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=int(timeout) if timeout is not None else None,
        n_jobs=n_jobs,
        gc_after_trial=True,
    )

    save_study_artifacts(study_dir, study, base_config)

    return {
        "study_dir": str(study_dir),
        "storage": storage_uri,
        "best_value": float(getattr(study, "best_value", float("nan"))),
        "best_params": dict(getattr(study, "best_params", {}) or {}),
    }


def save_study_artifacts(study_dir: Path, study, base_config: Config) -> None:
    """
    Save best params/config and a trial summary CSV for an Optuna study.

    The `study` object is treated duck-typed to keep this function unit-testable
    without Optuna installed.
    """
    from src.utils.config import save_config

    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    best_params = dict(getattr(study, "best_params", {}) or {})
    best_value = getattr(study, "best_value", None)

    payload = {
        "best_value": float(best_value) if best_value is not None else None,
        "best_params": best_params,
    }
    (study_dir / "best_params.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    best_config = apply_training_overrides(base_config, best_params)
    save_config(best_config, study_dir / "best_config.yaml")

    # Best-effort: write trial summary if available.
    summary_path = study_dir / "study_summary.csv"
    try:
        df = study.trials_dataframe()
        df.to_csv(summary_path, index=False)
    except Exception:
        summary_path.write_text("number,value\n", encoding="utf-8")

