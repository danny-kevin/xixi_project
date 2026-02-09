from pathlib import Path

import pytest
import torch

from src.utils.config import Config


def test_tune_with_optuna_requires_optuna(monkeypatch, tmp_path: Path):
    import src.tuning.optuna_tuner as tuner

    def _raise():
        raise RuntimeError("Optuna is not installed. Install it with: pip install optuna")

    monkeypatch.setattr(tuner, "require_optuna", _raise)

    with pytest.raises(RuntimeError, match="Optuna is not installed"):
        tuner.tune_with_optuna(
            config=Config(),
            device=torch.device("cpu"),
            output_dir=tmp_path,
            n_trials=1,
            timeout=1,
            study_name="unit_test_study",
            storage=None,
            sampler="tpe",
            pruner="median",
            num_workers=0,
        )

