from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import Config


def test_save_study_artifacts_writes_expected_files(tmp_path: Path):
    from src.tuning.optuna_tuner import save_study_artifacts

    class _Study:
        best_params = {"learning_rate": 0.001, "batch_size": 32}
        best_value = 1.23

        def trials_dataframe(self):
            return pd.DataFrame([{"number": 0, "value": self.best_value}])

    study_dir = tmp_path / "optuna" / "unit_test_study"
    save_study_artifacts(study_dir, _Study(), Config())

    assert (study_dir / "best_params.json").exists()
    assert (study_dir / "best_config.yaml").exists()
    assert (study_dir / "study_summary.csv").exists()

