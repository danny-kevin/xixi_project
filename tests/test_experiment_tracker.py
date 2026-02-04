import pytest

from src.utils.experiment_tracker import ExperimentTracker, TENSORBOARD_AVAILABLE


def test_log_metrics_skips_none(tmp_path):
    if not TENSORBOARD_AVAILABLE:
        pytest.skip("tensorboard not available")

    tracker = ExperimentTracker(
        experiment_name="test_metrics",
        use_tensorboard=True,
        use_wandb=False,
        log_dir=str(tmp_path),
    )
    try:
        tracker.log_metrics({"mse": 1.0, "crps": None})
    finally:
        if tracker.tensorboard_writer is not None:
            tracker.tensorboard_writer.close()
