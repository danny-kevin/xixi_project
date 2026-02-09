from src.utils.config import Config


def test_apply_training_overrides_returns_new_config():
    from src.tuning.optuna_tuner import apply_training_overrides

    cfg = Config()
    updated = apply_training_overrides(
        cfg,
        {
            "learning_rate": 0.123,
            "batch_size": 64,
        },
    )

    assert updated is not cfg
    assert updated.training.learning_rate == 0.123
    assert updated.training.batch_size == 64
    # Ensure original config is unchanged.
    assert cfg.training.learning_rate != 0.123
    assert cfg.training.batch_size != 64

