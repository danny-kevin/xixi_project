import importlib.util
from pathlib import Path

import yaml


def _load_module(script_path: Path):
    assert script_path.exists(), f"Missing script: {script_path}"
    spec = importlib.util.spec_from_file_location("smoke_all_modes", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_smoke_config_overrides(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "smoke_all_modes.py"
    module = _load_module(script_path)

    base_config = repo_root / "configs" / "default_config.yaml"
    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"

    config_path = module.build_smoke_config(
        base_config=base_config,
        work_dir=tmp_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["experiment_name"] == "smoke_experiment"
    assert data["training"]["epochs"] == 1
    assert data["training"]["use_pretrain"] is False
    assert data["training"]["pretrain_epochs"] == 0
    assert data["training"]["batch_size"] == 8
    assert data["training"]["num_workers"] == 0
    assert data["training"]["checkpoint_dir"] == str(checkpoint_dir)
    assert data["data"]["window_size"] == 7
    assert data["data"]["prediction_horizon"] == 1
    assert data["model"]["tcn_channels"] == [8]
    assert data["model"]["lstm_hidden_size"] == 16


def test_build_commands_includes_modes(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "smoke_all_modes.py"
    module = _load_module(script_path)

    config_path = tmp_path / "smoke.yaml"
    config_path.write_text("seed: 1\n", encoding="utf-8")

    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    commands = module.build_commands(
        python_exe="python",
        config_path=config_path,
        device="cpu",
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )

    assert len(commands) == 4
    modes = [cmd[cmd.index("--mode") + 1] for cmd in commands]
    assert modes == ["train", "eval", "predict", "experiment"]

    for cmd in commands:
        assert "--config" in cmd
        assert str(config_path) in cmd
        assert "--device" in cmd
        assert "cpu" in cmd

    checkpoint_path = str(checkpoint_dir / "best_model.pth")
    assert checkpoint_path in commands[1]
    assert checkpoint_path in commands[2]
