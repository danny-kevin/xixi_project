"""
Smoke test runner for train/eval/predict/experiment modes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import yaml


def build_smoke_config(
    base_config: Path,
    work_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    data = yaml.safe_load(base_config.read_text(encoding="utf-8"))

    data["experiment_name"] = "smoke_experiment"

    training = data.setdefault("training", {})
    training["epochs"] = 1
    training["use_pretrain"] = False
    training["pretrain_epochs"] = 0
    training["batch_size"] = 8
    training["num_workers"] = 0
    training["checkpoint_dir"] = str(checkpoint_dir)

    data_cfg = data.setdefault("data", {})
    data_cfg["window_size"] = 7
    data_cfg["prediction_horizon"] = 1

    model_cfg = data.setdefault("model", {})
    model_cfg["tcn_channels"] = [8]
    model_cfg["attention_embed_dim"] = 16
    model_cfg["attention_num_heads"] = 2
    model_cfg["lstm_hidden_size"] = 16
    model_cfg["lstm_num_layers"] = 1
    model_cfg["lstm_bidirectional"] = False
    model_cfg["prediction_horizon"] = 1

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "smoke_config.yaml"
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return config_path


def build_commands(
    python_exe: str,
    config_path: Path,
    device: str,
    output_dir: Path,
    checkpoint_dir: Path,
) -> List[List[str]]:
    checkpoint_path = checkpoint_dir / "best_model.pth"
    base = [
        python_exe,
        "main.py",
        "--config",
        str(config_path),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    return [
        base + ["--mode", "train", "--epochs", "1"],
        base + ["--mode", "eval", "--checkpoint", str(checkpoint_path)],
        base + ["--mode", "predict", "--checkpoint", str(checkpoint_path)],
        base + ["--mode", "experiment", "--epochs", "1"],
    ]


def run_commands(commands: List[List[str]]) -> None:
    for cmd in commands:
        print("[SMOKE] run:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test all modes.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Base config path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="",
        help="Keep artifacts in this directory (optional)",
    )
    args = parser.parse_args()

    base_config = Path(args.config)
    python_exe = sys.executable

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_dir = work_dir / "results"
        checkpoint_dir = work_dir / "checkpoints"
        config_path = build_smoke_config(base_config, work_dir, output_dir, checkpoint_dir)
        commands = build_commands(python_exe, config_path, args.device, output_dir, checkpoint_dir)
        run_commands(commands)
        print(f"[SMOKE] done (artifacts in {work_dir})")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)
        output_dir = work_dir / "results"
        checkpoint_dir = work_dir / "checkpoints"
        config_path = build_smoke_config(base_config, work_dir, output_dir, checkpoint_dir)
        commands = build_commands(python_exe, config_path, args.device, output_dir, checkpoint_dir)
        run_commands(commands)
        print("[SMOKE] done")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
