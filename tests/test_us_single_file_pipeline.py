import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pipeline import DataPipeline


def test_us_single_file_pipeline_state_one_hot_and_windows():
    data_dir = REPO_ROOT / "data"
    pipeline = DataPipeline(
        data_dir=str(data_dir),
        window_size=7,
        horizon=3,
        batch_size=8,
        num_workers=0,
        use_single_file=True,
        single_file_name="dataset_US_final.csv",
        target_column="Confirmed",
        date_column="Date",
        group_column="State",
        one_hot_columns=["State"],
        normalize=False,
    )

    train_loader, _, _ = pipeline.run()
    feature_names = pipeline.get_feature_names()
    state_cols = [i for i, name in enumerate(feature_names) if name.startswith("State_")]
    assert state_cols, "State one-hot columns missing"

    batch_x, _ = next(iter(train_loader))
    assert batch_x.dim() == 3

    state_block = batch_x[:, :, state_cols]
    diff = state_block - state_block[:, :1, :]
    assert torch.allclose(diff, torch.zeros_like(diff))
