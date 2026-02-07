"""
PyTorch Dataset and DataLoader helpers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


class EpidemicDataset(Dataset):
    """Dataset for epidemic forecasting."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        transform: Optional[callable] = None,
        state_ids: Optional[np.ndarray] = None,
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.transform = transform
        self.state_ids = state_ids

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]

        x_tensor = torch.FloatTensor(x)
        y_tensor = torch.FloatTensor(y) if isinstance(y, np.ndarray) else torch.FloatTensor([y])

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        if self.state_ids is not None:
            state_id = int(self.state_ids[idx])
            state_tensor = torch.tensor(state_id, dtype=torch.long)
            return x_tensor, y_tensor, state_tensor

        return x_tensor, y_tensor

    def get_feature_dim(self) -> int:
        if self.X.ndim == 3:
            return self.X.shape[2]
        if self.X.ndim == 2:
            return self.X.shape[1]
        return 1

    def get_window_size(self) -> int:
        if self.X.ndim == 3:
            return self.X.shape[1]
        return 1


def create_data_loaders(
    train_dataset: EpidemicDataset,
    val_dataset: EpidemicDataset,
    test_dataset: EpidemicDataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
