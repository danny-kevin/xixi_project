"""
Time series cross validation utilities.
"""

from typing import Iterable, Dict, Tuple


class TimeSeriesCV:
    """Time-series cross validation with expanding window splits."""

    def __init__(self, n_splits: int = 5, val_ratio: float = 0.2) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be between 0 and 1")
        self.n_splits = n_splits
        self.val_ratio = val_ratio

    def split(self, data: Iterable) -> Iterable[Dict[str, Tuple[int, int]]]:
        n = len(data)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + int(fold_size * self.val_ratio), n)
            yield {"train": (0, train_end), "val": (train_end, val_end)}
