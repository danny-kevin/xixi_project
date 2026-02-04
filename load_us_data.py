"""
US COVID-19 Data Loader

Supports:
- dataset_US_final.csv (single time series)
- all_states_processed_scaled_check.csv (weekly state-level panel)
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import DataPreprocessor
from src.data.dataset import EpidemicDataset, create_data_loaders


class USCovidDataLoader:
    """
    Data loader for US COVID-19 datasets.

    - Single-series CSV with Date column
    - Panel CSV with weekly index + State column
    """

    def __init__(self, data_path: str = "data/raw/dataset_US_final.csv"):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)

        # Resolve date/index column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        else:
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
            if df[first_col].notnull().any():
                df = df.set_index(first_col)

        df = df[df.index.notnull()].sort_index()

        print(f"\u2705 Loaded data: {len(df)} rows")
        print(f"\u2705 Date range: {df.index.min()} to {df.index.max()}")
        print(f"\u2705 Columns: {list(df.columns)}")

        return df

    def prepare_data(
        self,
        target_column: str = "Confirmed",
        window_size: int = 21,
        horizon: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        scaler_type: str = "standard",
        feature_columns: Optional[List[str]] = None,
    ) -> Dict:
        """Prepare train/val/test tensors."""
        df = self.load_data()

        # Handle missing values
        print("\nChecking missing values...")
        missing = df.isnull().sum()
        if missing.any():
            print(missing[missing > 0])
            print("  Using ffill + bfill")
            df = df.fillna(method="ffill").fillna(method="bfill")
        else:
            print("  \u2705 No missing values")

        # Panel path
        if "State" in df.columns:
            return self._prepare_panel_data(
                df=df,
                target_column=target_column,
                feature_columns=feature_columns,
                window_size=window_size,
                horizon=horizon,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                scaler_type=scaler_type,
            )

        # Single-series path
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != target_column]
        else:
            if target_column not in feature_columns:
                feature_columns = [target_column] + list(feature_columns)

        print(f"\nFeatures ({len(feature_columns)}): {feature_columns}")
        print(f"Target: {target_column}")

        preprocessor = DataPreprocessor()
        df_normalized = df.copy()
        if scaler_type and scaler_type.lower() != "none":
            df_normalized[feature_columns] = preprocessor.normalize(
                df[feature_columns], method=scaler_type
            )
            df_normalized[target_column] = preprocessor.normalize(
                df[[target_column]], method=scaler_type
            )[target_column]
            print(f"\u2705 Normalized ({scaler_type})")
        else:
            print("\u2705 Skip normalization (scaler_type=none)")

        all_data = df_normalized.values
        X, y = preprocessor.create_time_windows(
            all_data, window_size=window_size, horizon=horizon
        )

        print("\nCreated time windows:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

        X_train, X_val, X_test = preprocessor.temporal_train_test_split(
            X, train_ratio=train_ratio, val_ratio=val_ratio
        )
        y_train, y_val, y_test = preprocessor.temporal_train_test_split(
            y, train_ratio=train_ratio, val_ratio=val_ratio
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "original_df": df,
        }

    def _prepare_panel_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]],
        window_size: int,
        horizon: int,
        train_ratio: float,
        val_ratio: float,
        scaler_type: str,
    ) -> Dict:
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c != "State"]
        # Ensure target is the first feature
        feature_columns = [target_column] + [
            c for c in feature_columns if c not in (target_column, "State")
        ]

        print(f"\nPanel features ({len(feature_columns)}): {feature_columns}")
        print(f"Target: {target_column}")

        df_panel = df.copy()
        preprocessor = DataPreprocessor()
        if scaler_type and scaler_type.lower() != "none":
            df_panel[feature_columns] = preprocessor.normalize(
                df_panel[feature_columns], method=scaler_type
            )
            print(f"\u2705 Normalized ({scaler_type})")
        else:
            print("\u2705 Skip normalization (scaler_type=none)")

        unique_weeks = sorted(df_panel.index.unique())
        n_total = len(unique_weeks)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_weeks = set(unique_weeks[:n_train])
        val_weeks = set(unique_weeks[n_train:n_train + n_val])
        test_weeks = set(unique_weeks[n_train + n_val:])

        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []

        for state in df_panel["State"].unique():
            df_s = df_panel[df_panel["State"] == state].sort_index()
            mat = df_s[feature_columns].values
            weeks = df_s.index.to_list()

            L = len(mat)
            for i in range(L - window_size - horizon + 1):
                X = mat[i:i + window_size, :]
                y = mat[i + window_size:i + window_size + horizon, 0]

                target_week = weeks[i + window_size + horizon - 1]
                if target_week in train_weeks:
                    X_train.append(X); y_train.append(y)
                elif target_week in val_weeks:
                    X_val.append(X); y_val.append(y)
                elif target_week in test_weeks:
                    X_test.append(X); y_test.append(y)

        return {
            "X_train": np.array(X_train),
            "y_train": np.array(y_train),
            "X_val": np.array(X_val),
            "y_val": np.array(y_val),
            "X_test": np.array(X_test),
            "y_test": np.array(y_test),
            "preprocessor": preprocessor,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "original_df": df_panel,
        }

    def create_dataloaders(
        self,
        data_dict: Dict,
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Tuple:
        train_dataset = EpidemicDataset(
            data_dict["X_train"],
            data_dict["y_train"],
            feature_names=data_dict["feature_columns"],
        )
        val_dataset = EpidemicDataset(
            data_dict["X_val"],
            data_dict["y_val"],
            feature_names=data_dict["feature_columns"],
        )
        test_dataset = EpidemicDataset(
            data_dict["X_test"],
            data_dict["y_test"],
            feature_names=data_dict["feature_columns"],
        )

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        print("\nDataLoader created:")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("US COVID-19 Data Loader Test")
    print("=" * 60)

    loader = USCovidDataLoader()
    data_dict = loader.prepare_data(
        target_column="Confirmed",
        window_size=21,
        horizon=7,
    )

    train_loader, val_loader, test_loader = loader.create_dataloaders(
        data_dict,
        batch_size=32,
    )

    for batch_X, batch_y in train_loader:
        print(f"batch_X: {batch_X.shape}")
        print(f"batch_y: {batch_y.shape}")
        break

    print("\n[OK] data loader test passed")
