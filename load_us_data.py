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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

    print(f" Loaded data: {len(df)} rows")
    print(f" Date range: {df.index.min()} to {df.index.max()}")
    print(f" Columns: {list(df.columns)}")

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
    per_state_normalize: bool = False,
    target_log1p: bool = False,
    state_column: str = "State",
  ) -> Dict:
    """Prepare train/val/test tensors."""
    df = self.load_data()

    # Handle missing values
    print("\nChecking missing values...")
    missing = df.isnull().sum()
    if missing.any():
      print(missing[missing > 0])
      print(" Using ffill + bfill")
      df = df.fillna(method="ffill").fillna(method="bfill")
    else:
      print("  No missing values")

    # Panel path
    if state_column in df.columns:
      return self._prepare_panel_data(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        scaler_type=scaler_type,
        per_state_normalize=per_state_normalize,
        target_log1p=target_log1p,
        state_column=state_column,
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
      print(f" Normalized ({scaler_type})")
    else:
      print(" Skip normalization (scaler_type=none)")

    all_data = df_normalized.values
    X, y = preprocessor.create_time_windows(
      all_data, window_size=window_size, horizon=horizon
    )

    print("\nCreated time windows:")
    print(f" X shape: {X.shape}")
    print(f" y shape: {y.shape}")

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
    per_state_normalize: bool,
    target_log1p: bool,
    state_column: str,
  ) -> Dict:
    if feature_columns is None:
      feature_columns = [c for c in df.columns if c != state_column]
    # Ensure target is the first feature
    feature_columns = [target_column] + [
      c for c in feature_columns if c not in (target_column, state_column)
    ]

    print(f"\nPanel features ({len(feature_columns)}): {feature_columns}")
    print(f"Target: {target_column}")

    df_panel = df.copy()
    preprocessor = DataPreprocessor()
    if state_column not in df_panel.columns:
      raise ValueError(f"State column '{state_column}' not found in data")

    if target_log1p:
      df_panel[target_column] = np.log1p(df_panel[target_column].clip(lower=0))
      print(" Applied log1p to target")

    if scaler_type and scaler_type.lower() != "none" and not per_state_normalize:
      df_panel[feature_columns] = preprocessor.normalize(
        df_panel[feature_columns], method=scaler_type
      )
      print(f" Normalized globally ({scaler_type})")
    elif scaler_type and scaler_type.lower() != "none" and per_state_normalize:
      print(f" Per-state normalization ({scaler_type})")
    else:
      print(" Skip normalization (scaler_type=none)")

    unique_weeks = sorted(df_panel.index.unique())
    n_total = len(unique_weeks)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_weeks = set(unique_weeks[:n_train])
    val_weeks = set(unique_weeks[n_train:n_train + n_val])
    test_weeks = set(unique_weeks[n_train + n_val:])

    X_train, y_train, state_ids_train = [], [], []
    X_val, y_val, state_ids_val = [], [], []
    X_test, y_test, state_ids_test = [], [], []

    state_to_id = {s: i for i, s in enumerate(sorted(df_panel[state_column].unique()))}
    id_to_state = {i: s for s, i in state_to_id.items()}
    state_scalers: Dict[str, Dict[str, object]] = {}

    for state in df_panel[state_column].unique():
      df_s = df_panel[df_panel[state_column] == state].sort_index()

      if scaler_type and scaler_type.lower() != "none" and per_state_normalize:
        scalers = {}
        train_mask = df_s.index.isin(train_weeks)
        fit_df = df_s.loc[train_mask, feature_columns]
        if fit_df.empty:
          fit_df = df_s[feature_columns]

        for col in feature_columns:
          if scaler_type.lower() == "standard":
            scaler = StandardScaler()
          elif scaler_type.lower() == "minmax":
            scaler = MinMaxScaler()
          else:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}")

          scaler.fit(fit_df[[col]])
          df_s[col] = scaler.transform(df_s[[col]])
          scalers[col] = scaler
        state_scalers[state] = scalers

      mat = df_s[feature_columns].values
      weeks = df_s.index.to_list()

      L = len(mat)
      for i in range(L - window_size - horizon + 1):
        X = mat[i:i + window_size, :]
        y = mat[i + window_size:i + window_size + horizon, 0]
        state_id = state_to_id[state]

        target_week = weeks[i + window_size + horizon - 1]
        if target_week in train_weeks:
          X_train.append(X); y_train.append(y); state_ids_train.append(state_id)
        elif target_week in val_weeks:
          X_val.append(X); y_val.append(y); state_ids_val.append(state_id)
        elif target_week in test_weeks:
          X_test.append(X); y_test.append(y); state_ids_test.append(state_id)

    return {
      "X_train": np.array(X_train),
      "y_train": np.array(y_train),
      "X_val": np.array(X_val),
      "y_val": np.array(y_val),
      "X_test": np.array(X_test),
      "y_test": np.array(y_test),
      "preprocessor": preprocessor,
      "state_ids_train": np.array(state_ids_train),
      "state_ids_val": np.array(state_ids_val),
      "state_ids_test": np.array(state_ids_test),
      "state_to_id": state_to_id,
      "id_to_state": id_to_state,
      "state_scalers": state_scalers,
      "target_log1p": target_log1p,
      "feature_columns": feature_columns,
      "target_column": target_column,
      "original_df": df_panel,
    }

  @staticmethod
  def inverse_transform_target(
    values: np.ndarray,
    state_ids: Optional[np.ndarray],
    id_to_state: Optional[Dict[int, str]],
    state_scalers: Optional[Dict[str, Dict[str, object]]],
    target_column: str,
    target_log1p: bool,
  ) -> np.ndarray:
    """Inverse transform target values (per-state)."""
    if state_ids is None or id_to_state is None or not state_scalers:
      return values

    values = np.asarray(values)
    output = np.zeros_like(values)

    for i, sid in enumerate(state_ids):
      state = id_to_state.get(int(sid))
      scaler = None
      if state is not None:
        scaler = state_scalers.get(state, {}).get(target_column)

      row = values[i]
      if scaler is not None:
        row = scaler.inverse_transform(np.array(row).reshape(-1, 1)).flatten()
      output[i] = row

    if target_log1p:
      output = np.expm1(output)
    return output

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
      state_ids=data_dict.get("state_ids_train"),
    )
    val_dataset = EpidemicDataset(
      data_dict["X_val"],
      data_dict["y_val"],
      feature_names=data_dict["feature_columns"],
      state_ids=data_dict.get("state_ids_val"),
    )
    test_dataset = EpidemicDataset(
      data_dict["X_test"],
      data_dict["y_test"],
      feature_names=data_dict["feature_columns"],
      state_ids=data_dict.get("state_ids_test"),
    )

    train_loader, val_loader, test_loader = create_data_loaders(
      train_dataset,
      val_dataset,
      test_dataset,
      batch_size=batch_size,
      num_workers=num_workers,
    )

    print("\nDataLoader created:")
    print(f" Batch size: {batch_size}")
    print(f" Train batches: {len(train_loader)}")
    print(f" Val batches: {len(val_loader)}")
    print(f" Test batches: {len(test_loader)}")

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
