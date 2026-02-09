"""
Test Data Preprocessing Pipeline

This script is used in two ways:
- As a standalone smoke check: `python scripts/test_data_pipeline.py`
- As a pytest test module (two tests at the bottom)

Project reality:
- Multi-source raw files may not exist in your repo.
- The primary supported path is the single-file US dataset: `data/raw/dataset_US_final.csv`.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPipeline, prepare_data


def _detect_single_file(data_dir: Path) -> Optional[Path]:
    raw_dir = data_dir / "raw"
    candidate = raw_dir / "dataset_US_final.csv"
    return candidate if candidate.exists() else None


def run_data_pipeline_checks() -> bool:
    print("\n" + "=" * 70)
    print("[TEST] data preprocessing pipeline")
    print("=" * 70 + "\n")

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    single_file = _detect_single_file(data_dir)
    if single_file is None and not any(raw_dir.glob("*.csv")):
        print("[WARN] no CSV found under data/raw; generating sample data...")
        from scripts.generate_sample_data import generate_all_sample_data

        generate_all_sample_data(output_dir=str(raw_dir))
        single_file = _detect_single_file(data_dir)

    try:
        print("[NOTE] path: prepare_data()")

        if single_file is not None:
            print(f"[INFO] using single-file dataset: {single_file}")
            train_loader, val_loader, test_loader, preprocessor = prepare_data(
                data_dir=str(data_dir),
                window_size=21,
                horizon=7,
                batch_size=32,
                num_workers=0,
                use_single_file=True,
                single_file_name="dataset_US_final.csv",
                date_column="Date",
                target_column="WeeklyNewConfirmed_log1p_minmax",
                group_column="State",
                one_hot_columns=["State"],
                normalize=False,
            )
        else:
            print("[INFO] using multi-source dataset(s)")
            train_loader, val_loader, test_loader, preprocessor = prepare_data(
                data_dir=str(data_dir),
                window_size=21,
                horizon=7,
                batch_size=32,
                num_workers=0,
            )

        print("\n" + "=" * 70)
        print("[OK] data prepared")
        print("=" * 70)

        for batch_x, batch_y in train_loader:
            print("\n[INFO] train batch:")
            print(f"  X: {tuple(batch_x.shape)} dtype={batch_x.dtype}")
            print(f"  y: {tuple(batch_y.shape)} dtype={batch_y.dtype}")
            break

        for batch_x, batch_y in val_loader:
            print("\n[INFO] val batch:")
            print(f"  X: {tuple(batch_x.shape)}")
            print(f"  y: {tuple(batch_y.shape)}")
            break

        for batch_x, batch_y in test_loader:
            print("\n[INFO] test batch:")
            print(f"  X: {tuple(batch_x.shape)}")
            print(f"  y: {tuple(batch_y.shape)}")
            break

        print("\n" + "-" * 70)
        print("[CHECK] inverse transform (if scalers exist)")
        print("-" * 70)
        if getattr(preprocessor, "scalers", None):
            names = list(preprocessor.scalers.keys())
            if names:
                x = np.array([[0.5], [0.6], [0.7]])
                out = preprocessor.inverse_transform(x, names[0])
                print(f"[OK] inverse_transform works for feature={names[0]} -> {out}")
        else:
            print("[SKIP] no scalers present (normalize may be disabled)")

        print("\n" + "=" * 70)
        print("[OK] pipeline checks passed")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[ERR] pipeline checks failed: {e}")
        print("=" * 70 + "\n")
        import traceback

        traceback.print_exc()
        return False


def run_individual_components_checks() -> bool:
    print("\n" + "=" * 70)
    print("[TEST] individual components")
    print("=" * 70 + "\n")

    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Prefer a deterministic input frame for component tests.
        single_file = _detect_single_file(data_dir)
        if single_file is None:
            print("[SKIP] dataset_US_final.csv not found; skipping single-file component checks")
            return True

        pipeline = DataPipeline(
            data_dir=str(data_dir),
            window_size=21,
            horizon=7,
            train_ratio=0.7,
            val_ratio=0.15,
            batch_size=32,
            num_workers=0,
            use_single_file=True,
            single_file_name="dataset_US_final.csv",
            date_column="Date",
            target_column="WeeklyNewConfirmed_log1p_minmax",
            group_column="State",
            one_hot_columns=["State"],
            normalize=False,
        )
        df = pipeline.load_data()
        merged = pipeline.preprocess_data(df)

        print(f"[OK] loaded+processed frame: {merged.shape}")

        from src.data import DataPreprocessor, EpidemicDataset

        preprocessor = DataPreprocessor()

        processed = preprocessor.handle_missing_values(merged)
        print(f"[OK] handle_missing_values: remaining_na={processed.isnull().sum().sum()}")

        outliers = preprocessor.detect_outliers(processed)
        print(f"[OK] detect_outliers: total={int(outliers.sum().sum())}")

        normalized = preprocessor.normalize(processed)
        print(
            f"[OK] normalize: range=[{normalized.min().min():.3f}, {normalized.max().max():.3f}]"
        )

        X, y = preprocessor.create_time_windows(
            normalized.values, window_size=21, horizon=7
        )
        print(f"[OK] create_time_windows: X={X.shape} y={y.shape}")

        train, val, test = preprocessor.temporal_train_test_split(X)
        print(f"[OK] split: train={len(train)} val={len(val)} test={len(test)}")

        dataset = EpidemicDataset(X[:100], y[:100])
        sample_x, sample_y = dataset[0]
        print(f"[OK] dataset: len={len(dataset)} sample_X={sample_x.shape} sample_y={sample_y.shape}")

        print("\n" + "=" * 70)
        print("[OK] component checks passed")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"[ERR] component checks failed: {e}")
        print("=" * 70 + "\n")
        import traceback

        traceback.print_exc()
        return False


def test_data_pipeline():
    assert run_data_pipeline_checks()


def test_individual_components():
    assert run_individual_components_checks()


if __name__ == "__main__":
    ok1 = run_individual_components_checks()
    ok2 = run_data_pipeline_checks()
    sys.exit(0 if (ok1 and ok2) else 1)

