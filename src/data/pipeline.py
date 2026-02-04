"""
数据预处理管道
Data Preprocessing Pipeline

整合数据加载、预处理、数据集创建的完整流程
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .dataset import EpidemicDataset, create_data_loaders


class DataPipeline:
    """
    端到端数据预处理管道
    
    整合了数据加载、清洗、预处理、数据集构建的完整流程
    """
    
    def __init__(
        self,
        data_dir: str,
        window_size: int = 21,
        horizon: int = 7,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 4,
        use_single_file: bool = False,
        single_file_name: str = "dataset_US_final.csv",
        date_column: str = "Date",
        target_column: str = "Confirmed",
        group_column: Optional[str] = None,
        one_hot_columns: Optional[List[str]] = None,
        normalize: bool = True,
        normalize_method: str = "minmax",
        handle_missing: bool = True,
        detect_outliers: bool = True,
    ):
        """
        初始化数据管道
        
        Args:
            data_dir: 数据目录路径
            window_size: 输入时间窗口大小（天）
            horizon: 预测时间范围（天）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_single_file = use_single_file
        self.single_file_name = single_file_name
        self.date_column = date_column
        self.target_column = target_column
        self.group_column = group_column
        self.one_hot_columns = one_hot_columns or []
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.handle_missing = handle_missing
        self.detect_outliers = detect_outliers
        self.group_values: Optional[pd.Series] = None
        
        # 初始化组件
        self.loader = DataLoader(data_dir)
        self.preprocessor = DataPreprocessor()
        
        # 存储处理后的数据
        self.merged_data: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data sources and return a merged DataFrame.
        """
        if self.use_single_file:
            return self._load_single_file()

        print("[DATA] load multi-source data")
        data_dict = self.loader.load_all_data()

        if not data_dict:
            raise ValueError("No data sources found")

        print(f"[DATA] loaded sources: {len(data_dict)}")
        merged_data = self.loader.merge_data_sources(data_dict)
        print(f"[DATA] merged shape: {merged_data.shape}")

        self.merged_data = merged_data
        return merged_data

    def _load_single_file(self) -> pd.DataFrame:
        candidate_paths = [
            self.data_dir / "raw" / self.single_file_name,
            self.data_dir / self.single_file_name,
        ]
        file_path = next((p for p in candidate_paths if p.exists()), None)
        if file_path is None:
            worktree_root = Path(__file__).resolve().parents[2]
            primary_root = worktree_root.parent.parent
            fallback_paths = [
                worktree_root / "data" / "raw" / self.single_file_name,
                primary_root / "data" / "raw" / self.single_file_name,
            ]
            file_path = next((p for p in fallback_paths if p.exists()), None)
            if file_path is None:
                tried = candidate_paths + fallback_paths
                raise FileNotFoundError(
                    f"Data file not found. Tried: {', '.join(str(p) for p in tried)}"
                )

        df = pd.read_csv(file_path, parse_dates=[self.date_column])
        if self.date_column in df.columns:
            df = df.set_index(self.date_column)
        df = df.sort_index()

        self.merged_data = df
        return df

    def preprocess_data(
        self,
        df: Optional[pd.DataFrame] = None,
        handle_missing: Optional[bool] = None,
        detect_outliers: Optional[bool] = None,
        normalize: Optional[bool] = None,
    ) -> pd.DataFrame:
        """
        Preprocess data before windowing.
        """
        if df is None:
            if self.merged_data is None:
                raise ValueError("Call load_data() first")
            df = self.merged_data

        if handle_missing is None:
            handle_missing = self.handle_missing
        if detect_outliers is None:
            detect_outliers = self.detect_outliers
        if normalize is None:
            normalize = self.normalize

        df_processed = df.copy()
        self.group_values = None
        if self.group_column and self.group_column in df_processed.columns:
            self.group_values = df_processed[self.group_column].copy()
            self.group_values = self.group_values.fillna("Unknown")

        if self.one_hot_columns:
            one_hot_cols = [c for c in self.one_hot_columns if c in df_processed.columns]
            if one_hot_cols:
                df_processed = pd.get_dummies(
                    df_processed,
                    columns=one_hot_cols,
                    prefix=one_hot_cols,
                    dtype=float,
                )

        non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            df_processed = df_processed.drop(columns=non_numeric)

        if handle_missing:
            df_processed = self.preprocessor.handle_missing_values(df_processed, method="interpolate")

        if detect_outliers:
            _ = self.preprocessor.detect_outliers(df_processed, method="iqr", threshold=1.5)

        if normalize:
            df_processed = self.preprocessor.normalize(df_processed, method=self.normalize_method)

        self.feature_names = df_processed.columns.tolist()
        return df_processed

    def create_datasets(
        self,
        df: Optional[pd.DataFrame] = None
    ) -> Tuple[EpidemicDataset, EpidemicDataset, EpidemicDataset]:
        """
        Create train/val/test datasets.
        """
        if df is None:
            raise ValueError("Provide preprocessed data")

        if not self.feature_names:
            self.feature_names = df.columns.tolist()

        if self.target_column not in df.columns:
            if self.use_single_file:
                raise ValueError(f"Target column not found: {self.target_column}")
            print(f"[WARN] target column not found: {self.target_column}, fallback to first column")
            self.target_column = df.columns[0]

        target_col_idx = df.columns.get_loc(self.target_column)

        def split_arrays(X: np.ndarray, y: np.ndarray):
            n_samples = len(X)
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))
            return (
                X[:train_end], y[:train_end],
                X[train_end:val_end], y[train_end:val_end],
                X[val_end:], y[val_end:],
            )

        if self.group_column and self.group_values is not None:
            X_train_list, y_train_list = [], []
            X_val_list, y_val_list = [], []
            X_test_list, y_test_list = [], []

            for state in sorted(self.group_values.unique()):
                state_mask = self.group_values == state
                state_df = df.loc[state_mask]
                if state_df.empty:
                    continue
                state_df = state_df.sort_index()
                data_array = state_df.values
                X_state, y_state = self.preprocessor.create_time_windows(
                    data_array,
                    window_size=self.window_size,
                    horizon=self.horizon,
                    stride=1,
                    target_col_idx=target_col_idx,
                )
                if len(X_state) == 0:
                    continue

                X_tr, y_tr, X_vl, y_vl, X_te, y_te = split_arrays(X_state, y_state)
                if len(X_tr) > 0:
                    X_train_list.append(X_tr)
                    y_train_list.append(y_tr)
                if len(X_vl) > 0:
                    X_val_list.append(X_vl)
                    y_val_list.append(y_vl)
                if len(X_te) > 0:
                    X_test_list.append(X_te)
                    y_test_list.append(y_te)

            if not X_train_list:
                raise ValueError("No windows generated from grouped data")

            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            feature_dim = X_train.shape[2] if X_train.ndim == 3 else 0
            empty_X = np.empty((0, self.window_size, feature_dim))
            empty_y = np.empty((0, self.horizon)) if self.horizon > 1 else np.empty((0,))

            X_val = np.concatenate(X_val_list, axis=0) if X_val_list else empty_X
            y_val = np.concatenate(y_val_list, axis=0) if y_val_list else empty_y
            X_test = np.concatenate(X_test_list, axis=0) if X_test_list else empty_X
            y_test = np.concatenate(y_test_list, axis=0) if y_test_list else empty_y
        else:
            data_array = df.values
            X, y = self.preprocessor.create_time_windows(
                data_array,
                window_size=self.window_size,
                horizon=self.horizon,
                stride=1,
                target_col_idx=target_col_idx,
            )
            X_train, y_train, X_val, y_val, X_test, y_test = split_arrays(X, y)

        train_dataset = EpidemicDataset(X_train, y_train, self.feature_names)
        val_dataset = EpidemicDataset(X_val, y_val, self.feature_names)
        test_dataset = EpidemicDataset(X_test, y_test, self.feature_names)

        return train_dataset, val_dataset, test_dataset

    def create_dataloaders(
        self,
        train_dataset: EpidemicDataset,
        val_dataset: EpidemicDataset,
        test_dataset: EpidemicDataset
    ) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        创建DataLoader
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            test_dataset: 测试数据集
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print("[START] 创建DataLoader...")
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        
        print(f"   批次大小: {self.batch_size}")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   验证批次数: {len(val_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def run(self) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        运行完整的数据准备管道
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        print("=" * 60)
        print("[START] 开始数据预处理管道")
        print("=" * 60)
        
        # 1. 加载数据
        merged_data = self.load_data()
        
        # 2. 预处理
        processed_data = self.preprocess_data(
            merged_data,
            handle_missing=self.handle_missing,
            detect_outliers=self.detect_outliers,
            normalize=self.normalize
        )
        
        # 3. 创建数据集
        train_dataset, val_dataset, test_dataset = self.create_datasets(processed_data)
        
        # 4. 创建DataLoader
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset
        )
        
        print("=" * 60)
        print("[OK] 数据预处理管道完成！")
        print("=" * 60)
        
        return train_loader, val_loader, test_loader
    
    def get_preprocessor(self) -> DataPreprocessor:
        """获取预处理器（用于反归一化等操作）"""
        return self.preprocessor
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称列表"""
        return self.feature_names


# 便捷函数
def prepare_data(
    data_dir: str,
    window_size: int = 21,
    horizon: int = 7,
    batch_size: int = 32,
    **kwargs
) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader, DataPreprocessor]:
    """
    一键准备数据的便捷函数
    
    Args:
        data_dir: 数据目录
        window_size: 输入窗口大小
        horizon: 预测范围
        batch_size: 批次大小
        **kwargs: 其他参数
        
    Returns:
        (train_loader, val_loader, test_loader, preprocessor)
    """
    pipeline = DataPipeline(
        data_dir=data_dir,
        window_size=window_size,
        horizon=horizon,
        batch_size=batch_size,
        **kwargs
    )
    
    train_loader, val_loader, test_loader = pipeline.run()
    preprocessor = pipeline.get_preprocessor()
    
    return train_loader, val_loader, test_loader, preprocessor
