"""
可视化工具模块
Visualization Module

提供训练监控、结果展示等可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class Visualizer:
    """
    可视化工具类
    
    提供训练和评估过程中的各种可视化功能
    """
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-whitegrid',
        figsize: Tuple[int, int] = (12, 6),
        save_dir: Optional[str] = None
    ):
        """
        初始化可视化器
        
        Args:
            style: matplotlib样式
            figsize: 默认图形大小
            save_dir: 图片保存目录
        """
        self.style = style
        self.figsize = figsize
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制训练历史曲线
        
        Args:
            history: 包含训练历史的字典
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        # TODO: 由实现者完成
        if not history:
            raise ValueError("history is empty")

        plt.style.use(self.style)

        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        learning_rate = history.get("learning_rate", [])

        has_lr = len(learning_rate) > 0
        if has_lr:
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(self.figsize[0], int(self.figsize[1] * 1.4)),
                sharex=True,
            )
            loss_ax, lr_ax = axes
        else:
            fig, loss_ax = plt.subplots(figsize=self.figsize)
            lr_ax = None

        epochs = range(1, max(len(train_loss), len(val_loss), len(learning_rate)) + 1)

        if train_loss:
            loss_ax.plot(epochs[:len(train_loss)], train_loss, label="Train Loss")
        if val_loss:
            loss_ax.plot(epochs[:len(val_loss)], val_loss, label="Val Loss")

        loss_ax.set_title(title)
        loss_ax.set_ylabel("Loss")
        if train_loss or val_loss:
            loss_ax.legend()

        if lr_ax is not None:
            lr_ax.plot(epochs[:len(learning_rate)], learning_rate, label="Learning Rate")
            lr_ax.set_xlabel("Epoch")
            lr_ax.set_ylabel("LR")
            lr_ax.legend()
        else:
            loss_ax.set_xlabel("Epoch")

        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_predictions(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        dates: Optional[List] = None,
        title: str = "Predictions vs Actual",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制预测结果与实际值对比图
        
        Args:
            actual: 实际值
            predicted: 预测值
            dates: 日期标签
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        actual = np.asarray(actual).squeeze()
        predicted = np.asarray(predicted).squeeze()

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(actual, label="Actual", linewidth=2)
        ax.plot(predicted, label="Predicted", linestyle="--")

        if dates is not None:
            ax.set_xticks(np.arange(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha="right")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_residuals(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        title: str = "Residual Analysis",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制残差分析图
        
        Args:
            actual: 实际值
            predicted: 预测值
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        actual_arr = np.asarray(actual).squeeze()
        predicted_arr = np.asarray(predicted).squeeze()

        if actual_arr.shape != predicted_arr.shape:
            raise ValueError(
                f"actual/predicted shape mismatch: actual={actual_arr.shape} predicted={predicted_arr.shape}"
            )

        residuals = actual_arr - predicted_arr

        # Multi-step forecasts are typically 2D: (n_samples, horizon).
        # Flatten to a single distribution for robust hist/scatter defaults.
        residuals_flat = residuals.reshape(-1)
        predicted_flat = predicted_arr.reshape(-1)

        plt.style.use(self.style)
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        axes[0].hist(residuals_flat, bins=30, color="steelblue", edgecolor="white")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Count")

        axes[1].scatter(predicted_flat, residuals_flat, alpha=0.6)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1].set_title("Residuals vs Predicted")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")

        fig.suptitle(title)
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_correlation_matrix(
        self,
        data: np.ndarray,
        feature_names: List[str],
        title: str = "Feature Correlation Matrix",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制相关性矩阵热力图
        
        Args:
            data: 数据矩阵
            feature_names: 特征名称
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        # TODO: 由实现者完成
        if data is None:
            raise ValueError("data is required")

        data_arr = np.asarray(data)
        if data_arr.ndim != 2:
            raise ValueError("data must be 2D")
        if data_arr.shape[1] != len(feature_names):
            raise ValueError("feature_names length mismatch")

        corr = np.corrcoef(data_arr, rowvar=False)

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            corr,
            xticklabels=feature_names,
            yticklabels=feature_names,
            cmap="coolwarm",
            center=0.0,
            ax=ax,
            cbar=False,
        )
        ax.set_title(title)
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_forecast_horizon(
        self,
        metrics_per_step: Dict[int, Dict[str, float]],
        metric_name: str = "rmse",
        title: str = "Forecast Performance by Horizon",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制不同预测步长的性能图
        
        Args:
            metrics_per_step: 每个预测步的指标
            metric_name: 要绘制的指标名
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        steps = sorted(metrics_per_step.keys())
        values = [
            metrics_per_step[step].get(metric_name.lower(), metrics_per_step[step].get(metric_name))
            for step in steps
        ]

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(steps, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_time_series(
        self,
        series: np.ndarray,
        dates: Optional[List] = None,
        title: str = "Time Series",
        ylabel: str = "Value",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制时间序列图
        
        Args:
            series: 时间序列数据
            dates: 日期标签
            title: 图表标题
            ylabel: Y轴标签
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        # TODO: 由实现者完成
        series_arr = np.asarray(series).squeeze()
        if series_arr.ndim != 1:
            series_arr = series_arr.reshape(-1)

        if dates is not None and len(dates) != len(series_arr):
            raise ValueError("dates length must match series length")

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(series_arr, label="Series", linewidth=2)

        if dates is not None:
            ax.set_xticks(np.arange(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha="right")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_multi_series(
        self,
        series_dict: Dict[str, np.ndarray],
        dates: Optional[List] = None,
        title: str = "Multiple Time Series",
        ylabel: str = "Value",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制多条时间序列
        
        Args:
            series_dict: 时间序列字典 {名称: 数据}
            dates: 日期标签
            title: 图表标题
            ylabel: Y轴标签
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        # TODO: 由实现者完成
        if not series_dict:
            raise ValueError("series_dict is empty")

        series_lengths = []
        series_arrays: Dict[str, np.ndarray] = {}
        for name, series in series_dict.items():
            arr = np.asarray(series).squeeze()
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            series_arrays[name] = arr
            series_lengths.append(len(arr))

        max_len = max(series_lengths)
        if dates is not None and len(dates) != max_len:
            raise ValueError("dates length must match longest series length")

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, arr in series_arrays.items():
            ax.plot(arr, label=name)

        if dates is not None:
            ax.set_xticks(np.arange(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha="right")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def plot_prediction_interval(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        dates: Optional[List] = None,
        title: str = "Prediction with Confidence Interval",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制带置信区间的预测图
        
        Args:
            actual: 实际值
            predicted: 预测值
            lower_bound: 下界
            upper_bound: 上界
            dates: 日期标签
            title: 图表标题
            save_name: 保存文件名
            
        Returns:
            matplotlib Figure对象
        """
        actual = np.asarray(actual).squeeze()
        predicted = np.asarray(predicted).squeeze()
        lower_bound = np.asarray(lower_bound).squeeze()
        upper_bound = np.asarray(upper_bound).squeeze()

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(actual, label="Actual", linewidth=2)
        ax.plot(predicted, label="Predicted", linestyle="--")
        ax.fill_between(
            np.arange(len(predicted)),
            lower_bound,
            upper_bound,
            color="gray",
            alpha=0.3,
            label="Confidence Interval",
        )

        if dates is not None:
            ax.set_xticks(np.arange(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha="right")

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        if save_name:
            self._save_figure(fig, save_name)
        return fig
    
    def _save_figure(
        self, 
        fig: plt.Figure, 
        save_name: str,
        dpi: int = 300
    ) -> None:
        """
        保存图形
        
        Args:
            fig: matplotlib Figure对象
            save_name: 文件名
            dpi: 分辨率
        """
        if self.save_dir and save_name:
            save_path = self.save_dir / f"{save_name}.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')


class TrainingMonitor:
    """
    训练过程监控器
    
    实时可视化训练进度和指标
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        初始化训练监控器
        
        Args:
            log_dir: 日志目录
        """
        # TODO: 由实现者完成
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        记录训练指标
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典
        """
        # TODO: 由实现者完成
        raise NotImplementedError("待实现: 记录指标")
    
    def update_display(self) -> None:
        """更新可视化显示"""
        # TODO: 由实现者完成
        raise NotImplementedError("待实现: 更新显示")
