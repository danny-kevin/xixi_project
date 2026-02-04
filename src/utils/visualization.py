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
        raise NotImplementedError("待实现: 训练历史图")
    
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
        actual = np.asarray(actual).squeeze()
        predicted = np.asarray(predicted).squeeze()
        residuals = actual - predicted

        plt.style.use(self.style)
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))

        axes[0].hist(residuals, bins=30, color="steelblue", edgecolor="white")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual")
        axes[0].set_ylabel("Count")

        axes[1].scatter(predicted, residuals, alpha=0.6)
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
        raise NotImplementedError("待实现: 相关性矩阵图")
    
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
        raise NotImplementedError("待实现: 时间序列图")
    
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
        raise NotImplementedError("待实现: 多序列图")
    
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
