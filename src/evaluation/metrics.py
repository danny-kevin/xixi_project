"""
评估指标模块
Metrics Module

包含传染病预测的各种评估指标
"""

import numpy as np
import torch
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """
    评估指标结果
    """
    mse: float
    rmse: float
    mae: float
    mape: float
    r2: float
    correlation: float
    crps: Optional[float] = None


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size == 0 or b.size == 0:
        return float('nan')
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm
        return norm.pdf(z)
    except Exception:
        return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    try:
        from scipy.stats import norm
        return norm.cdf(z)
    except Exception:
        from math import erf
        if np.isscalar(z):
            return 0.5 * (1 + erf(z / np.sqrt(2)))
        z_arr = np.asarray(z)
        return 0.5 * (1 + np.vectorize(erf)(z_arr / np.sqrt(2)))


class RegressionMetrics:
    """
    回归评估指标
    
    包含:
    - MSE: 均方误差
    - RMSE: 均方根误差
    - MAE: 平均绝对误差
    - MAPE: 平均绝对百分比误差
    - R²: 决定系数
    - Correlation: 相关系数
    """
    
    @staticmethod
    def mse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算均方误差
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            MSE值
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        return float(np.mean((pred - target) ** 2))
    
    @staticmethod
    def rmse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算均方根误差
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            RMSE值
        """
        return float(np.sqrt(RegressionMetrics.mse(predictions, targets)))
    
    @staticmethod
    def mae(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算平均绝对误差
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            MAE值
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        return float(np.mean(np.abs(pred - target)))
    
    @staticmethod
    def mape(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        epsilon: float = 1e-8
    ) -> float:
        """
        计算平均绝对百分比误差
        
        Args:
            predictions: 预测值
            targets: 目标值
            epsilon: 防止除零的小值
            
        Returns:
            MAPE值 (百分比)
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        return float(np.mean(np.abs((target - pred) / (target + epsilon))) * 100.0)
    
    @staticmethod
    def r2_score(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算R²决定系数
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            R²值
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def correlation(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算皮尔逊相关系数
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            相关系数
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        return _safe_corrcoef(pred, target)

    @staticmethod
    def crps(
        pred_mean: Union[np.ndarray, torch.Tensor],
        pred_std: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute CRPS for Gaussian predictive distributions.
        """
        mean = _to_numpy(pred_mean)
        std = _to_numpy(pred_std)
        target = _to_numpy(targets)
        std = np.maximum(std, 1e-8)
        z = (target - mean) / std
        crps = std * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - 1 / np.sqrt(np.pi))
        return float(np.mean(crps))
    
    @classmethod
    def compute_all(
        cls,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        pred_std: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> MetricsResult:
        """
        计算所有评估指标
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            包含所有指标的结果对象
        """
        result = MetricsResult(
            mse=cls.mse(predictions, targets),
            rmse=cls.rmse(predictions, targets),
            mae=cls.mae(predictions, targets),
            mape=cls.mape(predictions, targets),
            r2=cls.r2_score(predictions, targets),
            correlation=cls.correlation(predictions, targets),
        )
        if pred_std is not None:
            result.crps = cls.crps(predictions, pred_std, targets)
        return result


class TimeSeriesMetrics:
    """
    时间序列专用评估指标
    
    包含:
    - 方向准确率: 预测趋势方向的准确性
    - 峰值误差: 在峰值区域的预测误差
    - 滞后分析: 预测与实际的时间滞后
    """
    
    @staticmethod
    def directional_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        time_axis: int = -1
    ) -> float:
        """
        计算方向准确率
        
        预测变化方向与实际变化方向一致的比例
        
        Args:
            predictions: 预测值序列
            targets: 目标值序列
            time_axis: 时间维度 (默认最后一维)
            
        Returns:
            方向准确率 (0-1)
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        pred = np.squeeze(pred)
        target = np.squeeze(target)
        axis = time_axis % pred.ndim
        if pred.shape[axis] < 2 or target.shape[axis] < 2:
            return float('nan')
        pred_diff = np.diff(pred, axis=axis)
        target_diff = np.diff(target, axis=axis)
        direction_match = np.sign(pred_diff) == np.sign(target_diff)
        return float(np.mean(direction_match))
    
    @staticmethod
    def peak_detection_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        评估峰值检测准确性
        
        Args:
            predictions: 预测值序列
            targets: 目标值序列
            threshold: 峰值判定阈值
            
        Returns:
            包含精确率、召回率、F1的字典
        """
        pred = _to_numpy(predictions).reshape(-1)
        target = _to_numpy(targets).reshape(-1)
        if pred.size == 0 or target.size == 0:
            return {"precision": float('nan'), "recall": float('nan'), "f1": float('nan')}

        threshold_value = np.max(target) * threshold
        pred_peaks = pred >= threshold_value
        target_peaks = target >= threshold_value

        tp = np.sum(pred_peaks & target_peaks)
        fp = np.sum(pred_peaks & ~target_peaks)
        fn = np.sum(~pred_peaks & target_peaks)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    
    @staticmethod
    def compute_lag_correlation(
        predictions: np.ndarray,
        targets: np.ndarray,
        max_lag: int = 7
    ) -> Dict[int, float]:
        """
        计算滞后相关性
        
        分析预测与实际之间的时间滞后关系
        
        Args:
            predictions: 预测值序列
            targets: 目标值序列
            max_lag: 最大滞后天数
            
        Returns:
            各滞后步数对应的相关系数
        """
        pred = _to_numpy(predictions).reshape(-1)
        target = _to_numpy(targets).reshape(-1)
        results = {}
        for lag in range(max_lag + 1):
            if lag == 0:
                results[lag] = _safe_corrcoef(pred, target)
                continue
            if pred.size <= lag or target.size <= lag:
                results[lag] = float('nan')
                continue
            results[lag] = _safe_corrcoef(pred[:-lag], target[lag:])
        return results
    
    @staticmethod
    def forecast_horizon_analysis(
        predictions: np.ndarray,
        targets: np.ndarray,
        horizon: int = 7
    ) -> Dict[int, Dict[str, float]]:
        """
        分析不同预测步长的性能
        
        Args:
            predictions: 预测值, shape: (samples, horizon)
            targets: 目标值, shape: (samples, horizon)
            horizon: 预测范围
            
        Returns:
            各预测步长的评估指标
        """
        pred = _to_numpy(predictions)
        target = _to_numpy(targets)
        if pred.ndim == 3 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if target.ndim == 3 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        if pred.ndim > 2:
            pred = pred.mean(axis=-1)
        if target.ndim > 2:
            target = target.mean(axis=-1)

        results: Dict[int, Dict[str, float]] = {}
        steps = min(horizon, pred.shape[1], target.shape[1])
        for step in range(steps):
            step_pred = pred[:, step]
            step_target = target[:, step]
            results[step + 1] = {
                "rmse": RegressionMetrics.rmse(step_pred, step_target),
                "mae": RegressionMetrics.mae(step_pred, step_target),
                "mape": RegressionMetrics.mape(step_pred, step_target),
            }
        return results


class ModelEvaluator:
    """
    模型综合评估器
    
    提供完整的模型评估流程，包括:
    - 基础回归指标
    - 时序特定指标
    - 可视化报告生成
    """
    
    def __init__(self, model, device: str = 'cuda'):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            device: 计算设备
        """
        self.model = model.to(device) if hasattr(model, "to") else model
        self.device = device
        
    def evaluate(
        self,
        test_loader,
        return_predictions: bool = False
    ) -> Dict:
        """
        完整评估流程
        
        Args:
            test_loader: 测试数据加载器
            return_predictions: 是否返回预测结果
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    y = batch[1] if len(batch) > 1 else None
                else:
                    x, y = batch, None

                x = x.to(self.device)
                if y is not None:
                    y = y.to(self.device)

                output = self.model(x)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                preds.append(output.detach().cpu())
                if y is not None:
                    targets.append(y.detach().cpu())

        pred_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0)
        target_tensor = torch.cat(targets, dim=0) if targets else torch.empty(0)

        pred_np = pred_tensor.numpy()
        target_np = target_tensor.numpy() if targets else None

        results: Dict[str, object] = {}
        if target_np is not None and target_np.size > 0:
            metrics = RegressionMetrics.compute_all(pred_np, target_np)
            results["metrics"] = metrics

            reduced_pred = pred_np
            reduced_target = target_np
            if reduced_pred.ndim == 3 and reduced_pred.shape[-1] == 1:
                reduced_pred = reduced_pred.squeeze(-1)
            if reduced_target.ndim == 3 and reduced_target.shape[-1] == 1:
                reduced_target = reduced_target.squeeze(-1)

            results["time_series"] = {
                "directional_accuracy": TimeSeriesMetrics.directional_accuracy(
                    reduced_pred, reduced_target
                ),
                "peak_detection": TimeSeriesMetrics.peak_detection_accuracy(
                    reduced_pred, reduced_target
                ),
                "lag_correlation": TimeSeriesMetrics.compute_lag_correlation(
                    reduced_pred.reshape(-1), reduced_target.reshape(-1)
                ),
            }
            if reduced_pred.ndim >= 2 and reduced_target.ndim >= 2:
                results["time_series"]["forecast_horizon"] = TimeSeriesMetrics.forecast_horizon_analysis(
                    reduced_pred, reduced_target, horizon=reduced_pred.shape[1]
                )

        if return_predictions:
            results["predictions"] = pred_np
            results["targets"] = target_np

        return results
    
    def generate_report(
        self,
        results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        生成评估报告
        
        Args:
            results: 评估结果
            save_path: 报告保存路径
            
        Returns:
            报告文本
        """
        lines = ["# Model Evaluation Report", ""]

        metrics = results.get("metrics")
        if metrics:
            lines.append("## Regression Metrics")
            lines.append(f"- RMSE: {metrics.rmse:.4f}")
            lines.append(f"- MAE: {metrics.mae:.4f}")
            lines.append(f"- MAPE: {metrics.mape:.2f}%")
            lines.append(f"- R2: {metrics.r2:.4f}")
            lines.append(f"- Correlation: {metrics.correlation:.4f}")
            if metrics.crps is not None:
                lines.append(f"- CRPS: {metrics.crps:.4f}")
            lines.append("")

        time_series = results.get("time_series", {})
        if time_series:
            lines.append("## Time Series Metrics")
            if "directional_accuracy" in time_series:
                lines.append(f"- Directional Accuracy: {time_series['directional_accuracy']:.4f}")
            if "peak_detection" in time_series:
                peak = time_series["peak_detection"]
                lines.append(f"- Peak Detection (F1): {peak.get('f1', float('nan')):.4f}")
            lines.append("")

        report = "\n".join(lines).strip() + "\n"
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
        return report


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray, h: int = 1) -> Dict[str, float]:
    """
    Diebold-Mariano test for predictive accuracy differences.
    """
    errors1 = _to_numpy(errors1).reshape(-1)
    errors2 = _to_numpy(errors2).reshape(-1)
    d = errors1 ** 2 - errors2 ** 2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    gamma = []
    for k in range(1, h):
        if len(d) <= k:
            break
        gamma.append(np.cov(d[:-k], d[k:])[0, 1])

    adjusted_var = var_d + 2 * sum(gamma)
    dm_stat = mean_d / np.sqrt(adjusted_var / len(d))
    p_value = 2 * (1 - _norm_cdf(abs(dm_stat)))
    return {"dm_statistic": float(dm_stat), "p_value": float(p_value)}
