"""
损失函数模块
Loss Functions Module

包含用于传染病预测的自定义损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMSELoss(nn.Module):
    """
    加权均方误差损失
    
    对不同预测步长应用不同权重，
    通常近期预测更重要，权重更高
    """
    
    def __init__(self, weights: Optional[torch.Tensor] = None):
        """
        初始化加权MSE损失
        
        Args:
            weights: 各预测步长的权重张量
        """
        super().__init__()
        self.weights = weights
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权MSE损失
        
        Args:
            predictions: 预测值, shape: (batch, horizon)
            targets: 目标值, shape: (batch, horizon)
            
        Returns:
            加权MSE损失值
        """
        errors = (predictions - targets) ** 2
        if errors.dim() == 3:
            errors = errors.mean(dim=-1)

        if self.weights is None:
            return errors.mean()

        weights = self.weights.to(errors.device)
        if weights.dim() == 1:
            weights = weights.view(1, -1)

        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        weighted_errors = errors * weights
        return weighted_errors.sum(dim=1).mean()


class HuberLoss(nn.Module):
    """
    Huber损失
    
    结合MSE和MAE的优点，对异常值更鲁棒
    """
    
    def __init__(self, delta: float = 1.0):
        """
        初始化Huber损失
        
        Args:
            delta: 切换阈值
        """
        super().__init__()
        self.delta = delta
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Huber损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            Huber损失值
        """
        diff = predictions - targets
        abs_diff = diff.abs()
        quadratic = 0.5 * diff ** 2
        linear = self.delta * (abs_diff - 0.5 * self.delta)
        loss = torch.where(abs_diff <= self.delta, quadratic, linear)
        return loss.mean()


class HybridLoss(nn.Module):
    """
    混合损失函数
    
    结合多种损失函数:
    - MSE: 主回归损失
    - 趋势损失: 确保预测趋势正确
    - 峰值损失: 强调峰值区域的准确性
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        trend_weight: float = 0.1,
        peak_weight: float = 0.2,
        peak_threshold: float = 0.7
    ):
        """
        初始化混合损失
        
        Args:
            mse_weight: MSE损失权重
            trend_weight: 趋势损失权重
            peak_weight: 峰值损失权重
            peak_threshold: 峰值判定阈值 (相对于最大值)
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.trend_weight = trend_weight
        self.peak_weight = peak_weight
        self.peak_threshold = peak_threshold
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算混合损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            混合损失值
        """
        predictions, targets = self._align_shapes(predictions, targets)

        mse_loss = F.mse_loss(predictions, targets)

        trend_loss = torch.tensor(0.0, device=predictions.device)
        if self.trend_weight > 0:
            trend_loss = self._compute_trend_loss(predictions, targets)

        peak_loss = torch.tensor(0.0, device=predictions.device)
        if self.peak_weight > 0:
            peak_loss = self._compute_peak_loss(predictions, targets)

        return (
            self.mse_weight * mse_loss
            + self.trend_weight * trend_loss
            + self.peak_weight * peak_loss
        )
    
    def _compute_trend_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算趋势损失
        
        惩罚预测趋势与真实趋势不一致的情况
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            趋势损失值
        """
        predictions, targets = self._reduce_series(predictions, targets)
        if predictions.size(1) < 2:
            return torch.tensor(0.0, device=predictions.device)

        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        target_diff = targets[:, 1:] - targets[:, :-1]
        return F.mse_loss(pred_diff, target_diff)
    
    def _compute_peak_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算峰值损失
        
        在峰值区域应用更高的权重
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            峰值损失值
        """
        predictions, targets = self._reduce_series(predictions, targets)
        max_vals = targets.abs().amax(dim=1, keepdim=True)
        threshold = max_vals * self.peak_threshold
        mask = (targets.abs() >= threshold).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        errors = (predictions - targets) ** 2
        return (errors * mask).sum() / (mask.sum() + 1e-8)

    @staticmethod
    def _align_shapes(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if predictions.dim() == 3 and predictions.size(-1) == 1 and targets.dim() == 2:
            predictions = predictions.squeeze(-1)
        if targets.dim() == 3 and targets.size(-1) == 1 and predictions.dim() == 2:
            targets = targets.squeeze(-1)
        return predictions, targets

    @staticmethod
    def _reduce_series(
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if predictions.dim() == 3:
            predictions = predictions.mean(dim=-1)
        if targets.dim() == 3:
            targets = targets.mean(dim=-1)
        return predictions, targets


class QuantileLoss(nn.Module):
    """
    分位数损失
    
    用于概率预测，生成预测区间
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """
        初始化分位数损失
        
        Args:
            quantiles: 目标分位数列表
        """
        super().__init__()
        self.quantiles = quantiles
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分位数损失
        
        Args:
            predictions: 预测值, shape: (batch, horizon, num_quantiles)
            targets: 目标值, shape: (batch, horizon)
            
        Returns:
            分位数损失值
        """
        quantiles = torch.tensor(self.quantiles, device=predictions.device, dtype=predictions.dtype)
        quantiles = quantiles.view(1, 1, -1)

        errors = targets.unsqueeze(-1) - predictions
        loss = torch.maximum((quantiles - 1) * errors, quantiles * errors)
        return loss.mean()
