"""
可解释性分析模块
Interpretability Module

提供模型决策透明化的多级解释框架
"""

import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


class AttentionVisualizer:
    """
    注意力权重可视化器
    
    可视化模型内部的注意力分布，
    帮助理解模型关注的重点
    """
    
    def __init__(self, model, device: str = 'cuda'):
        """
        初始化注意力可视化器
        
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model.to(device) if hasattr(model, "to") else model
        self.device = device
        
    def extract_attention_weights(
        self, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        提取模型的注意力权重
        
        Args:
            x: 输入数据
            
        Returns:
            包含各层注意力权重的字典
        """
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_attention_weights"):
                weights = self.model.get_attention_weights(x)
            else:
                output = self.model(x, return_attention=True)
                if isinstance(output, (tuple, list)) and len(output) > 1:
                    weights = output[1]
                else:
                    raise ValueError("Model does not return attention weights.")

        if isinstance(weights, torch.Tensor):
            return {"attention": weights}
        if isinstance(weights, dict):
            return weights
        raise TypeError("Unsupported attention weights type.")
    
    def plot_temporal_attention(
        self,
        attention_weights: torch.Tensor,
        time_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制时间维度的注意力热力图
        
        Args:
            attention_weights: 注意力权重矩阵
            time_labels: 时间轴标签
            save_path: 图片保存路径
            
        Returns:
            matplotlib Figure对象
        """
        weights = attention_weights
        if isinstance(weights, dict):
            weights = weights.get("temporal", next(iter(weights.values())))
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        while weights.ndim > 2:
            weights = weights.mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(weights, cmap="viridis", ax=ax)
        ax.set_xlabel("Key Time Step")
        ax.set_ylabel("Query Time Step")
        ax.set_title("Temporal Attention")

        if time_labels is not None:
            ax.set_xticks(np.arange(len(time_labels)) + 0.5)
            ax.set_yticks(np.arange(len(time_labels)) + 0.5)
            ax.set_xticklabels(time_labels, rotation=45, ha="right")
            ax.set_yticklabels(time_labels, rotation=0)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    
    def plot_variable_attention(
        self,
        attention_weights: torch.Tensor,
        variable_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制变量间注意力分布
        
        Args:
            attention_weights: 变量注意力权重
            variable_names: 变量名称列表
            save_path: 图片保存路径
            
        Returns:
            matplotlib Figure对象
        """
        weights = attention_weights
        if isinstance(weights, dict):
            weights = weights.get("variable", next(iter(weights.values())))
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        while weights.ndim > 2:
            weights = weights.mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        if weights.ndim == 1:
            ax.barh(variable_names, weights)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Variable")
        else:
            sns.heatmap(weights, cmap="Blues", ax=ax, xticklabels=variable_names, yticklabels=variable_names)
            ax.set_xlabel("Key Variable")
            ax.set_ylabel("Query Variable")

        ax.set_title("Variable Attention")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    
    def plot_attention_over_time(
        self,
        x: torch.Tensor,
        num_samples: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制注意力随时间的变化
        
        Args:
            x: 输入序列
            num_samples: 采样数量
            save_path: 图片保存路径
            
        Returns:
            matplotlib Figure对象
        """
        weights_dict = self.extract_attention_weights(x)
        if "lstm_temporal" in weights_dict:
            weights = weights_dict["lstm_temporal"]
        elif "temporal" in weights_dict:
            weights = weights_dict["temporal"]
        else:
            weights = next(iter(weights_dict.values()))

        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()

        if weights.ndim == 4:
            weights = weights.mean(axis=1)
            weights = weights.mean(axis=1)
        elif weights.ndim == 3:
            weights = weights.mean(axis=1)
        elif weights.ndim == 2:
            weights = weights
        else:
            weights = np.squeeze(weights)

        num_samples = min(num_samples, weights.shape[0])
        fig, ax = plt.subplots(figsize=(10, 5))
        for idx in range(num_samples):
            ax.plot(weights[idx], alpha=0.7, label=f"Sample {idx + 1}")

        ax.set_title("Attention Over Time")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attention Weight")
        if num_samples <= 5:
            ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class FeatureImportanceAnalyzer:
    """
    特征重要性分析器
    
    通过多种方法分析各输入变量对预测的贡献度
    """
    
    def __init__(self, model, device: str = 'cuda'):
        """
        初始化特征重要性分析器
        
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model.to(device) if hasattr(model, "to") else model
        self.device = device
        
    def permutation_importance(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: List[str],
        num_repeats: int = 10,
        feature_dim: int = -1
    ) -> Dict[str, float]:
        """
        排列重要性分析
        
        通过随机打乱各变量的值来评估其重要性
        
        Args:
            X: 输入数据
            y: 目标值
            feature_names: 特征名称列表
            num_repeats: 重复次数
            feature_dim: 特征所在维度 (默认最后一维)
            
        Returns:
            各特征的重要性分数
        """
        self.model.eval()
        X = X.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            base_output = self.model(X)
            if isinstance(base_output, (tuple, list)):
                base_output = base_output[0]

        from .metrics import RegressionMetrics

        base_error = RegressionMetrics.mae(base_output, y)
        importance: Dict[str, float] = {}
        feature_dim = feature_dim % X.dim()
        feature_count = X.shape[feature_dim]
        names = feature_names[:feature_count]

        for idx, name in enumerate(names):
            scores = []
            for _ in range(num_repeats):
                X_perm = X.clone()
                perm_idx = torch.randperm(X_perm.size(0))
                indexer = [slice(None)] * X_perm.dim()
                indexer[feature_dim] = idx
                feature_slice = X_perm[tuple(indexer)]
                X_perm[tuple(indexer)] = feature_slice[perm_idx, ...]
                with torch.no_grad():
                    output = self.model(X_perm)
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                perm_error = RegressionMetrics.mae(output, y)
                scores.append(perm_error)
            importance[name] = float(np.mean(scores) - base_error)

        return importance
    
    def gradient_based_importance(
        self,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
        output_step: Optional[int] = None
    ) -> torch.Tensor:
        """
        基于梯度的特征重要性
        
        通过输入梯度的绝对值评估重要性
        
        Args:
            x: 输入数据
            target_idx: 输出维度索引 (可选)
            output_step: 预测步长索引 (可选)
            
        Returns:
            各特征的梯度重要性
        """
        self.model.eval()
        x = x.to(self.device)
        x.requires_grad_(True)

        output = self.model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]

        if output_step is not None and output.dim() >= 2 and output.size(1) > output_step:
            output = output[:, output_step]
        if target_idx is not None:
            if output.dim() >= 2 and output.size(-1) > target_idx:
                output = output[..., target_idx]
            elif output.dim() == 1 and output.size(0) > target_idx:
                output = output[target_idx]

        self.model.zero_grad()
        output.sum().backward()
        grads = x.grad.abs()

        if grads.dim() >= 2:
            reduce_dims = tuple(range(grads.dim() - 1))
            return grads.mean(dim=reduce_dims)
        return grads
    
    def integrated_gradients(
        self,
        x: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        积分梯度方法
        
        更准确的特征归因方法
        
        Args:
            x: 输入数据
            baseline: 基线输入 (默认为零)
            num_steps: 积分步数
            
        Returns:
            各特征的归因分数
        """
        self.model.eval()
        x = x.to(self.device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)

        scaled_inputs = [
            baseline + (float(i) / num_steps) * (x - baseline)
            for i in range(1, num_steps + 1)
        ]
        gradients = []
        for scaled in scaled_inputs:
            scaled.requires_grad_(True)
            output = self.model(scaled)
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.model.zero_grad()
            output.sum().backward()
            gradients.append(scaled.grad.detach())

        avg_grads = torch.stack(gradients).mean(dim=0)
        return (x - baseline) * avg_grads
    
    def plot_feature_importance(
        self,
        importance_scores: Dict[str, float],
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制特征重要性条形图
        
        Args:
            importance_scores: 特征重要性分数
            title: 图表标题
            save_path: 图片保存路径
            
        Returns:
            matplotlib Figure对象
        """
        items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in items]
        scores = [item[1] for item in items]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(labels, scores)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Importance")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class SHAPExplainer:
    """
    SHAP值解释器
    
    使用SHAP (SHapley Additive exPlanations) 进行模型解释
    """
    
    def __init__(self, model, background_data: torch.Tensor):
        """
        初始化SHAP解释器
        
        Args:
            model: 训练好的模型
            background_data: 背景数据 (用于计算基准)
        """
        self.model = model
        self.background_data = background_data
        
    def explain(self, x: torch.Tensor) -> np.ndarray:
        """
        计算SHAP值
        
        Args:
            x: 待解释的输入数据
            
        Returns:
            SHAP值数组
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required for SHAP explanations.") from exc

        self.model.eval()
        explainer = shap.DeepExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.detach().cpu().numpy()
        return shap_values
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        绘制SHAP摘要图
        
        Args:
            shap_values: SHAP值
            feature_names: 特征名称
            save_path: 图片保存路径
            
        Returns:
            matplotlib Figure对象
        """
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required for SHAP summary plots.") from exc

        shap.summary_plot(shap_values, feature_names=feature_names, show=False)
        fig = plt.gcf()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class InterpretabilityReport:
    """
    可解释性报告生成器
    
    整合多种解释方法，生成综合的模型解释报告
    """
    
    def __init__(
        self,
        model,
        attention_visualizer: Optional[AttentionVisualizer] = None,
        feature_analyzer: Optional[FeatureImportanceAnalyzer] = None
    ):
        """
        初始化报告生成器
        
        Args:
            model: 训练好的模型
            attention_visualizer: 注意力可视化器
            feature_analyzer: 特征重要性分析器
        """
        self.model = model
        self.attention_visualizer = attention_visualizer
        self.feature_analyzer = feature_analyzer
        
    def generate_report(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        feature_names: List[str],
        output_dir: Union[str, Path]
    ) -> Dict:
        """
        生成完整的可解释性报告
        
        Args:
            X: 输入数据
            y: 目标值
            feature_names: 特征名称
            output_dir: 输出目录
            
        Returns:
            报告结果字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report: Dict[str, object] = {}

        if self.attention_visualizer is not None:
            attention_weights = self.attention_visualizer.extract_attention_weights(X)
            report["attention_weights"] = attention_weights

            if "variable" in attention_weights:
                path = output_dir / "attention_variable.png"
                self.attention_visualizer.plot_variable_attention(
                    attention_weights["variable"],
                    variable_names=feature_names,
                    save_path=str(path),
                )
                report["attention_variable_plot"] = str(path)

            if "temporal" in attention_weights or "lstm_temporal" in attention_weights:
                path = output_dir / "attention_temporal.png"
                self.attention_visualizer.plot_temporal_attention(
                    attention_weights,
                    save_path=str(path),
                )
                report["attention_temporal_plot"] = str(path)

        if self.feature_analyzer is not None:
            grad_importance = self.feature_analyzer.gradient_based_importance(X)
            if isinstance(grad_importance, torch.Tensor):
                grad_importance = grad_importance.detach().cpu().numpy()
            grad_importance = np.asarray(grad_importance)
            if grad_importance.ndim > 1:
                grad_importance = grad_importance.mean(axis=0)
            report["gradient_importance"] = {
                name: float(score) for name, score in zip(feature_names, grad_importance)
            }

            perm_importance = self.feature_analyzer.permutation_importance(
                X, y, feature_names
            )
            report["permutation_importance"] = perm_importance

            path = output_dir / "feature_importance.png"
            self.feature_analyzer.plot_feature_importance(
                perm_importance, save_path=str(path)
            )
            report["feature_importance_plot"] = str(path)

        return report


class CounterfactualAnalyzer:
    """
    Counterfactual analysis for feature interventions.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model.to(device) if hasattr(model, "to") else model
        self.device = device

    def analyze_intervention(
        self,
        x: torch.Tensor,
        variable_idx: int,
        reduction_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        self.model.eval()
        x = x.to(self.device)

        with torch.no_grad():
            original_pred = self.model(x)
            if isinstance(original_pred, (tuple, list)):
                original_pred = original_pred[0]

        x_intervention = x.clone()
        x_intervention[..., variable_idx] *= (1 - reduction_ratio)

        with torch.no_grad():
            intervention_pred = self.model(x_intervention)
            if isinstance(intervention_pred, (tuple, list)):
                intervention_pred = intervention_pred[0]

        effect = intervention_pred - original_pred
        effect_percentage = (effect / (original_pred.abs() + 1e-8) * 100).mean()

        return {
            "original": original_pred,
            "intervention": intervention_pred,
            "effect": effect,
            "effect_percentage": effect_percentage
        }

    def sensitivity_analysis(
        self,
        x: torch.Tensor,
        variable_idx: int,
        perturbations: List[float]
    ) -> Dict[str, List[float]]:
        effects = []
        for p in perturbations:
            result = self.analyze_intervention(x, variable_idx, p)
            effects.append(float(result["effect"].mean().item()))
        return {"perturbations": perturbations, "effects": effects}


class AblationStudy:
    """
    Ablation study helper for comparing model variants.
    """

    def __init__(self, model_class, config, test_loader, device: str = "cuda"):
        self.model_class = model_class
        self.config = config
        self.test_loader = test_loader
        self.device = device

    def run_ablation(self) -> Dict[str, Dict]:
        results = {}
        results["full_model"] = self._evaluate_model(
            use_attention=True, use_gated_skip=True
        )
        results["no_attention"] = self._evaluate_model(
            use_attention=False, use_gated_skip=True
        )
        results["no_gated_skip"] = self._evaluate_model(
            use_attention=True, use_gated_skip=False
        )
        results["mtcn_only"] = self._evaluate_model(use_lstm=False)
        results["lstm_only"] = self._evaluate_model(use_mtcn=False)
        return results

    def _clone_config(self):
        return copy.deepcopy(self.config)

    def _apply_overrides(self, cfg, overrides: Dict[str, object]):
        if isinstance(cfg, dict):
            cfg = copy.deepcopy(cfg)
            for key, value in overrides.items():
                if key in cfg:
                    cfg[key] = value
                elif "model" in cfg and key in cfg["model"]:
                    cfg["model"][key] = value
            return cfg

        cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
            elif hasattr(cfg, "model") and hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
        return cfg

    def _build_model(self, cfg):
        try:
            return self.model_class(cfg)
        except TypeError:
            if isinstance(cfg, dict):
                return self.model_class(**cfg)
            if hasattr(cfg, "to_dict"):
                return self.model_class(**cfg.to_dict())
            return self.model_class(**vars(cfg))

    def _evaluate_model(self, **kwargs) -> Dict:
        cfg = self._apply_overrides(self._clone_config(), kwargs)
        model = self._build_model(cfg)
        from .metrics import ModelEvaluator

        evaluator = ModelEvaluator(model, device=self.device)
        return evaluator.evaluate(self.test_loader)
