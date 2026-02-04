"""
评估模块
Evaluation Module

包含评估指标和可解释性分析工具
"""

from .metrics import RegressionMetrics, TimeSeriesMetrics, ModelEvaluator, diebold_mariano_test
from .interpretability import (
    AttentionVisualizer,
    FeatureImportanceAnalyzer,
    InterpretabilityReport,
    CounterfactualAnalyzer,
    AblationStudy,
    SHAPExplainer,
)

__all__ = [
    'RegressionMetrics',
    'TimeSeriesMetrics',
    'ModelEvaluator',
    'diebold_mariano_test',
    'AttentionVisualizer',
    'FeatureImportanceAnalyzer',
    'InterpretabilityReport',
    'CounterfactualAnalyzer',
    'AblationStudy',
    'SHAPExplainer'
]
