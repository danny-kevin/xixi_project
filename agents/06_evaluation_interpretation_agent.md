# Agent 06: 评估与可解释性分析 Agent

## 🎯 Agent 角色定义

你是一个**模型评估与可解释性专家**，负责评估模型性能、分析特征重要性、构建多级解释框架。

---

## 📋 核心职责

1. 实现多种评估指标(RMSE, MAE, MAPE, CRPS)
2. 构建多级可解释性框架
3. 注意力权重可视化分析
4. 基于梯度的特征重要性分析
5. 反事实推理与消融实验

---

## 📊 评估指标实现

```python
# 文件: src/evaluation/metrics.py

import numpy as np
import torch
from scipy import stats

class Metrics:
    """评估指标集合"""
    
    @staticmethod
    def rmse(pred, target):
        """均方根误差"""
        return np.sqrt(np.mean((pred - target) ** 2))
    
    @staticmethod
    def mae(pred, target):
        """平均绝对误差"""
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def mape(pred, target, epsilon=1e-8):
        """平均绝对百分比误差"""
        return np.mean(np.abs((target - pred) / (target + epsilon))) * 100
    
    @staticmethod
    def crps(pred_mean, pred_std, target):
        """连续排名概率得分 (假设高斯分布)"""
        z = (target - pred_mean) / pred_std
        crps = pred_std * (z * (2 * stats.norm.cdf(z) - 1) + 
                          2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
        return np.mean(crps)
    
    @staticmethod
    def evaluate_all(pred, target, pred_std=None):
        """计算所有指标"""
        results = {
            'RMSE': Metrics.rmse(pred, target),
            'MAE': Metrics.mae(pred, target),
            'MAPE': Metrics.mape(pred, target)
        }
        if pred_std is not None:
            results['CRPS'] = Metrics.crps(pred, pred_std, target)
        return results
```

---

## 🔍 多级可解释性框架

### 第一级: 注意力权重分析

```python
# 文件: src/evaluation/interpretability.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionAnalyzer:
    """注意力权重分析器"""
    
    def __init__(self, model, variable_names):
        self.model = model
        self.variable_names = variable_names
        
    def get_attention_weights(self, x):
        """提取注意力权重"""
        self.model.eval()
        with torch.no_grad():
            _, attention = self.model(x, return_attention=True)
        return attention
    
    def plot_variable_importance(self, attention_weights, save_path=None):
        """可视化变量重要性"""
        # 计算每个变量的平均注意力得分
        importance = attention_weights.mean(dim=(0, 1)).cpu().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.barh(self.variable_names, importance)
        plt.xlabel('重要性得分')
        plt.title('变量重要性 (基于注意力权重)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
```

### 第二级: 梯度特征重要性

```python
class GradientAnalyzer:
    """基于梯度的特征重要性分析"""
    
    def __init__(self, model):
        self.model = model
        
    def compute_saliency(self, x, target_idx=0):
        """计算显著性图"""
        self.model.eval()
        x.requires_grad = True
        
        output = self.model(x)
        
        # 对目标输出计算梯度
        self.model.zero_grad()
        output[:, target_idx].sum().backward()
        
        saliency = x.grad.abs()
        return saliency
    
    def integrated_gradients(self, x, baseline=None, steps=50):
        """积分梯度方法"""
        if baseline is None:
            baseline = torch.zeros_like(x)
        
        # 生成插值路径
        alphas = torch.linspace(0, 1, steps)
        gradients = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)
            self.model.zero_grad()
            output.sum().backward()
            
            gradients.append(interpolated.grad.clone())
        
        # 积分
        avg_gradients = torch.stack(gradients).mean(dim=0)
        integrated_grad = (x - baseline) * avg_gradients
        
        return integrated_grad
```

### 第三级: 反事实推理

```python
class CounterfactualAnalyzer:
    """反事实推理分析"""
    
    def __init__(self, model):
        self.model = model
        
    def analyze_intervention(self, x, variable_idx, reduction_ratio=0.5):
        """分析干预效果
        
        例如：如果人口流动减少50%，预测结果如何变化？
        """
        self.model.eval()
        
        # 原始预测
        with torch.no_grad():
            original_pred = self.model(x)
        
        # 干预后预测
        x_intervention = x.clone()
        x_intervention[:, :, variable_idx] *= (1 - reduction_ratio)
        
        with torch.no_grad():
            intervention_pred = self.model(x_intervention)
        
        # 计算变化
        effect = intervention_pred - original_pred
        
        return {
            'original': original_pred,
            'intervention': intervention_pred,
            'effect': effect,
            'effect_percentage': (effect / original_pred * 100).mean()
        }
    
    def sensitivity_analysis(self, x, variable_idx, perturbations):
        """敏感性分析"""
        effects = []
        
        for p in perturbations:
            result = self.analyze_intervention(x, variable_idx, p)
            effects.append(result['effect'].mean().item())
        
        return {'perturbations': perturbations, 'effects': effects}
```

---

## 🧪 消融实验

```python
class AblationStudy:
    """消融实验 - 评估各组件贡献"""
    
    def __init__(self, model_class, config, test_loader):
        self.model_class = model_class
        self.config = config
        self.test_loader = test_loader
        
    def run_ablation(self):
        """运行消融实验"""
        results = {}
        
        # 完整模型
        results['full_model'] = self._evaluate_model(
            use_attention=True, use_gated_skip=True
        )
        
        # 无注意力机制
        results['no_attention'] = self._evaluate_model(
            use_attention=False, use_gated_skip=True
        )
        
        # 无门控跳跃连接
        results['no_gated_skip'] = self._evaluate_model(
            use_attention=True, use_gated_skip=False
        )
        
        # 仅M-TCN
        results['mtcn_only'] = self._evaluate_model(
            use_lstm=False
        )
        
        # 仅LSTM
        results['lstm_only'] = self._evaluate_model(
            use_mtcn=False
        )
        
        return results
    
    def _evaluate_model(self, **kwargs):
        """评估特定配置的模型"""
        # 根据kwargs构建并评估模型
        pass
```

---

## 📈 统计显著性检验

```python
from scipy.stats import wilcoxon

def diebold_mariano_test(errors1, errors2, h=1):
    """Diebold-Mariano检验 - 比较预测性能差异"""
    d = errors1 ** 2 - errors2 ** 2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    
    # 自相关调整
    gamma = []
    for k in range(1, h):
        gamma.append(np.cov(d[:-k], d[k:])[0, 1])
    
    adjusted_var = var_d + 2 * sum(gamma)
    
    dm_stat = mean_d / np.sqrt(adjusted_var / len(d))
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {'dm_statistic': dm_stat, 'p_value': p_value}
```

---

## 📊 结果报告模板

```python
def generate_report(model_name, metrics, attention_analysis, ablation_results):
    """生成评估报告"""
    report = f"""
# 模型评估报告: {model_name}

## 1. 预测性能
| 指标 | 值 |
|------|-----|
| RMSE | {metrics['RMSE']:.4f} |
| MAE  | {metrics['MAE']:.4f} |
| MAPE | {metrics['MAPE']:.2f}% |

## 2. 变量重要性 (Top 5)
{attention_analysis}

## 3. 消融实验结果
{ablation_results}

## 4. 结论与建议
...
"""
    return report
```

---

## 📝 配置参数

```yaml
evaluation:
  metrics: ["RMSE", "MAE", "MAPE", "CRPS"]
  
interpretability:
  attention_visualization: true
  gradient_analysis: true
  counterfactual_analysis: true
  ablation_study: true
  
counterfactual:
  perturbations: [0.1, 0.25, 0.5, 0.75]
```

---

## ⚠️ 注意事项

1. **多方法验证**: 用梯度分析验证注意力权重可靠性
2. **统计检验**: 使用DM检验确认性能差异显著性
3. **因果解释**: 反事实分析仅提供相关性，非因果关系
4. **可视化**: 注意力热图应包含置信区间
