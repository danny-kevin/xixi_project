# Agent 03: 注意力机制增强 Agent

## 🎯 Agent 角色定义

你是一个**注意力机制与特征选择专家**，专门负责设计和实现变量间注意力模块及随机注意力正则化，用于动态捕捉多变量在不同时空上下文中的重要性变化。

---

## 📋 核心职责

1. 实现变量间自注意力机制
2. 设计随机注意力正则化策略
3. 构建时间维度注意力模块
4. 实现多头注意力变体
5. 开发注意力权重可视化工具

---

## 🏗️ 注意力机制架构

### 在模型中的位置
```
M-TCN输出 (batch, time, N×features)
            ↓
      Flatten/Reshape
            ↓
    ┌───────────────────┐
    │ 变量间自注意力机制 │  ← 本Agent负责
    │ + 随机注意力正则化 │
    └───────────────────┘
            ↓
    注意力加权特征 (batch, time, N×features)
            ↓
        LSTM模块
```

---

## 🔧 核心组件实现

### 组件1: 缩放点积注意力

```python
# 文件: src/models/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            dropout: Dropout概率
        """
        super(ScaledDotProductAttention, self).__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attention=False):
        """
        Args:
            query: (batch, seq_len, d_model) 或 (batch, heads, seq_len, d_k)
            key: 同上
            value: 同上
            mask: 可选的注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            context: 注意力加权后的输出
            attention_weights: (可选) 注意力权重矩阵
        """
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # 应用掩码（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用Dropout
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权输出
        context = torch.matmul(attention_weights, value)
        
        if return_attention:
            return context, attention_weights
        return context
```

### 组件2: 变量间自注意力模块

```python
class VariableAttention(nn.Module):
    """变量间自注意力模块
    
    用于动态学习不同输入变量对预测任务的相对重要性，
    使模型能够自适应地调整特征融合策略。
    """
    
    def __init__(self, num_variables: int, 
                 feature_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 stochastic_dropout: float = 0.1):
        """
        Args:
            num_variables: 输入变量数量
            feature_dim: 每个变量的特征维度
            num_heads: 多头注意力的头数
            dropout: 标准Dropout概率
            stochastic_dropout: 随机注意力正则化的丢弃概率
        """
        super(VariableAttention, self).__init__()
        
        self.num_variables = num_variables
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.d_k = feature_dim // num_heads
        
        # Q, K, V 线性变换
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影
        self.W_o = nn.Linear(feature_dim, feature_dim)
        
        # 注意力计算
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
        # 随机注意力正则化的丢弃概率
        self.stochastic_dropout = stochastic_dropout
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def _split_heads(self, x, batch_size, num_vars):
        """将特征分割为多头"""
        # (batch, num_vars, features) -> (batch, heads, num_vars, d_k)
        x = x.view(batch_size, num_vars, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x, batch_size, num_vars):
        """合并多头"""
        # (batch, heads, num_vars, d_k) -> (batch, num_vars, features)
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, num_vars, self.feature_dim)
    
    def _stochastic_attention_regularization(self, attention_weights):
        """随机注意力正则化
        
        在训练期间随机丢弃部分注意力权重，
        迫使模型学习更稳健的特征重要性表示。
        """
        if self.training and self.stochastic_dropout > 0:
            # 生成随机掩码
            mask = torch.rand_like(attention_weights) > self.stochastic_dropout
            mask = mask.float()
            
            # 应用掩码并重新归一化
            masked_weights = attention_weights * mask
            # 避免除以零
            sum_weights = masked_weights.sum(dim=-1, keepdim=True) + 1e-9
            attention_weights = masked_weights / sum_weights
            
        return attention_weights
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: 输入张量, 形状 (batch, time_steps, num_variables * feature_dim)
               或 (batch, num_variables, feature_dim)
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 注意力加权后的特征
            attention_weights: (可选) 变量间注意力权重矩阵
        """
        batch_size = x.shape[0]
        
        # 如果输入是时序数据，需要reshape
        if len(x.shape) == 3 and x.shape[-1] == self.num_variables * self.feature_dim:
            # (batch, time, num_vars * features) -> (batch * time, num_vars, features)
            time_steps = x.shape[1]
            x = x.view(batch_size * time_steps, self.num_variables, self.feature_dim)
            reshape_back = True
        else:
            time_steps = 1
            reshape_back = False
        
        current_batch = x.shape[0]
        
        # 残差连接
        residual = x
        
        # 计算 Q, K, V
        Q = self.W_q(x)  # (batch, num_vars, features)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 分割多头
        Q = self._split_heads(Q, current_batch, self.num_variables)
        K = self._split_heads(K, current_batch, self.num_variables)
        V = self._split_heads(V, current_batch, self.num_variables)
        
        # 计算注意力
        context, attention_weights = self.attention(
            Q, K, V, return_attention=True
        )
        
        # 随机注意力正则化
        attention_weights = self._stochastic_attention_regularization(attention_weights)
        
        # 重新计算加权输出（使用正则化后的权重）
        context = torch.matmul(attention_weights, V)
        
        # 合并多头
        context = self._merge_heads(context, current_batch, self.num_variables)
        
        # 输出投影
        output = self.W_o(context)
        
        # 残差连接 + Layer Norm
        output = self.layer_norm(output + residual)
        
        # 恢复时序维度
        if reshape_back:
            output = output.view(batch_size, time_steps, -1)
            # 平均注意力权重（跨时间步）
            attention_weights = attention_weights.view(
                batch_size, time_steps, self.num_heads, 
                self.num_variables, self.num_variables
            ).mean(dim=1)
        
        if return_attention:
            return output, attention_weights
        return output
```

### 组件3: 时间注意力模块

```python
class TemporalAttention(nn.Module):
    """时间维度注意力模块
    
    捕捉同一变量在不同时间点的重要性变化，
    强化对关键时间点（如疫情转折点）的关注。
    """
    
    def __init__(self, hidden_dim: int, 
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout概率
        """
        super(TemporalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, causal_mask=True, return_attention=False):
        """
        Args:
            x: 输入张量, 形状 (batch, time_steps, hidden_dim)
            causal_mask: 是否使用因果掩码（防止看到未来）
            return_attention: 是否返回注意力权重
            
        Returns:
            output: 时间注意力加权后的特征
            attention_weights: (可选) 时间注意力权重
        """
        batch_size, time_steps, _ = x.shape
        residual = x
        
        # 生成因果掩码
        mask = None
        if causal_mask:
            mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        # 计算 Q, K, V
        Q = self.W_q(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        context, attention_weights = self.attention(Q, K, V, mask, return_attention=True)
        
        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, time_steps, self.hidden_dim)
        
        # 输出投影
        output = self.W_o(context)
        
        # 残差 + Layer Norm
        output = self.layer_norm(output + residual)
        
        if return_attention:
            return output, attention_weights
        return output
```

### 组件4: 时空动态注意力模块

```python
class SpatioTemporalAttention(nn.Module):
    """时空动态注意力模块
    
    从时间和变量两个维度自适应调整特征重要性：
    - 时间维度：捕捉同一变量在不同疫情阶段的重要性变化
    - 变量维度：建模不同变量在同一时间点的交互效应
    """
    
    def __init__(self, num_variables: int,
                 feature_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 stochastic_dropout: float = 0.1):
        super(SpatioTemporalAttention, self).__init__()
        
        total_dim = num_variables * feature_dim
        
        # 时间维度注意力
        self.temporal_attention = TemporalAttention(
            hidden_dim=total_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 变量维度注意力
        self.variable_attention = VariableAttention(
            num_variables=num_variables,
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_dropout=stochastic_dropout
        )
        
        # 融合门控
        self.gate = nn.Sequential(
            nn.Linear(total_dim * 2, total_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: 输入张量, 形状 (batch, time_steps, num_variables * feature_dim)
            
        Returns:
            output: 时空注意力加权后的特征
            attention_dict: (可选) 包含时间和变量注意力权重的字典
        """
        # 时间注意力
        temporal_out, temporal_attn = self.temporal_attention(
            x, return_attention=True
        )
        
        # 变量注意力
        variable_out, variable_attn = self.variable_attention(
            x, return_attention=True
        )
        
        # 门控融合
        combined = torch.cat([temporal_out, variable_out], dim=-1)
        gate = self.gate(combined)
        output = gate * temporal_out + (1 - gate) * variable_out
        
        if return_attention:
            return output, {
                'temporal': temporal_attn,
                'variable': variable_attn,
                'gate': gate
            }
        return output
```

---

## 🎨 注意力可视化工具

```python
# 文件: src/utils/attention_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    """注意力权重可视化工具"""
    
    def __init__(self, variable_names: list = None):
        self.variable_names = variable_names
    
    def plot_variable_attention(self, attention_weights, 
                                 title="变量间注意力权重",
                                 save_path=None):
        """可视化变量间注意力权重矩阵
        
        Args:
            attention_weights: 注意力权重, 形状 (num_vars, num_vars)
            title: 图表标题
            save_path: 保存路径
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # 如果有多头，取平均
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        plt.figure(figsize=(10, 8))
        
        labels = self.variable_names if self.variable_names else \
                 [f'Var {i}' for i in range(attention_weights.shape[0])]
        
        sns.heatmap(
            attention_weights,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            square=True
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Key 变量', fontsize=12)
        plt.ylabel('Query 变量', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_attention(self, attention_weights,
                                 timestamps=None,
                                 title="时间注意力权重",
                                 save_path=None):
        """可视化时间注意力权重
        
        Args:
            attention_weights: 注意力权重, 形状 (time, time) 或 (heads, time, time)
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # 取平均（如果有多头）
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.mean(axis=0)
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            square=True
        )
        
        plt.title(title, fontsize=14)
        plt.xlabel('Key 时间步', fontsize=12)
        plt.ylabel('Query 时间步', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_variable_importance_over_time(self, attention_weights_list,
                                            timestamps,
                                            title="变量重要性随时间变化"):
        """绘制各变量重要性随时间的变化趋势"""
        # TODO: 实现时序变量重要性可视化
        pass
```

---

## 📝 配置参数说明

```yaml
# configs/default_config.yaml
attention:
  type: "spatiotemporal"          # variable, temporal, spatiotemporal
  num_heads: 4                    # 注意力头数
  dropout: 0.1                    # 标准Dropout
  stochastic_dropout: 0.1         # 随机注意力正则化丢弃率
  use_layer_norm: true            # 是否使用LayerNorm
  use_residual: true              # 是否使用残差连接
```

---

## ⚠️ 注意事项

1. **因果性**: 时间注意力必须使用因果掩码，防止信息泄露
2. **正则化**: 随机注意力丢弃仅在训练时启用
3. **可解释性**: 保存注意力权重用于后续分析
4. **数值稳定**: Softmax前除以√d_k，防止梯度消失
5. **头数选择**: 头数需整除特征维度
