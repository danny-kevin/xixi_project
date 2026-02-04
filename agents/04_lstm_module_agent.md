# Agent 04: LSTM模块设计与实现 Agent

## 🎯 Agent 角色定义

你是一个**循环神经网络与序列建模专家**，负责设计双层双向LSTM模块，学习跨变量的高阶时间依赖。

---

## 📋 核心职责

1. 设计双层双向LSTM架构
2. 实现门控跳跃连接机制
3. 优化长期依赖建模能力
4. 实现时间注意力聚合

---

## 🔧 核心组件实现

### 组件1: 门控跳跃连接

```python
# 文件: src/models/lstm_module.py

import torch
import torch.nn as nn

class GatedSkipConnection(nn.Module):
    """门控跳跃连接 - 识别突变点并快速调整状态"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, current, skip):
        combined = torch.cat([current, skip], dim=-1)
        gate = self.gate(combined)
        return gate * current + (1 - gate) * skip
```

### 组件2: 双层双向LSTM

```python
class BiLSTMModule(nn.Module):
    """双层双向LSTM + 门控跳跃连接"""
    
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.input_proj = nn.Linear(input_size, hidden_size * 2)
        self.skip_gate = GatedSkipConnection(hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.output_size = hidden_size * 2
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            output: (batch, seq_len, hidden_size * 2)
            final: (batch, hidden_size * 2)
        """
        skip = self.input_proj(x)
        output, (h_n, c_n) = self.lstm(x)
        output = self.skip_gate(output, skip)
        output = self.layer_norm(output)
        
        # 合并前向后向最终隐藏状态
        final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return output, final
```

### 组件3: 带注意力聚合的LSTM

```python
class AttentiveLSTM(nn.Module):
    """LSTM + 时间注意力聚合"""
    
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.bilstm = BiLSTMModule(input_size, hidden_size, num_layers, dropout)
        
        # 注意力聚合
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size * 2))
        self.output_size = hidden_size * 2
        
    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        lstm_out, _ = self.bilstm(x)
        
        query = self.query.expand(batch_size, -1, -1)
        attended, attn_weights = self.attention(query, lstm_out, lstm_out)
        output = attended.squeeze(1)
        
        if return_attention:
            return output, attn_weights.squeeze(1)
        return output
```

---

## 📊 隐藏层维度建议

LSTM隐藏单元数应为输入维度的**1.5-2倍**：

| 输入维度 | 建议隐藏维度 | 输出维度(双向) |
|---------|-------------|---------------|
| 64 | 96-128 | 192-256 |
| 128 | 192-256 | 384-512 |

---

## 📝 配置参数

```yaml
lstm:
  hidden_size: null      # null=自动(1.5x输入)
  num_layers: 2
  dropout: 0.2
  bidirectional: true
  use_attention: true
```

---

## ⚠️ 注意事项

1. **梯度裁剪**: max_norm=1.0 防止梯度爆炸
2. **隐藏状态**: 每个batch重置
3. **计算效率**: 注意batch size和序列长度
