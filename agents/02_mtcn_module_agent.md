# Agent 02: M-TCN模块设计与实现 Agent

## 🎯 Agent 角色定义

你是一个**时间卷积网络(TCN)架构专家**，专门负责设计和实现M-TCN（多头分离式时间卷积网络）模块，用于提取多变量时间序列的变量特异性特征。

---

## 📋 核心职责

1. 设计因果卷积层和扩张卷积层
2. 实现残差块结构
3. 构建多头并行TCN子网络
4. 优化感受野覆盖范围
5. 实现层次化扩张卷积策略

---

## 🏗️ M-TCN架构详解

### 整体结构
```
M-TCN模块
├── 变量1 → TCN子网络1 (独立处理)
├── 变量2 → TCN子网络2 (独立处理)
├── 变量3 → TCN子网络3 (独立处理)
├── ...
└── 变量N → TCN子网络N (独立处理)
          ↓
    特征拼接 (Concatenation)
          ↓
    输出: (batch, time_steps, N × feature_dim)
```

### 单个TCN子网络结构
```
输入: (batch, time_steps, 1)
        ↓
残差块1 (扩张系数 d=1)  → 感受野: 1-3天
        ↓
残差块2 (扩张系数 d=2)  → 感受野: 4-7天
        ↓
残差块3 (扩张系数 d=4)  → 感受野: 8-14天
        ↓
残差块4 (扩张系数 d=8)  → 感受野: 15-28天
        ↓
输出: (batch, time_steps, feature_dim)
```

---

## 🔧 核心组件实现

### 组件1: 因果卷积层

```python
# 文件: src/models/tcn.py

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class CausalConv1d(nn.Module):
    """因果卷积层 - 确保不使用未来信息"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dilation: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 扩张系数
        """
        super(CausalConv1d, self).__init__()
        
        # 计算因果填充：确保输出只依赖于过去和当前的输入
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = weight_norm(nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        ))
        
    def forward(self, x):
        """
        Args:
            x: 输入张量, 形状 (batch, channels, time_steps)
        Returns:
            输出张量, 形状 (batch, out_channels, time_steps)
        """
        out = self.conv(x)
        # 移除右侧填充，确保因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out
```

### 组件2: 残差块

```python
class ResidualBlock(nn.Module):
    """TCN残差块 - 包含两层因果卷积和残差连接"""
    
    def __init__(self, n_inputs: int, n_outputs: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        """
        Args:
            n_inputs: 输入通道数
            n_outputs: 输出通道数
            kernel_size: 卷积核大小
            dilation: 扩张系数
            dropout: Dropout概率
        """
        super(ResidualBlock, self).__init__()
        
        # 第一层因果卷积
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层因果卷积
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接：如果输入输出通道数不同，需要1x1卷积调整
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: 输入张量, 形状 (batch, n_inputs, time_steps)
        Returns:
            输出张量, 形状 (batch, n_outputs, time_steps)
        """
        # 主路径
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        
        return self.relu(out + res)
```

### 组件3: 单变量TCN子网络

```python
class TCNSubNetwork(nn.Module):
    """单变量TCN子网络 - 提取单个变量的时间特征"""
    
    def __init__(self, input_size: int = 1, 
                 num_channels: list = [32, 32, 32, 32],
                 kernel_size: int = 3, 
                 dropout: float = 0.2):
        """
        Args:
            input_size: 输入特征维度（单变量为1）
            num_channels: 每个残差块的输出通道数列表
            kernel_size: 卷积核大小
            dropout: Dropout概率
        """
        super(TCNSubNetwork, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            # 扩张系数按2的指数增长: 1, 2, 4, 8, ...
            dilation = 2 ** i
            
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(ResidualBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout
            ))
        
        self.network = nn.Sequential(*layers)
        self.receptive_field = self._calculate_receptive_field(
            num_levels, kernel_size
        )
        
    def _calculate_receptive_field(self, num_levels: int, kernel_size: int) -> int:
        """计算感受野大小"""
        # 感受野 = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(num_levels))
        return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量, 形状 (batch, time_steps, 1)
        Returns:
            输出张量, 形状 (batch, time_steps, num_channels[-1])
        """
        # 调整维度: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        out = self.network(x)
        # 还原维度: (batch, channels, time) -> (batch, time, channels)
        return out.transpose(1, 2)
```

### 组件4: 完整M-TCN模块

```python
class MTCN(nn.Module):
    """多头分离式时间卷积网络 (M-TCN)
    
    每个输入变量通过独立的TCN子网络处理，
    提取变量特异性的时间模式，然后拼接输出。
    """
    
    def __init__(self, num_variables: int,
                 num_channels: list = [32, 32, 32, 32],
                 kernel_size: int = 3,
                 dropout: float = 0.2,
                 share_weights: bool = False):
        """
        Args:
            num_variables: 输入变量数量
            num_channels: 每个残差块的通道数
            kernel_size: 卷积核大小
            dropout: Dropout概率
            share_weights: 是否共享各子网络的权重
        """
        super(MTCN, self).__init__()
        
        self.num_variables = num_variables
        self.share_weights = share_weights
        
        if share_weights:
            # 所有变量共享同一个TCN网络
            self.shared_tcn = TCNSubNetwork(
                input_size=1,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout
            )
        else:
            # 每个变量有独立的TCN网络
            self.tcn_list = nn.ModuleList([
                TCNSubNetwork(
                    input_size=1,
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    dropout=dropout
                ) for _ in range(num_variables)
            ])
        
        # 输出特征维度
        self.output_dim = num_variables * num_channels[-1]
        
        # 打印感受野信息
        sample_tcn = self.shared_tcn if share_weights else self.tcn_list[0]
        print(f"M-TCN感受野: {sample_tcn.receptive_field} 时间步")
        
    def forward(self, x):
        """
        Args:
            x: 输入张量, 形状 (batch, time_steps, num_variables)
        Returns:
            输出张量, 形状 (batch, time_steps, num_variables * channel_dim)
        """
        batch_size, time_steps, _ = x.shape
        
        outputs = []
        for i in range(self.num_variables):
            # 提取第i个变量: (batch, time, 1)
            var_input = x[:, :, i:i+1]
            
            if self.share_weights:
                var_output = self.shared_tcn(var_input)
            else:
                var_output = self.tcn_list[i](var_input)
            
            outputs.append(var_output)
        
        # 拼接所有变量的输出: (batch, time, num_vars * channels)
        concatenated = torch.cat(outputs, dim=-1)
        
        return concatenated
```

---

## 📊 层次化扩张卷积策略

针对传染病的多尺度滞后效应，设计层次化感受野：

```python
class HierarchicalMTCN(nn.Module):
    """层次化M-TCN - 分层捕捉不同时间尺度的模式"""
    
    def __init__(self, num_variables: int):
        super(HierarchicalMTCN, self).__init__()
        
        # 短期模式捕捉 (1-3天) - 扩张系数: [1]
        self.short_term = MTCN(
            num_variables=num_variables,
            num_channels=[16, 16],
            kernel_size=3,
            dropout=0.1
        )
        
        # 中期模式捕捉 (7-10天) - 扩张系数: [1, 2, 4]
        self.medium_term = MTCN(
            num_variables=num_variables,
            num_channels=[32, 32, 32],
            kernel_size=3,
            dropout=0.15
        )
        
        # 长期模式捕捉 (14-21天) - 扩张系数: [1, 2, 4, 8]
        self.long_term = MTCN(
            num_variables=num_variables,
            num_channels=[32, 32, 32, 32],
            kernel_size=3,
            dropout=0.2
        )
        
        # 融合层
        total_features = (self.short_term.output_dim + 
                         self.medium_term.output_dim + 
                         self.long_term.output_dim)
        self.fusion = nn.Linear(total_features, 128)
        
    def forward(self, x):
        short = self.short_term(x)
        medium = self.medium_term(x)
        long = self.long_term(x)
        
        # 多尺度特征融合
        fused = torch.cat([short, medium, long], dim=-1)
        return self.fusion(fused)
```

---

## 🧮 感受野计算公式

对于标准TCN：
```
感受野 = 1 + Σ(kernel_size - 1) × dilation_i × 2
       = 1 + 2 × (kernel_size - 1) × (2^num_layers - 1)
```

| 层数 | 扩张系数 | 感受野(kernel=3) |
|-----|---------|-----------------|
| 1层 | [1] | 5 |
| 2层 | [1,2] | 9 |
| 3层 | [1,2,4] | 17 |
| 4层 | [1,2,4,8] | 33 |
| 5层 | [1,2,4,8,16] | 65 |

**推荐配置**: 4层残差块，感受野33天，可覆盖14-21天的滞后周期。

---

## 📝 配置参数说明

```yaml
# configs/mtcn_config.yaml
mtcn:
  num_variables: 10           # 输入变量数量
  num_channels: [32, 32, 32, 32]  # 各残差块通道数
  kernel_size: 3              # 卷积核大小
  dropout: 0.2                # Dropout概率
  share_weights: false        # 是否共享权重
```

---

## ⚠️ 注意事项

1. **因果性保证**: 确保所有卷积操作只使用历史信息
2. **感受野覆盖**: 感受野必须覆盖最长滞后周期（21天以上）
3. **梯度稳定**: 使用权重归一化防止梯度消失/爆炸
4. **内存优化**: 对于大量变量，考虑分批处理或共享权重
5. **输入验证**: 确保输入序列长度大于感受野
