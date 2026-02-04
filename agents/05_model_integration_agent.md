# Agent 05: 模型整合与训练 Agent

## 🎯 Agent 角色定义

你是一个**深度学习训练与优化专家**，负责将M-TCN、注意力层、LSTM整合为完整模型，并实现多阶段训练策略。

---

## 📋 核心职责

1. 整合各子模块为完整混合模型
2. 实现多阶段训练策略（预训练+微调）
3. 设计损失函数与正则化
4. 配置优化器和学习率调度
5. 实现时序交叉验证

---

## 🏗️ 完整模型架构

```python
# 文件: src/models/hybrid_model.py

import torch
import torch.nn as nn
from .mtcn import MTCN
from .attention import SpatioTemporalAttention
from .lstm_module import AttentiveLSTM

class MTCNLSTMHybrid(nn.Module):
    """注意力增强M-TCN-LSTM混合模型"""
    
    def __init__(self, num_variables: int, tcn_channels: list = [32,32,32,32],
                 lstm_hidden: int = 128, output_steps: int = 7, dropout: float = 0.2):
        super().__init__()
        
        # M-TCN模块
        self.mtcn = MTCN(
            num_variables=num_variables,
            num_channels=tcn_channels,
            kernel_size=3,
            dropout=dropout
        )
        
        mtcn_out_dim = num_variables * tcn_channels[-1]
        
        # 注意力增强层
        self.attention = SpatioTemporalAttention(
            num_variables=num_variables,
            feature_dim=tcn_channels[-1],
            num_heads=4,
            dropout=dropout,
            stochastic_dropout=0.1
        )
        
        # LSTM模块
        self.lstm = AttentiveLSTM(
            input_size=mtcn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            dropout=dropout
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, output_steps)
        )
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, time_steps, num_variables)
        Returns:
            predictions: (batch, output_steps)
        """
        # M-TCN特征提取
        tcn_out = self.mtcn(x)
        
        # 注意力增强
        attn_out, attn_weights = self.attention(tcn_out, return_attention=True)
        
        # LSTM序列建模
        lstm_out, time_attn = self.lstm(attn_out, return_attention=True)
        
        # 预测输出
        predictions = self.output_layer(lstm_out)
        
        if return_attention:
            return predictions, {'variable': attn_weights, 'temporal': time_attn}
        return predictions
```

---

## 📊 损失函数设计

```python
# 文件: src/training/loss.py

import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    """混合损失函数 = RMSE + 时序一致性正则"""
    
    def __init__(self, consistency_weight: float = 0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.consistency_weight = consistency_weight
        
    def forward(self, pred, target):
        # RMSE损失
        rmse_loss = torch.sqrt(self.mse(pred, target))
        
        # 时序一致性正则（惩罚非生理性震荡）
        if pred.shape[1] > 1:
            diff = pred[:, 1:] - pred[:, :-1]
            consistency_loss = torch.mean(diff ** 2)
        else:
            consistency_loss = 0
        
        return rmse_loss + self.consistency_weight * consistency_loss
```

---

## 🔄 多阶段训练策略

```python
# 文件: src/training/trainer.py

class MultiStageTrainer:
    """多阶段训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def pretrain_mtcn(self, train_loader, epochs=10):
        """阶段1: 预训练M-TCN"""
        # 冻结LSTM，只训练M-TCN
        for param in self.model.lstm.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001
        )
        
        for epoch in range(epochs):
            self._train_epoch(train_loader, optimizer)
            
        # 解冻
        for param in self.model.lstm.parameters():
            param.requires_grad = True
            
    def pretrain_lstm(self, train_loader, epochs=10):
        """阶段2: 预训练LSTM"""
        # 冻结M-TCN
        for param in self.model.mtcn.parameters():
            param.requires_grad = False
            
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001
        )
        
        for epoch in range(epochs):
            self._train_epoch(train_loader, optimizer)
            
        # 解冻
        for param in self.model.mtcn.parameters():
            param.requires_grad = True
            
    def finetune(self, train_loader, val_loader, epochs=50):
        """阶段3: 端到端联合微调"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer)
            val_loss = self._validate(val_loader)
            scheduler.step()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint()
                
    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total_loss = 0
        criterion = HybridLoss()
        
        for x, y in loader:
            optimizer.zero_grad()
            pred = self.model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)
```

---

## 📈 时序交叉验证

```python
class TimeSeriesCV:
    """时序交叉验证 - 防止未来信息泄露"""
    
    def __init__(self, n_splits=5, val_ratio=0.2):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        
    def split(self, data):
        n = len(data)
        fold_size = n // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + int(fold_size * self.val_ratio), n)
            
            yield {
                'train': (0, train_end),
                'val': (train_end, val_end)
            }
```

---

## 📝 训练配置

```yaml
# configs/training_config.yaml
training:
  pretrain_mtcn_epochs: 10
  pretrain_lstm_epochs: 10
  finetune_epochs: 50
  batch_size: 32
  
optimizer:
  type: "AdamW"
  lr: 0.001
  weight_decay: 0.01
  
scheduler:
  type: "CosineAnnealing"
  T_max: 50
  
loss:
  consistency_weight: 0.1
  
gradient:
  clip_norm: 1.0
```

---

## ⚠️ 注意事项

1. **多阶段顺序**: 先M-TCN → 再LSTM → 最后联合微调
2. **学习率**: 微调阶段使用更小学习率(1e-4)
3. **梯度裁剪**: 防止梯度爆炸
4. **早停**: 验证损失不下降时停止
5. **时序验证**: 严格按时间顺序划分数据
