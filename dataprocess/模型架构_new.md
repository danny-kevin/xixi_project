# 模型架构概述（代码实现版）

本文档是对当前代码中模型实现的结构化总结，便于快速复现与沟通。
模型对应 `src/models/hybrid_model.py` 的 `AttentionMTCNLSTM`。

---

## 1. 输入输出与张量形状

- 输入：`X ∈ R^{B×T×V}`，其中 `T=12`、`V=8`
- 输出：`ŷ ∈ R^{B×T_out×1}`，其中 `T_out=4`

---

## 2. 模型总体流程

1. **M‑TCN（每变量独立 TCN）**：为每个变量提取局部时序特征
2. **时空注意力（Temporal + Variable + Gate）**：融合时间与变量维度信息
3. **注意力 LSTM**：双向 LSTM + MultiheadAttention 汇聚
4. **MLP 输出头**：直接输出多步预测

---

## 3. M‑TCN 细节

- 对每个变量 `v`，独立一套 TCN 子网络
- 单变量 TCN 由多层因果卷积残差块组成（dilation 递增）
- 每个变量输出形状：`(B, T, d)`
- 拼接后总输出：`H ∈ R^{B×T×(V·d)}`

---

## 4. 时空注意力模块

包含两条分支：

- **TemporalAttention**：在时间维度做 self‑attention（因果 mask）
- **VariableAttention**：在变量维度做 multi‑head attention

两条分支通过门控融合：

```
A = gate * temporal_out + (1 - gate) * variable_out
```

输出形状保持：`(B, T, V·d)`

---

## 5. LSTM 模块

- **双向 LSTM** 输出：`(B, T, 2h)`
- 使用 learnable query 的 MultiheadAttention 做序列汇聚
- 得到上下文向量 `context ∈ R^{B×2h}`

---

## 6. 输出头

- 两层 MLP：`Linear(2h → h) → ReLU → Dropout → Linear(h → T_out)`
- reshape 到 `ŷ ∈ R^{B×T_out×1}`

---

## 7. 逐层参数尺寸表（贴近代码）

记：`V=8`, `T=12`, `d = tcn_channels[-1]`, `H = attention_embed_dim`, `h = lstm_hidden_size`

| 层级 | 输入张量 | 核心参数 | 输出张量 |
|---|---|---|---|
| 输入 | `(B, 12, 8)` | - | `(B, 12, 8)` |
| M‑TCN 单变量 | `(B, 12, 1)` | `tcn_channels=[.., d]` | `(B, 12, d)` |
| M‑TCN 拼接 | 8 个 `(B,12,d)` | concat | `(B, 12, 8d)` |
| 投影（可选） | `(B, 12, 8d)` | `Linear(d→H)`/变量内投影 | `(B, 12, 8H)` |
| TemporalAttention | `(B, 12, 8H)` | `num_heads` | `(B, 12, 8H)` |
| VariableAttention | `(B, 12, 8H)` | `num_heads` | `(B, 12, 8H)` |
| Gate 融合 | `(B, 12, 8H)` | `Linear(16H→8H)` | `(B, 12, 8H)` |
| BiLSTM | `(B, 12, 8H)` | `hidden=h, layers=L` | `(B, 12, 2h)` |
| MHA 汇聚 | `(B, 12, 2h)` | `heads=n` | `(B, 1, 2h)` |
| MLP 输出头 | `(B, 2h)` | `Linear(2h→h→T_out)` | `(B, T_out)` |
| reshape | `(B, T_out)` | - | `(B, T_out, 1)` |

> 注意：当 `attention_embed_dim == d` 时，不需要额外投影；否则会线性映射到 `H`。

---

## 8. 图示版结构图（Mermaid）

```mermaid
flowchart LR
    X[B×12×8 输入] --> MTCN[M‑TCN: 每变量独立 TCN]
    MTCN --> H[B×12×(8d)]
    H --> TA[TemporalAttention]
    H --> VA[VariableAttention]
    TA --> G[Gate 融合]
    VA --> G
    G --> LSTM[BiLSTM]
    LSTM --> MHA[MultiheadAttention 聚合]
    MHA --> MLP[MLP 输出头]
    MLP --> Y[B×4×1 输出]
```

---

## 9. 公式版前向传播流程图

记：
- `B` 批大小
- `T=12` 输入长度
- `V=8` 变量数
- `d = tcn_channels[-1]`
- `H = attention_embed_dim`
- `h = lstm_hidden_size`

```
X ∈ R^{B×T×V}

(1) M‑TCN
H_v = TCN_v(X[:,:,v]) ∈ R^{B×T×d}, v=1..V
H = concat_v(H_v) ∈ R^{B×T×(V·d)}
if d != H: H = Linear(H)

(2) 时空注意力
H_t = TemporalAttn(H) ∈ R^{B×T×(V·H)}
H_v = VariableAttn(H) ∈ R^{B×T×(V·H)}
G = sigmoid(W_g [H_t; H_v])
A = G ⊙ H_t + (1-G) ⊙ H_v  ∈ R^{B×T×(V·H)}

(3) 注意力 LSTM
L = BiLSTM(A) ∈ R^{B×T×(2h)}
q = learnable query
c = MHA(q, L, L) ∈ R^{B×1×(2h)}

(4) 输出
ŷ = MLP(c) -> reshape -> R^{B×T_out×1} (T_out=4)
```

---

## 10. 关键模块伪代码

### 10.1 M‑TCN

```python
# X: (B, T, V)
outputs = []
for v in range(V):
    x_v = X[:, :, v].unsqueeze(-1)      # (B, T, 1)
    h_v = TCN_v(x_v)                    # (B, T, d)
    outputs.append(h_v)
H = concat(outputs, dim=-1)             # (B, T, V*d)
```

### 10.2 SpatioTemporalAttention

```python
# H: (B, T, V*d)
H_t = TemporalAttention(H)              # (B, T, V*d)
H_v = VariableAttention(H)              # (B, T, V*d)
G = sigmoid(W_g([H_t, H_v]))            # (B, T, V*d)
A = G * H_t + (1-G) * H_v               # (B, T, V*d)
```

### 10.3 AttentiveLSTM

```python
# A: (B, T, V*d)
L, _ = BiLSTM(A)                         # (B, T, 2h)
q = learnable_query.expand(B, 1, 2h)
context, _ = MHA(q, L, L)                # (B, 1, 2h)
context = context.squeeze(1)             # (B, 2h)
```

### 10.4 Output Head

```python
# context: (B, 2h)
out = Linear(2h, h) -> ReLU -> Dropout -> Linear(h, T_out)
y_hat = out.view(B, T_out, 1)
```

---

## 11. 损失函数与训练流程（代码一致）

### 11.1 损失函数

代码使用 `HybridLoss`（`src/training/loss.py`）：
- 基础项：MSE
- 正则项：时间一致性约束（鼓励预测序列平滑/稳定）

若你在论文里只想描述经典损失，可注明“实现中使用 MSE + 时序一致性正则”。

### 11.2 训练流程

1. 数据加载与滑窗构造
2. 前向传播得到多步预测
3. 计算 `HybridLoss`
4. 反向传播 + 梯度裁剪（若配置）
5. 学习率调度（WarmupCosine）
6. 验证集早停（EarlyStopping）
7. 保存最佳模型

---

## 12. 超参数表（推荐起步值）

| 模块 | 参数 | 取值（建议） | 说明 |
|---|---|---|---|
| 输入 | `T_in` | 12 | 输入长度（周） |
| 输出 | `T_out` | 4 | 预测步数 |
| 输入 | `V` | 8 | 特征维度 |
| M‑TCN | `tcn_channels` | `[32, 64, 64]` | 最后一层 d=64 |
| M‑TCN | `tcn_kernel_size` | 3 | Causal Conv kernel |
| M‑TCN | `dropout` | 0.2 | TCN dropout |
| 注意力 | `attention_embed_dim` | 128 | 若不等于 d，会线性投影 |
| 注意力 | `attention_num_heads` | 8 | 多头数 |
| 注意力 | `attention_dropout` | 0.1 | 随机注意力丢弃 |
| LSTM | `lstm_hidden_size` | 128 | LSTM hidden |
| LSTM | `lstm_num_layers` | 2 | LSTM 层数 |
| 训练 | `learning_rate` | 1e-3 | AdamW |
| 训练 | `weight_decay` | 1e-4 | 正则 |

---

## 13. 一句话总结

该模型先用 **M‑TCN** 为每个变量独立提取局部时序特征，再用 **时空注意力**融合变量与时间信息，最后通过 **双向注意力 LSTM** 汇聚，输出未来 4 周预测结果。
