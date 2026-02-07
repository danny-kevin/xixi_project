

# CCTAL v2 模型架构说明书 (优化版)

**(Causal CI-TCN Variable Attention LSTM)**

## 0. 模型定位与设计哲学

本模型专为**美国州级新冠疫情时空面板数据**设计，基于 `2020-12-06` 至 `2021-10-16` 的黄金数据窗口。

* **核心理念**：
1. **解耦 (Decoupling)**：利用 **CI-TCN** 防止确诊数据的高频噪声（Log变换后）淹没政策数据的低频信号。
2. **重连 (Re-coupling)**：利用 **Variable Attention** 在特征提取后动态加权，特别是捕捉**滞后14天处理后**的政策信号与当前确诊的因果关联。
3. **整合 (Integration)**：利用 **LSTM** 汇总短期（4周）的时序状态，输出稳健的预测。



---

## 1. 任务定义与张量规范

### 1.1 数据输入与输出

* **输入张量 (Input Tensor)**:


* : Batch Size (批大小，建议 32 或 64)
* : **4** (**重要变更**：根据最新数据方案，输入为过去 4 周)
* : **8** (变量数)
1. `Confirmed`: 已做  变换 + MinMax 归一化。
2. `Mobility` (6列): 零售、公园、办公等。
3. `StringencyIndex` (1列): **已做 14 天前向滞后处理**。




* **输出张量 (Output Tensor)**:


* : **4** (预测窗口，未来 4 周)
* *策略*：**Direct Multi-horizon Forecasting**，一次性输出未来 4 个时间步。
* *注*：模型输出的值是在 Log 空间归一化后的数值，计算 RMSE/MAE 时需先反归一化再反 Log。



---

## 2. 总体架构流水线 (Pipeline)

模型由四个核心模块串联而成：

1. **Encoder (CI-TCN)**: 独立通道特征提取器。针对  的短序列进行了轻量化适配。
2. **Mixer (Variable Attention)**: 变量混合器。学习在短窗口内不同变量的重要性。
3. **Decoder (LSTM)**: 时序状态解码器。
4. **Head (Linear)**: 预测头。

---

## 3. 模块详解 (针对 4 周窗口优化)

### 3.1 CI-TCN Encoder (特征解耦)

由于输入只有 4 个时间步，TCN 不需要过深，否则会导致 Padding 噪音过多。

* **输入变换**: 将输入维度调整为  以适配 1D 卷积。
* **核心实现 (Grouped Convolution)**:
* `nn.Conv1d(in_channels=V, out_channels=V*d, groups=V)`。
* *解释*：确诊、流动性、政策这 8 个变量拥有独立的卷积核。


* **结构参数调整 (适配 )**:
* **Kernel Size ()**: 推荐设为 **2**。
* **Dilation Sequence ()**: 推荐设为 **(1, 2)**。
* *感受野计算*：
* 第一层 (d=1, k=2): 看 2 个时间步。
* 第二层 (d=2, k=2): 感受野增加 2，总感受野为 。
* **完美覆盖**：刚好覆盖输入的 4 周，无需过多的 Zero Padding。





### 3.2 Variable Attention Mixer (动态重连)

* **Step-specific Attention**:
鉴于预测目标是未来 4 周，且政策效果（已滞后14天）可能在未来第 3-4 周体现得更明显，我们继续为每个预测步  学习独立的权重。
* **计算逻辑**:


* 这里  将直接反映：在预测未来第  周时，历史第  周的第  个变量有多重要。



### 3.3 LSTM Decoder (时序整合)

* **架构**: 单层、单向 LSTM。
* **输入**: 融合后的特征序列  (长度为 4)。
* **操作**:


* **输出选择**: 提取 **Last Hidden State ()** 作为上下文向量。

### 3.4 Linear Head (预测输出)

* **公式**: 
* **维度**: `nn.Linear(hidden_size, 4)`。

---

## 4. 可解释性与分析支持

### 4.1 变量权重热力图 (Heatmaps)

* **X轴**: 历史时间步 (t-3, t-2, t-1, t)。
* **Y轴**: 8 个变量。
* **关键看点**:
* 你的政策数据已经滞后了 14 天（2周）。
* 如果在热力图中，政策变量在  (当前周) 或  (上周) 的权重很高，说明**实际发生在一个月前的政策**对现在的预测有重大影响。
* **论文叙事点**：结合你的数据处理，这证明了政策的**长期滞后效应**被模型成功捕捉。



---

## 5. 训练配置与防泄漏 (基于你的数据方案)

* **归一化策略 (Data Leakage Prevention)**:
* 模型加载的 `scaler_data_min.npy` 和 `scaler_data_max.npy` 必须**仅由 Training Weeks 计算得出**。
* Validation 和 Test 数据在输入模型前，使用这两个文件参数进行 Transform。


* **损失函数**: `MSELoss` (在 Log-Scaled 空间计算)。
* **评估指标**:
* 在计算 MAE/RMSE 时，必须执行 **Inverse Transform**：
1. `Inverse_MinMax(pred)`
2. `exp(pred) - 1` (反 Log 变换)


* 这样汇报的是真实的“确诊人数”误差。



---

## 6. 消融实验设计 (保持不变)

| 模型代号 | 架构描述 | 验证目的 |
| --- | --- | --- |
| **M1: Base (LSTM)** | 原始变量(8维)  LSTM | Baseline。 |
| **M2: CI-Only** | CI-TCN(dilations=[1,2])  LSTM | 验证 CI 结构对 Log 变换后数据的特征提取能力。 |
| **M3: CCTAL (Full)** | CI-TCN  Attention  LSTM | 验证动态权重带来的提升。 |
| **M4: No-Policy** | 输入 7 维 (去掉 Stringency) | **关键实验**：证明加上处理过的政策数据，误差确实降低了。 |

