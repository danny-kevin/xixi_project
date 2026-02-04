# 端到端数据流文档

## 📊 完整数据流图

```
原始数据 (CSV/Excel)
    ↓
DataLoader.load_all_data()
    ↓ shape: Dict[str, pd.DataFrame]
DataPreprocessor.preprocess()
    ↓ shape: pd.DataFrame (samples, features)
DataPreprocessor.create_time_windows()
    ↓ shape: (X: (N, window_size, num_variables), y: (N, horizon))
EpidemicDataset
    ↓ shape: (batch, window_size, num_variables), (batch, horizon)
M-TCN模块
    ↓ shape: (batch, window_size, num_variables * tcn_channels[-1])
特征投影层
    ↓ shape: (batch, window_size, attention_embed_dim)
注意力层
    ↓ shape: (batch, window_size, attention_embed_dim)
BiLSTM模块
    ↓ shape: (batch, window_size, lstm_hidden_size * 2)
全连接输出层
    ↓ shape: (batch, prediction_horizon, output_size)
预测结果
```

---

## 🔍 各模块详细张量形状约束

### 1. 数据加载阶段

#### DataLoader.load_all_data()
```python
输入: 无
输出: Dict[str, pd.DataFrame]
  - 'epidemic': shape (T, 3)  # [new_cases, new_deaths, new_recovered]
  - 'mobility': shape (T, 2)  # [mobility_index, transport_flow]
  - 'environment': shape (T, 3)  # [temperature, humidity, uv_index]
  - 'intervention': shape (T, 3)  # [lockdown_level, social_distance, vaccination_rate]
```

#### DataLoader.merge_data_sources()
```python
输入: Dict[str, pd.DataFrame]
输出: pd.DataFrame
  shape: (T, 11)  # 所有特征合并
  columns: ['new_cases', 'new_deaths', ..., 'vaccination_rate']
```

### 2. 预处理阶段

#### DataPreprocessor.normalize()
```python
输入: pd.DataFrame, shape (T, num_features)
输出: pd.DataFrame, shape (T, num_features)
约束: 
  - 所有值归一化到 [0, 1] (minmax) 或标准化 (standard)
  - 无NaN值
```

#### DataPreprocessor.create_time_windows()
```python
输入: 
  - data: np.ndarray, shape (T, num_features)
  - window_size: int = 21
  - horizon: int = 7
  - stride: int = 1

输出: Tuple[np.ndarray, np.ndarray]
  - X: shape (N, window_size, num_features)
    其中 N = (T - window_size - horizon + 1) // stride
  - y: shape (N, horizon)

约束:
  - N >= 1 (至少有一个样本)
  - window_size >= 7 (至少一周历史)
  - horizon >= 1 (至少预测一天)
```

### 3. Dataset阶段

#### EpidemicDataset.__getitem__()
```python
输入: idx (int)
输出: Tuple[torch.Tensor, torch.Tensor]
  - input: shape (window_size, num_features)
    dtype: torch.float32
  - target: shape (horizon,) 或 (horizon, output_size)
    dtype: torch.float32

约束:
  - 0 <= idx < len(dataset)
  - 所有值为有限数 (无inf, nan)
```

#### DataLoader (PyTorch)
```python
输入: EpidemicDataset
输出: Tuple[torch.Tensor, torch.Tensor]
  - batch_input: shape (batch_size, window_size, num_features)
  - batch_target: shape (batch_size, horizon)

约束:
  - batch_size >= 1
  - 最后一个batch可能小于batch_size
```

### 4. 模型前向传播

#### M-TCN模块
```python
输入: x, shape (batch, window_size, num_variables)
输出: shape (batch, window_size, num_variables * tcn_channels[-1])

约束:
  - num_variables = 输入特征数
  - tcn_channels[-1] = 最后一层TCN通道数
  - window_size保持不变 (因果卷积)
```

#### 特征投影层
```python
输入: x, shape (batch, window_size, num_variables * tcn_channels[-1])
输出: shape (batch, window_size, attention_embed_dim)

约束:
  - attention_embed_dim % attention_num_heads == 0
```

#### 注意力层
```python
输入: x, shape (batch, seq_len, embed_dim)
输出: Tuple[torch.Tensor, Optional[torch.Tensor]]
  - output: shape (batch, seq_len, embed_dim)
  - attention_weights: shape (batch, num_heads, seq_len, seq_len) 或 None

约束:
  - embed_dim % num_heads == 0
  - attention_weights 每行和为1 (softmax后)
```

#### BiLSTM模块
```python
输入: x, shape (batch, seq_len, input_size)
输出: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
  - output: shape (batch, seq_len, hidden_size * 2)
  - (h_n, c_n): 
    - h_n: shape (num_layers * 2, batch, hidden_size)
    - c_n: shape (num_layers * 2, batch, hidden_size)

约束:
  - 双向: num_directions = 2
  - 输出维度 = hidden_size * num_directions
```

#### 全连接输出层
```python
输入: x, shape (batch, seq_len, lstm_hidden_size * 2)
处理: 取最后一个时间步或池化
输出: shape (batch, prediction_horizon, output_size)

约束:
  - prediction_horizon = 配置的预测天数
  - output_size = 预测的变量数 (通常为1)
```

### 5. 损失计算

#### 损失函数输入
```python
predictions: shape (batch, horizon, output_size)
targets: shape (batch, horizon, output_size) 或 (batch, horizon)

约束:
  - predictions 和 targets 形状必须兼容
  - 所有值为有限数
```

---

## ✅ 形状验证检查点

在以下位置应进行形状验证：

### 1. 数据加载后
```python
assert merged_data.shape[1] == num_expected_features, \
    f"Expected {num_expected_features} features, got {merged_data.shape[1]}"
```

### 2. 创建窗口后
```python
assert X.shape == (num_samples, window_size, num_features), \
    f"X shape mismatch: expected {(num_samples, window_size, num_features)}, got {X.shape}"
assert y.shape == (num_samples, horizon), \
    f"y shape mismatch: expected {(num_samples, horizon)}, got {y.shape}"
```

### 3. Dataset返回后
```python
input_tensor, target_tensor = dataset[0]
assert input_tensor.shape == (window_size, num_features), \
    f"Input shape mismatch: {input_tensor.shape}"
assert target_tensor.shape == (horizon,), \
    f"Target shape mismatch: {target_tensor.shape}"
```

### 4. 模型前向传播
```python
# 在每个模块的forward方法中
def forward(self, x):
    expected_shape = (batch_size, seq_len, feature_dim)
    assert x.shape == expected_shape, \
        f"Input shape mismatch: expected {expected_shape}, got {x.shape}"
    
    # ... 处理 ...
    
    assert output.shape == expected_output_shape, \
        f"Output shape mismatch: expected {expected_output_shape}, got {output.shape}"
    return output
```

---

## 🔧 调试技巧

### 1. 打印中间形状
```python
# 在模型中添加调试输出
def forward(self, x):
    print(f"Input shape: {x.shape}")
    
    x = self.mtcn(x)
    print(f"After M-TCN: {x.shape}")
    
    x = self.attention(x)
    print(f"After Attention: {x.shape}")
    
    # ... 继续
```

### 2. 使用断点调试
```python
import pdb

def forward(self, x):
    x = self.mtcn(x)
    pdb.set_trace()  # 在此处暂停，检查x的形状和值
    x = self.attention(x)
```

### 3. 单元测试
```python
def test_model_output_shape():
    model = AttentionMTCNLSTM(...)
    x = torch.randn(batch_size, window_size, num_variables)
    
    output, _ = model(x)
    
    expected_shape = (batch_size, prediction_horizon, output_size)
    assert output.shape == expected_shape, \
        f"Expected {expected_shape}, got {output.shape}"
```

---

## 📝 常见形状错误及解决方案

### 错误1: 维度不匹配
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x256 and 512x128)
```
**原因**: 全连接层输入维度与定义不符  
**解决**: 检查上一层输出维度，确保与fc层输入维度一致

### 错误2: Batch维度丢失
```
RuntimeError: Expected 3D tensor, got 2D
```
**原因**: 某处操作移除了batch维度  
**解决**: 使用`unsqueeze(0)`添加batch维度，或使用`keepdim=True`

### 错误3: 序列长度变化
```
AssertionError: Expected seq_len=21, got 20
```
**原因**: 卷积操作改变了序列长度  
**解决**: 使用因果卷积或适当的padding保持序列长度

---

## 🎯 最佳实践

1. **始终在模块开头验证输入形状**
2. **在模块结尾验证输出形状**
3. **使用类型注解明确张量形状**
4. **编写形状测试用例**
5. **在文档中明确标注每个函数的输入输出形状**
