# 接口契约文档

## 📜 什么是接口契约？

**接口契约（Interface Contract）** 是模块之间的"约定"，规定了：
- 输入参数的类型和格式
- 输出结果的类型和格式
- 方法的行为和副作用
- 异常处理规则

就像现实中的合同一样，接口契约确保不同开发者（或Agent）实现的模块能够无缝协作。

---

## 🎓 新手指南：如何使用接口契约

### 步骤1: 理解契约定义

每个接口契约包含：

```python
class DataLoaderProtocol(Protocol):
    """
    数据加载器接口契约
    
    这是一个"协议"，定义了数据加载器必须实现的方法
    """
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据
        
        Returns:
            包含各类数据的字典
            
        契约要求:
        - 必须返回字典类型
        - 字典的值必须是pandas DataFrame
        - 不能返回None
        """
        ...
```

### 步骤2: 实现契约

当你实现一个类时，确保它满足契约要求：

```python
class MyDataLoader:
    """我的数据加载器实现"""
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        # ✅ 正确：返回符合契约的字典
        return {
            'epidemic': pd.DataFrame(...),
            'mobility': pd.DataFrame(...)
        }
        
        # ❌ 错误：返回None违反契约
        # return None
        
        # ❌ 错误：返回列表违反契约
        # return [pd.DataFrame(...)]
```

### 步骤3: 类型检查

使用类型检查工具验证实现是否符合契约：

```bash
# 安装mypy
pip install mypy

# 检查类型
mypy src/data/data_loader.py
```

---

## 📋 核心接口契约定义

### 1. 数据模块接口

#### DataLoaderProtocol
```python
from typing import Protocol, Dict
import pandas as pd

class DataLoaderProtocol(Protocol):
    """数据加载器接口契约"""
    
    def load_epidemic_data(self, filename: str) -> pd.DataFrame:
        """
        加载疫情数据
        
        Args:
            filename: 数据文件名
            
        Returns:
            疫情数据DataFrame
            
        契约要求:
        - 返回的DataFrame必须包含 'date' 列
        - 必须包含至少一列数值数据
        - 不能有重复的日期
        """
        ...
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有类型数据
        
        Returns:
            包含所有数据类型的字典
            
        契约要求:
        - 字典必须包含键: 'epidemic', 'mobility', 'environment', 'intervention'
        - 所有DataFrame必须有相同的日期范围
        """
        ...
    
    def merge_data_sources(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        on: str = 'date'
    ) -> pd.DataFrame:
        """
        合并多源数据
        
        Args:
            data_dict: 数据字典
            on: 合并键列名
            
        Returns:
            合并后的DataFrame
            
        契约要求:
        - 返回的DataFrame行数 = 输入DataFrame的最小行数
        - 必须保留所有输入的列
        - 不能引入NaN值（除非原始数据就有）
        """
        ...
```

#### DataPreprocessorProtocol
```python
class DataPreprocessorProtocol(Protocol):
    """数据预处理器接口契约"""
    
    def normalize(
        self, 
        df: pd.DataFrame, 
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        数据归一化
        
        Args:
            df: 输入DataFrame
            method: 归一化方法
            
        Returns:
            归一化后的DataFrame
            
        契约要求:
        - 输入输出形状必须相同
        - method必须是 'standard' 或 'minmax'
        - 归一化后不能有inf或nan值
        """
        ...
    
    def create_time_windows(
        self, 
        data: np.ndarray,
        window_size: int,
        horizon: int,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列窗口
        
        Args:
            data: 输入数据
            window_size: 窗口大小
            horizon: 预测范围
            stride: 滑动步长
            
        Returns:
            (X, y) 元组
            
        契约要求:
        - X.shape = (N, window_size, num_features)
        - y.shape = (N, horizon)
        - N = (len(data) - window_size - horizon + 1) // stride
        - 不能有数据泄露（未来信息不能出现在X中）
        """
        ...
```

### 2. 模型模块接口

#### TCNProtocol
```python
class TCNProtocol(Protocol):
    """TCN模块接口契约"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape (batch, seq_len, input_size)
            
        Returns:
            输出张量, shape (batch, seq_len, output_channels)
            
        契约要求:
        - 必须保持序列长度不变 (seq_len_out = seq_len_in)
        - 必须是因果卷积（不使用未来信息）
        - 输出不能有inf或nan
        """
        ...
    
    def get_receptive_field(self) -> int:
        """
        获取感受野大小
        
        Returns:
            感受野大小（时间步数）
            
        契约要求:
        - 返回值必须 >= 14 (覆盖14天滞后)
        - 返回值必须 <= window_size
        """
        ...
```

#### AttentionProtocol
```python
class AttentionProtocol(Protocol):
    """注意力机制接口契约"""
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量, shape (batch, seq_len, embed_dim)
            mask: 可选的注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            (output, attention_weights) 元组
            
        契约要求:
        - output.shape = x.shape
        - 如果return_attention=True, attention_weights不能为None
        - attention_weights每行和必须为1 (softmax)
        - 注意力权重必须在[0, 1]范围内
        """
        ...
```

### 3. 训练模块接口

#### TrainerProtocol
```python
class TrainerProtocol(Protocol):
    """训练器接口契约"""
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """
        执行训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练历史字典
            
        契约要求:
        - 返回字典必须包含键: 'train_loss', 'val_loss'
        - 每个列表长度 = 训练的epoch数
        - 损失值必须 >= 0
        - 必须在验证集上评估
        """
        ...
    
    def save_checkpoint(self, path: str, is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            is_best: 是否为最佳模型
            
        契约要求:
        - 必须保存模型状态字典
        - 必须保存优化器状态
        - 必须保存当前epoch
        - 文件必须可以被load_checkpoint加载
        """
        ...
```

### 4. 评估模块接口

#### MetricsProtocol
```python
class MetricsProtocol(Protocol):
    """评估指标接口契约"""
    
    @staticmethod
    def mse(
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        计算MSE
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            MSE值
            
        契约要求:
        - predictions和targets形状必须相同
        - 返回值必须 >= 0
        - 返回值不能是inf或nan
        """
        ...
```

---

## 🔍 契约验证示例

### 示例1: 验证数据加载器

```python
def validate_data_loader(loader: DataLoaderProtocol) -> None:
    """验证数据加载器是否符合契约"""
    
    # 测试1: load_all_data返回正确类型
    data = loader.load_all_data()
    assert isinstance(data, dict), "load_all_data必须返回字典"
    
    # 测试2: 包含所有必需的键
    required_keys = {'epidemic', 'mobility', 'environment', 'intervention'}
    assert required_keys.issubset(data.keys()), f"缺少必需的键: {required_keys - data.keys()}"
    
    # 测试3: 所有值都是DataFrame
    for key, value in data.items():
        assert isinstance(value, pd.DataFrame), f"{key}的值必须是DataFrame"
    
    # 测试4: 所有DataFrame有相同的日期范围
    date_ranges = [df.index for df in data.values()]
    assert all(dr.equals(date_ranges[0]) for dr in date_ranges), \
        "所有DataFrame必须有相同的日期范围"
    
    print("✅ 数据加载器通过契约验证")
```

### 示例2: 验证模型输出

```python
def validate_model_output(
    model: nn.Module,
    batch_size: int = 32,
    window_size: int = 21,
    num_variables: int = 11
) -> None:
    """验证模型输出是否符合契约"""
    
    # 创建测试输入
    x = torch.randn(batch_size, window_size, num_variables)
    
    # 前向传播
    output, attention = model(x, return_attention=True)
    
    # 验证输出形状
    expected_shape = (batch_size, model.prediction_horizon, model.output_size)
    assert output.shape == expected_shape, \
        f"输出形状错误: 期望{expected_shape}, 得到{output.shape}"
    
    # 验证注意力权重
    assert attention is not None, "return_attention=True时必须返回注意力权重"
    
    # 验证无inf/nan
    assert torch.isfinite(output).all(), "输出包含inf或nan"
    
    print("✅ 模型输出通过契约验证")
```

---

## 🛠️ 使用契约的最佳实践

### 1. 在开发前定义契约
```python
# 先定义接口
class MyModuleProtocol(Protocol):
    def process(self, data: np.ndarray) -> np.ndarray:
        ...

# 再实现
class MyModule:
    def process(self, data: np.ndarray) -> np.ndarray:
        # 实现细节
        return processed_data
```

### 2. 编写契约测试
```python
# tests/test_contracts.py
def test_data_loader_contract():
    loader = DataLoader('data')
    validate_data_loader(loader)

def test_model_contract():
    model = AttentionMTCNLSTM(...)
    validate_model_output(model)
```

### 3. 使用类型检查
```python
# 在函数签名中使用Protocol
def train_model(
    loader: DataLoaderProtocol,  # 接受任何符合契约的加载器
    model: nn.Module
) -> None:
    data = loader.load_all_data()
    # ...
```

### 4. 文档化契约
```python
class MyClass:
    """
    我的类
    
    实现的契约:
    - DataLoaderProtocol: 提供数据加载功能
    - PreprocessorProtocol: 提供预处理功能
    
    契约保证:
    - load_all_data()返回的数据已经过验证
    - normalize()不会引入NaN值
    """
    pass
```

---

## ⚠️ 常见契约违反及修复

### 违反1: 返回类型错误
```python
# ❌ 错误
def load_all_data(self) -> Dict[str, pd.DataFrame]:
    return None  # 违反契约！

# ✅ 正确
def load_all_data(self) -> Dict[str, pd.DataFrame]:
    return {'epidemic': pd.DataFrame()}
```

### 违反2: 形状不匹配
```python
# ❌ 错误
def forward(self, x):
    # x.shape = (batch, seq_len, features)
    return x.mean(dim=1)  # 返回 (batch, features)，丢失了seq_len维度

# ✅ 正确
def forward(self, x):
    # 保持序列长度
    return self.process(x)  # 返回 (batch, seq_len, output_dim)
```

### 违反3: 引入NaN
```python
# ❌ 错误
def normalize(self, df):
    return (df - df.mean()) / df.std()  # std=0时会产生NaN

# ✅ 正确
def normalize(self, df):
    std = df.std()
    std = std.replace(0, 1)  # 避免除零
    return (df - df.mean()) / std
```

---

## 📚 进一步学习

1. **Python类型提示**: https://docs.python.org/3/library/typing.html
2. **Protocol使用**: https://peps.python.org/pep-0544/
3. **契约式设计**: https://en.wikipedia.org/wiki/Design_by_contract

---

## 🎯 总结

接口契约的核心作用：
1. **明确期望**: 清楚地定义输入输出
2. **早期发现错误**: 在集成前就能发现不兼容
3. **文档化**: 契约本身就是最好的文档
4. **团队协作**: 不同人可以并行开发，只要遵守契约

记住：**契约是承诺，必须遵守！**
