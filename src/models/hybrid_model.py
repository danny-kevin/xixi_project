"""
混合模型
Hybrid Model Module

整合M-TCN、注意力机制和LSTM的完整模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .mtcn import MTCN
from .attention import SpatioTemporalAttention
from .lstm_module import AttentiveLSTM


class AttentionMTCNLSTM(nn.Module):
    """
    注意力增强M-TCN-LSTM混合神经网络
    
    模型架构:
    1. 输入层: 多变量时间序列窗口
    2. M-TCN模块: 为每个变量提取局部时间特征
    3. 特征拼接: 合并所有变量的TCN输出
    4. 注意力增强层: 自注意力 + 随机Dropout正则化
    5. 双层双向LSTM: 捕获长期序列依赖
    6. 全连接输出层: 生成预测结果
    
    创新点:
    - 混合架构设计: 融合TCN的局部特征提取与LSTM的序列建模
    - 注意力机制增强: 变量间注意力 + 随机注意力正则化
    - 传染病特性优化: 感受野覆盖14-21天滞后周期
    """
    
    def __init__(
        self,
        num_variables: int,
        input_size: int,
        tcn_channels: List[int],
        tcn_kernel_size: int = 3,
        attention_embed_dim: int = 128,
        attention_num_heads: int = 8,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        output_size: int = 1,
        prediction_horizon: int = 7,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        use_state_embedding: bool = False,
        num_states: int = 0,
        state_embed_dim: int = 16
    ):
        """
        初始化混合模型
        
        Args:
            num_variables: 输入变量数量
            input_size: 每个变量的特征维度
            tcn_channels: TCN各层输出通道数
            tcn_kernel_size: TCN卷积核大小
            attention_embed_dim: 注意力嵌入维度
            attention_num_heads: 注意力头数量
            lstm_hidden_size: LSTM隐藏层维度
            lstm_num_layers: LSTM层数
            output_size: 输出维度 (预测的变量数)
            prediction_horizon: 预测时间范围 (7天或14天)
            dropout: Dropout概率
            attention_dropout: 注意力权重Dropout概率
        """
        super().__init__()
        self.num_variables = num_variables
        self.prediction_horizon = prediction_horizon
        self.output_size = output_size

        self.mtcn = MTCN(
            num_variables=num_variables,
            input_size=input_size,
            tcn_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
        )

        self.mtcn_feature_dim = tcn_channels[-1]
        self.attention_feature_dim = attention_embed_dim or self.mtcn_feature_dim

        if self.attention_feature_dim != self.mtcn_feature_dim:
            self.feature_projection = nn.Linear(self.mtcn_feature_dim, self.attention_feature_dim)
        else:
            self.feature_projection = None

        self.attention = SpatioTemporalAttention(
            num_variables=num_variables,
            feature_dim=self.attention_feature_dim,
            num_heads=attention_num_heads,
            dropout=dropout,
            stochastic_dropout=attention_dropout,
        )

        self.use_state_embedding = bool(use_state_embedding and num_states > 0)
        if self.use_state_embedding:
            self.state_embedding = nn.Embedding(num_states, state_embed_dim)
            self.state_proj = nn.Linear(state_embed_dim, self.attention_feature_dim)
        else:
            self.state_embedding = None
            self.state_proj = None

        self.lstm = AttentiveLSTM(
            input_size=num_variables * self.attention_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )

        self.fc_out = nn.Sequential(
            nn.Linear(self.lstm.output_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, output_size * prediction_horizon),
        )

        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        state_ids: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x: 输入张量, shape: (batch, seq_len, num_variables)
            return_attention: 是否返回注意力权重 (用于可解释性)
            
        Returns:
            (predictions, attention_dict) 元组
            - predictions: shape (batch, prediction_horizon, output_size)
            - attention_dict: 包含注意力权重的字典 (可选)
        """
        tcn_out = self.mtcn(x)

        batch_size, seq_len, _ = tcn_out.shape
        tcn_out = tcn_out.view(batch_size, seq_len, self.num_variables, self.mtcn_feature_dim)
        if self.feature_projection is not None:
            tcn_out = self.feature_projection(tcn_out)

        if self.use_state_embedding and state_ids is not None:
            state_emb = self.state_embedding(state_ids)
            state_emb = self.state_proj(state_emb).view(batch_size, 1, 1, self.attention_feature_dim)
            tcn_out = tcn_out + state_emb
        tcn_out = tcn_out.reshape(batch_size, seq_len, -1)

        attn_out, attn_weights = self.attention(tcn_out, return_attention=True)
        lstm_out, time_attn = self.lstm(attn_out, return_attention=True)

        predictions = self.fc_out(lstm_out)
        predictions = predictions.view(batch_size, self.prediction_horizon, self.output_size)

        if return_attention:
            attention_dict = dict(attn_weights)
            attention_dict["lstm_temporal"] = time_attn
            return predictions, attention_dict
        return predictions
    
    def get_attention_weights(self, x: torch.Tensor, state_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        获取注意力权重 (用于模型解释)
        
        Args:
            x: 输入张量
            
        Returns:
            包含各层注意力权重的字典
        """
        self.eval()
        with torch.no_grad():
            _, attention = self.forward(x, state_ids=state_ids, return_attention=True)
        return attention
    
    def predict(
        self, 
        x: torch.Tensor,
        forecast_steps: int = 7,
        state_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多步预测
        
        Args:
            x: 输入张量
            forecast_steps: 预测步数
            
        Returns:
            预测结果张量
        """
        predictions = self.forward(x, state_ids=state_ids, return_attention=False)
        if forecast_steps == self.prediction_horizon:
            return predictions
        if forecast_steps < self.prediction_horizon:
            return predictions[:, :forecast_steps, :]

        repeat_steps = forecast_steps - self.prediction_horizon
        tail = predictions[:, -1:, :].repeat(1, repeat_steps, 1)
        return torch.cat([predictions, tail], dim=1)


class ModelFactory:
    """
    模型工厂类
    
    根据配置创建不同规模的模型
    """
    
    @staticmethod
    def create_model(config: Dict) -> AttentionMTCNLSTM:
        """
        根据配置创建模型
        
        Args:
            config: 模型配置字典
            
        Returns:
            初始化的模型实例
        """
        if hasattr(config, "model"):
            model_config = config.model
            attention_config = getattr(config, "attention", None)
        else:
            model_config = config.get("model", config)
            attention_config = config.get("attention")

        def _get(cfg, key, default):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        tcn_channels = _get(model_config, "tcn_channels", [32, 64, 64])
        attention_num_heads = _get(
            attention_config,
            "num_heads",
            _get(model_config, "attention_num_heads", 4),
        )
        attention_dropout = _get(
            attention_config,
            "stochastic_dropout",
            _get(model_config, "attention_dropout", 0.1),
        )

        return AttentionMTCNLSTM(
            num_variables=_get(model_config, "num_variables", 1),
            input_size=_get(model_config, "input_size", 1),
            tcn_channels=tcn_channels,
            tcn_kernel_size=_get(model_config, "tcn_kernel_size", 3),
            attention_embed_dim=_get(model_config, "attention_embed_dim", tcn_channels[-1]),
            attention_num_heads=attention_num_heads,
            lstm_hidden_size=_get(model_config, "lstm_hidden_size", 128),
            lstm_num_layers=_get(model_config, "lstm_num_layers", 2),
            output_size=_get(model_config, "output_size", 1),
            prediction_horizon=_get(model_config, "prediction_horizon", 7),
            dropout=_get(model_config, "dropout", 0.2),
            attention_dropout=attention_dropout,
            use_state_embedding=_get(model_config, "use_state_embedding", False),
            num_states=_get(model_config, "num_states", 0),
            state_embed_dim=_get(model_config, "state_embed_dim", 16),
        )
    
    @staticmethod
    def create_small_model(num_variables: int, prediction_horizon: int = 7) -> AttentionMTCNLSTM:
        """创建小型模型 (用于快速实验)"""
        return AttentionMTCNLSTM(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[16, 16],
            tcn_kernel_size=3,
            attention_embed_dim=64,
            attention_num_heads=4,
            lstm_hidden_size=64,
            lstm_num_layers=1,
            output_size=1,
            prediction_horizon=prediction_horizon,
            dropout=0.1,
            attention_dropout=0.1,
        )
    
    @staticmethod
    def create_base_model(num_variables: int, prediction_horizon: int = 7) -> AttentionMTCNLSTM:
        """创建基础模型"""
        return AttentionMTCNLSTM(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[32, 64, 64],
            tcn_kernel_size=3,
            attention_embed_dim=128,
            attention_num_heads=8,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            output_size=1,
            prediction_horizon=prediction_horizon,
            dropout=0.2,
            attention_dropout=0.1,
        )
    
    @staticmethod
    def create_large_model(num_variables: int, prediction_horizon: int = 14) -> AttentionMTCNLSTM:
        """创建大型模型 (用于最终训练)"""
        return AttentionMTCNLSTM(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[64, 64, 64, 64],
            tcn_kernel_size=3,
            attention_embed_dim=256,
            attention_num_heads=8,
            lstm_hidden_size=256,
            lstm_num_layers=3,
            output_size=1,
            prediction_horizon=prediction_horizon,
            dropout=0.3,
            attention_dropout=0.1,
        )
