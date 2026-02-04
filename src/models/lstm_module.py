"""
LSTM模块
LSTM Module

实现双层双向LSTM与门控跳跃连接，并支持时间注意力聚合。
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class GatedSkipConnection(nn.Module):
    """门控跳跃连接 / Gated skip connection for fusing LSTM outputs with projected inputs."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, current: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current, skip], dim=-1)
        gate = self.gate(combined)
        return gate * current + (1 - gate) * skip


class BiLSTMModule(nn.Module):
    """
    双层双向LSTM模块

    用于捕获长期时间依赖关系，
    并通过门控跳跃连接融合输入投影。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        projected_dim = hidden_size * self.num_directions
        self.input_proj = nn.Linear(input_size, projected_dim)
        self.skip_gate = GatedSkipConnection(projected_dim)
        self.layer_norm = nn.LayerNorm(projected_dim)
        self.output_size = projected_dim

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_final_only: bool = False,
    ) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量, shape: (batch, seq_len, input_size)
            hidden: 可选的初始隐藏状态 (h_0, c_0)
            return_final_only: 是否仅返回最终隐藏状态(兼容开关)

        Returns:
            - 默认: (output, (h_n, c_n))
            - 若return_final_only=True: (output, final)
        """
        skip = self.input_proj(x)
        output, (h_n, c_n) = self.lstm(x, hidden)
        output = self.skip_gate(output, skip)
        output = self.layer_norm(output)

        if not return_final_only:
            return output, (h_n, c_n)

        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final = h_n[-1]

        return output, final

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化隐藏状态

        Args:
            batch_size: 批次大小
            device: 计算设备

        Returns:
            (h_0, c_0) 初始隐藏状态
        """
        num_layers = self.num_layers * self.num_directions
        h_0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0

    def get_output_dim(self) -> int:
        """获取输出特征维度"""
        return self.hidden_size * self.num_directions


class LSTMWithAttention(nn.Module):
    """
    带注意力的LSTM模块

    在LSTM输出后应用时间注意力机制，用于聚合序列信息。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        attention_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.lstm = BiLSTMModule(input_size, hidden_size, num_layers, dropout)
        if self.lstm.output_size % num_heads != 0:
            raise ValueError("num_heads must divide lstm output size")
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm.output_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, self.lstm.output_size))
        self.output_size = self.lstm.output_size
        self.attention_dim = attention_dim  # compatibility placeholder

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入张量
            key_padding_mask: optional padding mask, shape: (batch, seq_len), True indicates padding
            return_attention: 是否返回注意力权重

        Returns:
            - 默认: context向量
            - 若return_attention=True: (context向量, 注意力权重)
        """
        batch_size = x.shape[0]
        lstm_out, _ = self.lstm(x)

        query = self.query.expand(batch_size, -1, -1)
        if key_padding_mask is not None:
            if (
                key_padding_mask.dim() != 2
                or key_padding_mask.size(0) != batch_size
                or key_padding_mask.size(1) != lstm_out.size(1)
            ):
                raise ValueError("key_padding_mask must have shape (batch, seq_len)")
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)

        attended, attn_weights = self.attention(
            query,
            lstm_out,
            lstm_out,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
        )
        output = attended.squeeze(1)

        if return_attention:
            return output, attn_weights.squeeze(1)
        return output


class AttentiveLSTM(LSTMWithAttention):
    """Alias for the attention-augmented LSTM module."""
