"""
注意力机制模块
Attention Mechanism Module

实现:
- 缩放点积注意力
- 自注意力机制 (Self-Attention)
- 变量间注意力机制 (Variable Attention)
- 时间注意力机制
- 时空注意力机制
- 随机注意力正则化
"""

from typing import Dict, Optional, Tuple, Union
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.to(dtype=scores.dtype, device=scores.device)
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attention_weights = F.softmax(scores, dim=-1)
        if mask is not None:
            attention_weights = attention_weights * mask
            denom = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = torch.where(
                denom > 0,
                attention_weights / denom,
                torch.zeros_like(attention_weights),
            )
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, value)

        if return_attention:
            return context, attention_weights
        return context, None


class SelfAttention(nn.Module):
    """
    自注意力层

    基于Scaled Dot-Product Attention，
    支持训练时的随机注意力丢弃正则化
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(self.head_dim, attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def _prepare_attention_mask(
        self,
        mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if mask.dim() == 2:
            if mask.size(0) != batch_size or mask.size(1) != seq_len:
                raise ValueError("mask shape must match (batch, seq_len)")
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            if mask.size(0) != batch_size or mask.size(1) != seq_len or mask.size(2) != seq_len:
                raise ValueError("mask shape must match (batch, seq_len, seq_len)")
            mask = mask[:, None, :, :]
        elif mask.dim() == 4:
            if mask.size(0) != batch_size or mask.size(2) != seq_len or mask.size(3) != seq_len:
                raise ValueError("mask shape must match (batch, heads, seq_len, seq_len)")
            if mask.size(1) not in (1, self.num_heads):
                raise ValueError("mask head dimension must be 1 or num_heads")
        else:
            raise ValueError("mask must have 2, 3, or 4 dimensions")
        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = self._prepare_attention_mask(mask, batch_size, seq_len).to(x.device)

        context, attention_weights = self.attention(
            q, k, v, mask=attn_mask, return_attention=return_attention
        )

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        output = self.dropout(output)

        if return_attention:
            return output, attention_weights
        return output


class VariableAttention(nn.Module):
    """
    变量间自注意力模块

    用于动态学习不同输入变量对预测任务的相对重要性，
    使模型能够自适应地调整特征融合策略。
    """

    def __init__(
        self,
        num_variables: int,
        feature_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        stochastic_dropout: float = 0.1,
        *,
        hidden_dim: Optional[int] = None,
        attention_dropout: Optional[float] = None,
    ):
        super().__init__()

        if isinstance(num_heads, float):
            warnings.warn(
                "Passing attention_dropout as the third positional argument is deprecated. "
                "Use attention_dropout=... instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            attention_dropout = num_heads
            num_heads = 4

        if hidden_dim is not None:
            if feature_dim is not None and feature_dim != hidden_dim:
                raise ValueError("feature_dim and hidden_dim must match when both are set")
            warnings.warn(
                "hidden_dim is deprecated; use feature_dim instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            feature_dim = hidden_dim

        if attention_dropout is not None:
            warnings.warn(
                "attention_dropout is deprecated; use dropout instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            dropout = attention_dropout

        if feature_dim is None:
            raise ValueError("feature_dim must be provided")

        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")

        self.num_variables = num_variables
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.d_k = feature_dim // num_heads

        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)
        self.W_o = nn.Linear(feature_dim, feature_dim)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.stochastic_dropout = stochastic_dropout
        self.layer_norm = nn.LayerNorm(feature_dim)

    def _split_heads(self, x: torch.Tensor, batch_size: int, num_vars: int) -> torch.Tensor:
        x = x.view(batch_size, num_vars, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor, batch_size: int, num_vars: int) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, num_vars, self.feature_dim)

    def _stochastic_attention_regularization(self, attention_weights: torch.Tensor) -> torch.Tensor:
        if self.training and self.stochastic_dropout > 0:
            mask = torch.rand_like(attention_weights) > self.stochastic_dropout
            mask = mask.float()

            masked_weights = attention_weights * mask
            sum_weights = masked_weights.sum(dim=-1, keepdim=True) + 1e-9
            attention_weights = masked_weights / sum_weights

        return attention_weights

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]

        if x.dim() == 3 and x.shape[-1] == self.num_variables * self.feature_dim:
            time_steps = x.shape[1]
            x = x.reshape(batch_size * time_steps, self.num_variables, self.feature_dim)
            reshape_back = True
        elif x.dim() == 3 and x.shape[1] == self.num_variables and x.shape[2] == self.feature_dim:
            time_steps = 1
            reshape_back = False
        else:
            raise ValueError(
                "Expected shape (batch, time, num_variables * feature_dim) or "
                "(batch, num_variables, feature_dim)"
            )

        current_batch = x.shape[0]
        residual = x

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = self._split_heads(q, current_batch, self.num_variables)
        k = self._split_heads(k, current_batch, self.num_variables)
        v = self._split_heads(v, current_batch, self.num_variables)

        context, attention_weights = self.attention(q, k, v, return_attention=True)
        if self.training and self.stochastic_dropout > 0:
            attention_weights = self._stochastic_attention_regularization(attention_weights)
            context = torch.matmul(attention_weights, v)

        context = self._merge_heads(context, current_batch, self.num_variables)
        output = self.W_o(context)
        output = self.layer_norm(output + residual)

        if reshape_back:
            output = output.view(batch_size, time_steps, -1)
            attention_weights = attention_weights.view(
                batch_size,
                time_steps,
                self.num_heads,
                self.num_variables,
                self.num_variables,
            ).mean(dim=1)

        if return_attention:
            return output, attention_weights
        return output


class TemporalAttention(nn.Module):
    """
    时间维度注意力模块

    捕捉同一变量在不同时间点的重要性变化，
    强化对关键时间点的关注。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, time_steps, _ = x.shape
        residual = x

        mask = None
        if causal_mask:
            mask = torch.tril(torch.ones(time_steps, time_steps, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)

        q = self.W_q(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, time_steps, self.num_heads, self.d_k).transpose(1, 2)

        context, attention_weights = self.attention(
            q,
            k,
            v,
            mask=mask,
            return_attention=return_attention,
        )
        context = context.transpose(1, 2).contiguous().view(batch_size, time_steps, self.hidden_dim)
        output = self.W_o(context)
        output = self.layer_norm(output + residual)

        if return_attention:
            return output, attention_weights
        return output, None


class SpatioTemporalAttention(nn.Module):
    """
    时空动态注意力模块

    从时间和变量两个维度自适应调整特征重要性:
    - 时间维度: 捕捉同一变量在不同阶段的重要性变化
    - 变量维度: 建模不同变量在同一时间点的交互效应
    """

    def __init__(
        self,
        num_variables: int,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        stochastic_dropout: float = 0.1,
    ):
        super().__init__()

        total_dim = num_variables * feature_dim

        self.temporal_attention = TemporalAttention(
            hidden_dim=total_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.variable_attention = VariableAttention(
            num_variables=num_variables,
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            stochastic_dropout=stochastic_dropout,
        )

        self.gate = nn.Sequential(
            nn.Linear(total_dim * 2, total_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        temporal_out, temporal_attn = self.temporal_attention(x, return_attention=True)
        variable_out, variable_attn = self.variable_attention(x, return_attention=True)

        combined = torch.cat([temporal_out, variable_out], dim=-1)
        gate = self.gate(combined)
        output = gate * temporal_out + (1 - gate) * variable_out

        if return_attention:
            return output, {
                "temporal": temporal_attn,
                "variable": variable_attn,
                "gate": gate,
            }
        return output, None


class AttentionWithRegularization(nn.Module):
    """
    带正则化的注意力层

    结合自注意力与随机注意力丢弃正则化
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        regularization_prob: float = 0.1,
    ):
        super().__init__()
        self.attention = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=regularization_prob,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attention(x, mask=mask, return_attention=False)


class PositionalEncoding(nn.Module):
    """
    位置编码层

    为序列添加位置信息，使模型能够感知时间顺序
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def _extend_pe(self, seq_len: int) -> None:
        current_len = self.pe.size(1)
        if seq_len <= current_len:
            return

        device = self.pe.device
        dtype = self.pe.dtype
        embed_dim = self.pe.size(2)

        position = torch.arange(current_len, seq_len, dtype=dtype, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=dtype, device=device)
            * (-math.log(10000.0) / embed_dim)
        )

        extra_len = seq_len - current_len
        pe_extra = torch.zeros(extra_len, embed_dim, dtype=dtype, device=device)
        pe_extra[:, 0::2] = torch.sin(position * div_term)
        pe_extra[:, 1::2] = torch.cos(position * div_term)

        pe = torch.cat([self.pe.squeeze(0), pe_extra], dim=0).unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self._extend_pe(seq_len)
        pe = self.pe[:, :seq_len].to(device=x.device, dtype=x.dtype)
        x = x + pe
        return self.dropout(x)
