"""Multi-head Temporal Convolutional Network modules."""

from typing import List

import torch
import torch.nn as nn

from .tcn import TCN


class MTCN(nn.Module):
    """
    Multi-head Temporal Convolutional Network (M-TCN).

    Each input variable is processed by an independent TCN sub-network and the
    outputs are concatenated along the feature dimension.
    """

    def __init__(
        self,
        num_variables: int,
        input_size: int,
        tcn_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
        share_weights: bool = False,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_size = input_size
        self.share_weights = share_weights

        if share_weights:
            self.shared_tcn = TCN(
                input_size=input_size,
                num_channels=tcn_channels,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            self.tcn_networks = None
        else:
            self.shared_tcn = None
            self.tcn_networks = nn.ModuleList(
                [
                    TCN(
                        input_size=input_size,
                        num_channels=tcn_channels,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                    for _ in range(num_variables)
                ]
            )

        self.output_dim = num_variables * tcn_channels[-1]
        sample_tcn = self.shared_tcn if self.share_weights else self.tcn_networks[0]
        self.receptive_field = sample_tcn.get_receptive_field()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (batch, seq_len, num_variables, input_size)
               or (batch, seq_len, num_variables) when input_size == 1.
        Returns:
            Tensor with shape (batch, seq_len, num_variables * tcn_channels[-1])
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (batch, seq_len, num_variables, input_size), got {tuple(x.shape)}")

        batch_size, seq_len, num_vars, feature_dim = x.shape
        if num_vars != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {num_vars}")
        if feature_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {feature_dim}")

        outputs = []
        for i in range(self.num_variables):
            var_input = x[:, :, i, :]
            tcn_module = self.shared_tcn if self.share_weights else self.tcn_networks[i]
            outputs.append(tcn_module(var_input))

        return torch.cat(outputs, dim=-1)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_receptive_field(self) -> int:
        return self.receptive_field


class MTCNWithVariableEmbedding(nn.Module):
    """
    M-TCN with variable embedding to capture inter-variable relationships.
    """

    def __init__(
        self,
        num_variables: int,
        input_size: int,
        embedding_dim: int,
        tcn_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_variables = num_variables
        self.input_size = input_size
        self.variable_embedding = nn.Embedding(num_variables, embedding_dim)
        self.mtcn = MTCN(
            num_variables=num_variables,
            input_size=input_size + embedding_dim,
            tcn_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            share_weights=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor with shape (batch, seq_len, num_variables, input_size)
               or (batch, seq_len, num_variables) when input_size == 1.
        Returns:
            Tensor with shape (batch, seq_len, num_variables * tcn_channels[-1])
        """
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        if x.dim() != 4:
            raise ValueError(f"Expected input shape (batch, seq_len, num_variables, input_size), got {tuple(x.shape)}")

        batch_size, seq_len, num_vars, feature_dim = x.shape
        if num_vars != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables, got {num_vars}")
        if feature_dim != self.input_size:
            raise ValueError(f"Expected input_size={self.input_size}, got {feature_dim}")

        device = x.device
        variable_ids = torch.arange(self.num_variables, device=device)
        var_emb = self.variable_embedding(variable_ids)  # (num_variables, embedding_dim)
        var_emb = var_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)

        enhanced_input = torch.cat([x, var_emb], dim=-1)
        return self.mtcn(enhanced_input)

    def get_output_dim(self) -> int:
        return self.mtcn.get_output_dim()


class HierarchicalMTCN(nn.Module):
    """
    Hierarchical M-TCN to capture multi-scale temporal patterns.
    """

    def __init__(self, num_variables: int):
        super().__init__()
        self.short_term = MTCN(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[16, 16],
            kernel_size=3,
            dropout=0.1,
        )
        self.medium_term = MTCN(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[32, 32, 32],
            kernel_size=3,
            dropout=0.15,
        )
        self.long_term = MTCN(
            num_variables=num_variables,
            input_size=1,
            tcn_channels=[32, 32, 32, 32],
            kernel_size=3,
            dropout=0.2,
        )

        total_features = (
            self.short_term.get_output_dim()
            + self.medium_term.get_output_dim()
            + self.long_term.get_output_dim()
        )
        self.fusion = nn.Linear(total_features, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_term(x)
        medium = self.medium_term(x)
        long = self.long_term(x)
        fused = torch.cat([short, medium, long], dim=-1)
        return self.fusion(fused)

    def get_output_dim(self) -> int:
        return self.fusion.out_features

    def get_receptive_field(self) -> int:
        return max(
            self.short_term.get_receptive_field(),
            self.medium_term.get_receptive_field(),
            self.long_term.get_receptive_field(),
        )
