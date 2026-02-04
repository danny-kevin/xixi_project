"""Temporal Convolutional Network building blocks."""

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """1D causal convolution that avoids peeking into the future."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.padding,
                dilation=dilation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, time_steps)
        """
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.out_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, channels, time_steps)
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        residual = x if self.downsample is None else self.downsample(x)
        return self.out_activation(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network for a single variable."""

    def __init__(self, input_size: int = 1, num_channels: List[int] | None = None, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 32, 32, 32]

        layers: List[nn.Module] = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                ResidualBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]
        self.receptive_field = self._calculate_receptive_field(num_levels, kernel_size)

    def _calculate_receptive_field(self, num_levels: int, kernel_size: int) -> int:
        # receptive_field = 1 + 2 * (kernel_size - 1) * sum(2**i for i in range(num_levels))
        return 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time_steps, input_size)
        Returns:
            Tensor of shape (batch, time_steps, output_dim)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape (batch, time_steps, channels), got {tuple(x.shape)}")
        out = self.network(x.transpose(1, 2))
        return out.transpose(1, 2)

    def get_receptive_field(self) -> int:
        return self.receptive_field

    def get_output_dim(self) -> int:
        return self.output_dim


# Alias to keep exports aligned with module init
TCNBlock = ResidualBlock


__all__ = ["CausalConv1d", "ResidualBlock", "TCNBlock", "TCN"]
