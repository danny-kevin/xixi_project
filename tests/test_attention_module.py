import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.attention import (
    AttentionWithRegularization,
    ScaledDotProductAttention,
    SelfAttention,
    SpatioTemporalAttention,
    TemporalAttention,
    VariableAttention,
    PositionalEncoding,
)
from src.models.lstm_module import LSTMWithAttention
from src.utils import load_config
from src.utils.attention_visualization import AttentionVisualizer


def test_scaled_dot_product_attention_weights_sum():
    attention = ScaledDotProductAttention(d_model=4, dropout=0.0)
    q = torch.randn(2, 3, 4)
    k = torch.randn(2, 3, 4)
    v = torch.randn(2, 3, 4)
    _, weights = attention(q, k, v, return_attention=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)


def test_scaled_dot_product_attention_all_masked():
    attention = ScaledDotProductAttention(d_model=4, dropout=0.0)
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 2, 4)
    v = torch.randn(1, 2, 4)
    mask = torch.zeros(1, 1, 2, 2)
    _, weights = attention(q, k, v, mask=mask, return_attention=True)
    assert not torch.isnan(weights).any()
    assert torch.allclose(weights.sum(dim=-1), torch.zeros_like(weights.sum(dim=-1)))


def test_self_attention_returns_tensor_by_default():
    attention = SelfAttention(embed_dim=16, num_heads=4, dropout=0.0, attention_dropout=0.0)
    x = torch.randn(2, 5, 16)
    output = attention(x)
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape


def test_self_attention_returns_weights():
    attention = SelfAttention(embed_dim=16, num_heads=4, dropout=0.0, attention_dropout=0.0)
    x = torch.randn(2, 5, 16)
    output, weights = attention(x, return_attention=True)
    assert output.shape == x.shape
    assert weights.shape == (2, 4, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)


def test_self_attention_mask_applied():
    attention = SelfAttention(embed_dim=8, num_heads=2, dropout=0.0, attention_dropout=0.0)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int)
    _, weights = attention(x, mask=mask, return_attention=True)
    masked_weights = weights[..., 2:]
    assert masked_weights.max().item() < 1e-5


def test_variable_attention_shapes():
    attention = VariableAttention(num_variables=3, feature_dim=12, num_heads=3, dropout=0.0)
    x = torch.randn(4, 3, 12)
    output, weights = attention(x, return_attention=True)
    assert output.shape == x.shape
    assert weights.shape == (4, 3, 3, 3)


def test_variable_attention_time_series_shape():
    attention = VariableAttention(num_variables=3, feature_dim=6, num_heads=3, dropout=0.0)
    x = torch.randn(2, 5, 18)
    output, weights = attention(x, return_attention=True)
    assert output.shape == x.shape
    assert weights.shape == (2, 3, 3, 3)


def test_variable_attention_noncontiguous_input():
    attention = VariableAttention(num_variables=3, feature_dim=4, num_heads=2, dropout=0.0)
    x = torch.randn(2, 12, 5).transpose(1, 2)
    assert not x.is_contiguous()
    output, weights = attention(x, return_attention=True)
    assert output.shape == x.shape
    assert weights.shape == (2, 2, 3, 3)


def test_variable_attention_backward_compat():
    with pytest.warns(DeprecationWarning):
        attention = VariableAttention(3, 12, 0.2)
    x = torch.randn(2, 3, 12)
    output = attention(x)
    assert output.shape == x.shape


def test_temporal_attention_causal_mask():
    attention = TemporalAttention(hidden_dim=12, num_heads=3, dropout=0.0)
    x = torch.randn(1, 4, 12)
    _, weights = attention(x, causal_mask=True, return_attention=True)
    upper_triangle = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    masked_values = weights[0, 0][upper_triangle]
    assert masked_values.max().item() < 1e-5


def test_spatiotemporal_attention_output():
    attention = SpatioTemporalAttention(num_variables=2, feature_dim=4, num_heads=2, dropout=0.0)
    x = torch.randn(3, 6, 8)
    output, attn_dict = attention(x, return_attention=True)
    assert output.shape == x.shape
    assert set(attn_dict.keys()) == {"temporal", "variable", "gate"}
    assert attn_dict["gate"].shape == x.shape


def test_positional_encoding_adds_signal():
    encoding = PositionalEncoding(embed_dim=6, max_len=10, dropout=0.0)
    x = torch.zeros(2, 4, 6)
    output = encoding(x)
    assert output.shape == x.shape
    assert not torch.allclose(output, x)


def test_positional_encoding_extends_max_len():
    encoding = PositionalEncoding(embed_dim=6, max_len=4, dropout=0.0)
    x = torch.zeros(2, 6, 6)
    output = encoding(x)
    assert output.shape == x.shape
    assert encoding.pe.size(1) >= 6


def test_attention_with_regularization_output():
    attention = AttentionWithRegularization(embed_dim=8, num_heads=2, dropout=0.0, regularization_prob=0.0)
    x = torch.randn(2, 3, 8)
    output = attention(x)
    assert output.shape == x.shape


def test_lstm_with_attention_padding_mask():
    model = LSTMWithAttention(input_size=4, hidden_size=8, num_layers=1, dropout=0.0, num_heads=2)
    x = torch.randn(2, 5, 4)
    padding_mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )
    output, weights = model(x, key_padding_mask=padding_mask, return_attention=True)
    assert output.shape == (2, model.output_size)
    assert weights.shape == (2, 5)
    assert weights[0, 2:].max().item() < 1e-5
    assert weights[1, 3:].max().item() < 1e-5


def test_attention_visualizer_runs(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    visualizer = AttentionVisualizer(variable_names=["a", "b", "c"])
    weights = np.random.rand(3, 3)
    visualizer.plot_variable_attention(weights)
    visualizer.plot_temporal_attention(np.random.rand(4, 4))


def test_attention_config_loading():
    config = load_config("configs/attention_config.yaml")
    assert config.attention.num_heads == 4
    assert config.attention.type == "spatiotemporal"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
