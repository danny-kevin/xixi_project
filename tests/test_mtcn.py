import torch
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.tcn import TCN
from src.models.mtcn import HierarchicalMTCN, MTCN, MTCNWithVariableEmbedding


def test_tcn_forward_shape():
    batch, seq_len, input_size = 2, 16, 3
    model = TCN(input_size=input_size, num_channels=[8, 8], kernel_size=3, dropout=0.0)
    x = torch.randn(batch, seq_len, input_size)
    out = model(x)
    assert out.shape == (batch, seq_len, model.get_output_dim())


def test_mtcn_concat_shape():
    batch, seq_len, num_vars = 2, 12, 3
    model = MTCN(num_variables=num_vars, input_size=1, tcn_channels=[4, 4], kernel_size=3, dropout=0.0)
    x = torch.randn(batch, seq_len, num_vars)
    out = model(x)
    assert out.shape == (batch, seq_len, num_vars * 4)
    assert model.get_receptive_field() > 0


def test_mtcn_share_weights_with_features():
    batch, seq_len, num_vars, feat_dim = 2, 10, 4, 2
    model = MTCN(
        num_variables=num_vars,
        input_size=feat_dim,
        tcn_channels=[6, 6],
        kernel_size=3,
        dropout=0.0,
        share_weights=True,
    )
    x = torch.randn(batch, seq_len, num_vars, feat_dim)
    out = model(x)
    assert out.shape == (batch, seq_len, num_vars * 6)


def test_mtcn_with_variable_embedding():
    batch, seq_len, num_vars = 2, 14, 5
    model = MTCNWithVariableEmbedding(
        num_variables=num_vars,
        input_size=1,
        embedding_dim=3,
        tcn_channels=[5, 5],
        kernel_size=3,
        dropout=0.0,
    )
    x = torch.randn(batch, seq_len, num_vars)
    out = model(x)
    assert out.shape == (batch, seq_len, num_vars * 5)


def test_hierarchical_mtcn_shape():
    batch, seq_len, num_vars = 2, 18, 3
    model = HierarchicalMTCN(num_variables=num_vars)
    x = torch.randn(batch, seq_len, num_vars)
    out = model(x)
    assert out.shape == (batch, seq_len, model.get_output_dim())


def _run_all_tests():
    test_tcn_forward_shape()
    test_mtcn_concat_shape()
    test_mtcn_share_weights_with_features()
    test_mtcn_with_variable_embedding()
    test_hierarchical_mtcn_shape()


if __name__ == "__main__":
    _run_all_tests()
    print("All tests passed.")
