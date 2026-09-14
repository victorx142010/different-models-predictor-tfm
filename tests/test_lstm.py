"""Pruebas de la rama cuantitativa (lstm.py): formas de salida, h_L igual al
último paso de la secuencia e independencia entre ejemplos del mismo lote.
"""

import torch

from src.quant_module.lstm import QuantLSTMEncoder


def test_output_shapes_single_layer() -> None:
    enc = QuantLSTMEncoder(input_dim=3, hidden_size=16, num_layers=1)
    x = torch.randn(4, 20, 3)
    h_L, seq = enc(x)
    assert h_L.shape == (4, 16)
    assert seq.shape == (4, 20, 16)


def test_output_shapes_two_layers_with_dropout() -> None:
    enc = QuantLSTMEncoder(input_dim=3, hidden_size=32, num_layers=2, dropout=0.3)
    x = torch.randn(5, 10, 3)
    h_L, seq = enc(x)
    assert h_L.shape == (5, 32)
    assert seq.shape == (5, 10, 32)


def test_h_L_matches_last_timestep_of_sequence() -> None:
    enc = QuantLSTMEncoder(input_dim=3, hidden_size=8, num_layers=1)
    enc.eval()
    x = torch.randn(2, 7, 3)
    h_L, seq = enc(x)
    assert torch.allclose(h_L, seq[:, -1, :])


def test_different_batches_are_independent() -> None:
    enc = QuantLSTMEncoder(input_dim=3, hidden_size=8, num_layers=1)
    enc.eval()
    x1 = torch.randn(1, 5, 3)
    x2 = torch.randn(1, 5, 3)
    h1, _ = enc(x1)
    h2, _ = enc(x2)
    batched = torch.cat([x1, x2], dim=0)
    h_batched, _ = enc(batched)
    assert torch.allclose(h_batched[0], h1[0], atol=1e-6)
    assert torch.allclose(h_batched[1], h2[0], atol=1e-6)
