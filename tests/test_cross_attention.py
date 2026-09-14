"""Pruebas de la atención cruzada (cross_attention.py): formas de salida, pesos
que suman uno, posiciones enmascaradas sin influencia y pesos que dependen de
la consulta.
"""

import pytest
import torch

from src.fusion_model.cross_attention import CrossAttention


def test_output_shapes() -> None:
    attn_mod = CrossAttention(query_dim=32, kv_dim=768, d_k=64, n_heads=4)
    query = torch.randn(5, 32)
    kv = torch.randn(5, 32, 768)
    mask = torch.rand(5, 32) > 0.3
    mask[:, 0] = True  # asegurar al menos 1 posición real por fila

    context, attn = attn_mod(query, kv, mask)
    assert context.shape == (5, 64)
    assert attn.shape == (5, 4, 32)


def test_attention_weights_sum_to_one() -> None:
    attn_mod = CrossAttention(query_dim=16, kv_dim=768, d_k=32, n_heads=2)
    query = torch.randn(3, 16)
    kv = torch.randn(3, 10, 768)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[0, 5:] = False  # primera fila: solo 5 posiciones reales

    _, attn = attn_mod(query, kv, mask)
    sums = attn.sum(dim=-1)  # [B, H]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_positions_get_near_zero_weight() -> None:
    attn_mod = CrossAttention(query_dim=8, kv_dim=16, d_k=8, n_heads=2)
    attn_mod.eval()
    query = torch.randn(1, 8)
    kv = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, False, False, False, False]])

    _, attn = attn_mod(query, kv, mask)
    assert (attn[:, :, 2:] < 1e-6).all()


def test_all_masked_row_does_not_produce_nan() -> None:
    attn_mod = CrossAttention(query_dim=8, kv_dim=16, d_k=8, n_heads=2)
    query = torch.randn(2, 8)
    kv = torch.randn(2, 5, 16)
    mask = torch.zeros(2, 5, dtype=torch.bool)  # ninguna posición real

    context, attn = attn_mod(query, kv, mask)
    assert not torch.isnan(context).any()
    assert not torch.isnan(attn).any()
    # sin ninguna posición real, los pesos quedan repartidos por igual
    assert torch.allclose(attn, torch.full_like(attn, 1.0 / 5), atol=1e-4)


def test_masked_key_values_do_not_affect_context() -> None:
    """Cambiar el valor de una posición enmascarada no debe alterar el
    contexto: el relleno no puede aportar información."""
    attn_mod = CrossAttention(query_dim=8, kv_dim=16, d_k=8, n_heads=2)
    attn_mod.eval()
    query = torch.randn(1, 8)
    mask = torch.tensor([[True, True, False]])

    kv_a = torch.randn(1, 3, 16)
    kv_b = kv_a.clone()
    kv_b[0, 2] = torch.randn(16) * 1000  # solo cambia la posición enmascarada

    with torch.no_grad():
        context_a, _ = attn_mod(query, kv_a, mask)
        context_b, _ = attn_mod(query, kv_b, mask)

    assert torch.allclose(context_a, context_b, atol=1e-5)


def test_d_k_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError):
        CrossAttention(query_dim=8, kv_dim=16, d_k=10, n_heads=3)


def test_different_query_changes_attention_distribution() -> None:
    """Los pesos de atención deben depender de la consulta, que viene del
    estado de la LSTM. Si dos consultas muy distintas dieran los mismos
    pesos, la atención no aportaría nada."""
    attn_mod = CrossAttention(query_dim=8, kv_dim=16, d_k=8, n_heads=1)
    attn_mod.eval()
    kv = torch.randn(1, 6, 16)
    mask = torch.ones(1, 6, dtype=torch.bool)

    q1 = torch.randn(1, 8)
    q2 = torch.randn(1, 8) * 5  # query muy distinto

    with torch.no_grad():
        _, attn1 = attn_mod(q1, kv, mask)
        _, attn2 = attn_mod(q2, kv, mask)

    assert not torch.allclose(attn1, attn2)
