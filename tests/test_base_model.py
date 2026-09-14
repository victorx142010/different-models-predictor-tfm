"""Pruebas de base_model.py: formas de salida de las cinco variantes, uso del
vector nulo cuando no hay noticias, papel de la conexión residual en
cross_attention y funciones de pérdida multitarea.
"""

import pytest
import torch

from src.fusion_model.base_model import (
    EMBEDDING_DIM,
    MultiModalModel,
    UncertaintyWeightedLoss,
    multitask_loss,
)


M_MAX_TEST = 8


def _make_batch(B: int = 4, L: int = 20, m_max: int = M_MAX_TEST) -> dict:
    news_set_mask = torch.rand(B, m_max) > 0.5
    news_set_mask[:, 0] = True  # asegurar al menos 1 posición real por fila
    return {
        "quant_seq": torch.randn(B, L, 3),
        "text_today_emb": torch.randn(B, EMBEDDING_DIM),
        "text_today_has_news": torch.tensor([True, False, True, False][:B]),
        "text_seq_emb": torch.randn(B, L, EMBEDDING_DIM),
        "text_seq_mask": torch.rand(B, L) > 0.5,
        "news_set_emb": torch.randn(B, m_max, EMBEDDING_DIM),
        "news_set_mask": news_set_mask,
    }


@pytest.mark.parametrize(
    "variant", ["early_fusion", "late_fusion", "price_only", "text_only", "cross_attention"]
)
def test_forward_shapes_all_variants(variant: str) -> None:
    model = MultiModalModel(
        variant=variant, lstm_hidden_size=16, text_proj_dim=8, dense_hidden=8, attn_d_k=8, attn_n_heads=2
    )
    batch = _make_batch(B=4, L=10)
    out = model(batch)
    assert out["direction_logit"].shape == (4,)
    assert out["vol_pred"].shape == (4,)


def test_vol_pred_always_nonnegative() -> None:
    model = MultiModalModel(variant="late_fusion", lstm_hidden_size=16, text_proj_dim=8, dense_hidden=8)
    batch = _make_batch()
    out = model(batch)
    assert (out["vol_pred"] >= 0).all()


def test_null_embedding_ignores_masked_input_value() -> None:
    """Si has_news=False, el embedding de entrada no debe influir en la
    salida, porque se sustituye por el vector nulo aprendido."""
    model = MultiModalModel(variant="text_only", lstm_hidden_size=16, text_proj_dim=8, dense_hidden=8)
    model.eval()

    has_news = torch.tensor([False])
    emb_a = torch.randn(1, EMBEDDING_DIM)
    emb_b = torch.randn(1, EMBEDDING_DIM) * 100  # deliberadamente muy distinto

    batch_a = {
        "quant_seq": torch.randn(1, 5, 3),
        "text_today_emb": emb_a,
        "text_today_has_news": has_news,
        "text_seq_emb": torch.randn(1, 5, EMBEDDING_DIM),
        "text_seq_mask": torch.zeros(1, 5, dtype=torch.bool),
    }
    batch_b = dict(batch_a, text_today_emb=emb_b)

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)

    assert torch.allclose(out_a["direction_logit"], out_b["direction_logit"])
    assert torch.allclose(out_a["vol_pred"], out_b["vol_pred"])


def test_null_embedding_used_when_present_changes_output() -> None:
    """Si has_news=True, cambiar el embedding sí debe cambiar la salida."""
    model = MultiModalModel(variant="text_only", lstm_hidden_size=16, text_proj_dim=8, dense_hidden=8)
    model.eval()

    has_news = torch.tensor([True])
    batch_a = {
        "quant_seq": torch.randn(1, 5, 3),
        "text_today_emb": torch.randn(1, EMBEDDING_DIM),
        "text_today_has_news": has_news,
        "text_seq_emb": torch.randn(1, 5, EMBEDDING_DIM),
        "text_seq_mask": torch.zeros(1, 5, dtype=torch.bool),
    }
    batch_b = dict(batch_a, text_today_emb=torch.randn(1, EMBEDDING_DIM) * 100)

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)

    assert not torch.allclose(out_a["direction_logit"], out_b["direction_logit"])


def test_price_only_ignores_text_inputs() -> None:
    model = MultiModalModel(variant="price_only", lstm_hidden_size=16, dense_hidden=8)
    model.eval()
    quant = torch.randn(2, 10, 3)
    batch_a = {
        "quant_seq": quant,
        "text_today_emb": torch.randn(2, EMBEDDING_DIM),
        "text_today_has_news": torch.tensor([True, False]),
        "text_seq_emb": torch.randn(2, 10, EMBEDDING_DIM),
        "text_seq_mask": torch.rand(2, 10) > 0.5,
    }
    batch_b = dict(batch_a, text_today_emb=torch.randn(2, EMBEDDING_DIM) * 100)

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)
    assert torch.allclose(out_a["direction_logit"], out_b["direction_logit"])


def test_multitask_loss_combines_bce_and_huber() -> None:
    direction_logit = torch.tensor([0.5, -0.5, 2.0])
    vol_pred = torch.tensor([0.1, 0.2, 0.3])
    y_direction = torch.tensor([1.0, 0.0, 1.0])
    y_vol = torch.tensor([0.12, 0.18, 0.25])

    total, parts = multitask_loss(direction_logit, vol_pred, y_direction, y_vol)
    assert total.item() > 0
    assert set(parts.keys()) == {"bce", "huber"}
    assert abs(total.item() - (parts["bce"] + parts["huber"])) < 1e-5


def test_multitask_loss_respects_lambda_weights() -> None:
    direction_logit = torch.tensor([0.5, -0.5])
    vol_pred = torch.tensor([0.1, 0.2])
    y_direction = torch.tensor([1.0, 0.0])
    y_vol = torch.tensor([0.5, 0.5])  # error grande -> huber alto

    total_equal, _ = multitask_loss(direction_logit, vol_pred, y_direction, y_vol, 1.0, 1.0)
    total_no_vol, _ = multitask_loss(direction_logit, vol_pred, y_direction, y_vol, 1.0, 0.0)
    assert total_no_vol.item() < total_equal.item()


def test_invalid_variant_raises() -> None:
    with pytest.raises(ValueError):
        MultiModalModel(variant="nonexistent")


def test_cross_attention_returns_weights() -> None:
    model = MultiModalModel(
        variant="cross_attention", lstm_hidden_size=16, dense_hidden=8, attn_d_k=8, attn_n_heads=2
    )
    batch = _make_batch(B=3)
    out = model(batch)
    assert out["attn_weights"] is not None
    assert out["attn_weights"].shape == (3, 2, M_MAX_TEST)


def test_other_variants_have_no_attn_weights() -> None:
    model = MultiModalModel(variant="late_fusion", lstm_hidden_size=16, text_proj_dim=8, dense_hidden=8)
    out = model(_make_batch(B=2))
    assert out["attn_weights"] is None


def test_cross_attention_no_news_day_uses_null_context() -> None:
    """Si `news_set_mask` es todo False, los embeddings de noticias no deben
    influir en la salida, porque se usa el contexto nulo aprendido."""
    model = MultiModalModel(
        variant="cross_attention", lstm_hidden_size=16, dense_hidden=8, attn_d_k=8, attn_n_heads=2
    )
    model.eval()

    quant = torch.randn(1, 10, 3)
    mask_no_news = torch.zeros(1, M_MAX_TEST, dtype=torch.bool)

    batch_a = {
        "quant_seq": quant,
        "text_today_emb": torch.randn(1, EMBEDDING_DIM),
        "text_today_has_news": torch.tensor([False]),
        "text_seq_emb": torch.randn(1, 10, EMBEDDING_DIM),
        "text_seq_mask": torch.zeros(1, 10, dtype=torch.bool),
        "news_set_emb": torch.randn(1, M_MAX_TEST, EMBEDDING_DIM),
        "news_set_mask": mask_no_news,
    }
    batch_b = dict(batch_a, news_set_emb=torch.randn(1, M_MAX_TEST, EMBEDDING_DIM) * 100)

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)

    assert torch.allclose(out_a["direction_logit"], out_b["direction_logit"])
    assert torch.allclose(out_a["vol_pred"], out_b["vol_pred"])


def test_cross_attention_with_news_day_uses_real_context() -> None:
    """Si hay al menos una noticia, cambiar su embedding sí debe cambiar la
    salida: se usa la atención y no el contexto nulo."""
    model = MultiModalModel(
        variant="cross_attention", lstm_hidden_size=16, dense_hidden=8, attn_d_k=8, attn_n_heads=2
    )
    model.eval()

    quant = torch.randn(1, 10, 3)
    mask_with_news = torch.zeros(1, M_MAX_TEST, dtype=torch.bool)
    mask_with_news[0, :3] = True

    batch_a = {
        "quant_seq": quant,
        "text_today_emb": torch.randn(1, EMBEDDING_DIM),
        "text_today_has_news": torch.tensor([True]),
        "text_seq_emb": torch.randn(1, 10, EMBEDDING_DIM),
        "text_seq_mask": torch.zeros(1, 10, dtype=torch.bool),
        "news_set_emb": torch.randn(1, M_MAX_TEST, EMBEDDING_DIM),
        "news_set_mask": mask_with_news,
    }
    batch_b = dict(batch_a, news_set_emb=torch.randn(1, M_MAX_TEST, EMBEDDING_DIM) * 100)

    with torch.no_grad():
        out_a = model(batch_a)
        out_b = model(batch_b)

    assert not torch.allclose(out_a["direction_logit"], out_b["direction_logit"])


def test_cross_attention_residual_connection_with_h_L() -> None:
    """Cambiar quant_seq, y con ello h_L, debe cambiar la salida aunque las
    noticias sean las mismas. Confirma que h_L entra en la fusión final a
    través de la conexión residual, y no solo como consulta de la atención."""
    model = MultiModalModel(
        variant="cross_attention", lstm_hidden_size=16, dense_hidden=8, attn_d_k=8, attn_n_heads=2
    )
    model.eval()

    batch = _make_batch(B=1)
    batch_b = dict(batch, quant_seq=torch.randn(1, 20, 3) * 10)

    with torch.no_grad():
        out_a = model(batch)
        out_b = model(batch_b)

    assert not torch.allclose(out_a["direction_logit"], out_b["direction_logit"])


def test_uncertainty_loss_starts_at_equal_weighting() -> None:
    """Al inicio s_dir = s_vol = 0, así que e^0 = 1 y la pérdida equivale a
    ponderar las dos tareas 1:1."""
    loss_mod = UncertaintyWeightedLoss()
    direction_logit = torch.tensor([0.5, -0.5])
    vol_pred = torch.tensor([0.1, 0.2])
    y_direction = torch.tensor([1.0, 0.0])
    y_vol = torch.tensor([0.5, 0.5])

    total_uw, parts = loss_mod(direction_logit, vol_pred, y_direction, y_vol)
    total_fixed, _ = multitask_loss(direction_logit, vol_pred, y_direction, y_vol, 1.0, 1.0)
    assert abs(total_uw.item() - total_fixed.item()) < 1e-5
    assert parts["s_dir"] == 0.0 and parts["s_vol"] == 0.0


def test_uncertainty_loss_weights_adapt_during_training() -> None:
    """Tras varios pasos de entrenamiento, s_dir y s_vol deben alejarse de
    su valor inicial: se aprenden, no son constantes."""
    torch.manual_seed(0)
    model = MultiModalModel(variant="price_only", lstm_hidden_size=8, dense_hidden=8)
    loss_mod = UncertaintyWeightedLoss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_mod.parameters()), lr=0.05)

    batch = _make_batch(B=16)
    y_direction = torch.randint(0, 2, (16,)).float()
    y_vol = torch.rand(16) * 0.05

    for _ in range(30):
        optimizer.zero_grad()
        out = model(batch)
        loss, _ = loss_mod(out["direction_logit"], out["vol_pred"], y_direction, y_vol)
        loss.backward()
        optimizer.step()

    assert loss_mod.s_dir.item() != 0.0
    assert loss_mod.s_vol.item() != 0.0
