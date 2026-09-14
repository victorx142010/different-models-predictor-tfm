"""Pruebas de hpo.py: la pérdida con la que se comparan los trials no debe
depender del esquema de ponderación de las tareas.
"""

import numpy as np
import torch

from src.training.hpo import train_and_eval_fold

M_MAX = 8
EMBEDDING_DIM = 768


def _make_tensors(n: int, L: int = 5) -> dict:
    return {
        "quant_seq": torch.randn(n, L, 3),
        "text_today_emb": torch.randn(n, EMBEDDING_DIM),
        "text_today_has_news": torch.rand(n) > 0.5,
        "text_seq_emb": torch.randn(n, L, EMBEDDING_DIM),
        "text_seq_mask": torch.rand(n, L) > 0.5,
        "news_set_emb": torch.randn(n, M_MAX, EMBEDDING_DIM),
        "news_set_mask": torch.rand(n, M_MAX) > 0.5,
        "y_direction": (torch.rand(n) > 0.5).float(),
        "y_vol": torch.rand(n) * 0.05,
    }


def test_scoring_loss_is_fixed_metric_not_raw_uncertainty_loss() -> None:
    """La pérdida con la que se comparan los trials debe ser siempre BCE +
    Huber con pesos fijos 1:1, y por tanto no negativa, aunque el trial se
    entrenara con `learned_uncertainty`. La pérdida de ese esquema suma
    s_dir + s_vol, que pueden ser negativos, y usarla para comparar
    favorecería artificialmente a ese esquema."""
    torch.manual_seed(0)
    train_t = _make_tensors(20)
    val_t = _make_tensors(10)

    params = {
        "hidden_size": 8,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 5e-4,
        "loss_weighting": "learned_uncertainty",
    }
    scoring_loss = train_and_eval_fold("price_only", params, train_t, val_t, "cpu")
    assert scoring_loss >= 0.0


def test_scoring_loss_comparable_across_weighting_schemes() -> None:
    """La pérdida de comparación no debe depender del esquema de ponderación
    con el que se entrenó el trial."""
    torch.manual_seed(0)
    train_t = _make_tensors(20)
    val_t = _make_tensors(10)

    base_params = {"hidden_size": 8, "num_layers": 1, "dropout": 0.2, "learning_rate": 5e-4}

    loss_fixed = train_and_eval_fold(
        "price_only", dict(base_params, loss_weighting="1:1"), train_t, val_t, "cpu"
    )
    loss_uncertainty = train_and_eval_fold(
        "price_only", dict(base_params, loss_weighting="learned_uncertainty"), train_t, val_t, "cpu"
    )
    assert loss_fixed >= 0.0
    assert loss_uncertainty >= 0.0
