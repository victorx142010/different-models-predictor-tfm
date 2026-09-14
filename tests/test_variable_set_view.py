"""Pruebas de variable_set_view.py: cada sesión conserva como máximo M_MAX
noticias, las de menor p_neu, y los huecos se rellenan con ceros y quedan
marcados en la máscara.
"""

import numpy as np
import pandas as pd
import pytest

from src.nlp_module.variable_set_view import EMBEDDING_DIM, M_MAX, _select_top_m, build_variable_set_view


def _make_fake_embeddings_parquet(tmp_path, monkeypatch, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    features_dir = tmp_path / "features" / "nlp"
    features_dir.mkdir(parents=True)
    df.to_parquet(features_dir / "FAKE_news_embeddings.parquet", index=False)

    import src.nlp_module.variable_set_view as mod

    monkeypatch.setattr(mod, "FEATURES_NLP_DIR", features_dir)


def _fake_row(session_date: str, p_neu: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "session_date": pd.Timestamp(session_date),
        "p_pos": 0.3,
        "p_neg": 0.3,
        "p_neu": p_neu,
        "embedding": rng.standard_normal(EMBEDDING_DIM).astype("float32"),
    }


def test_select_top_m_keeps_all_when_under_limit() -> None:
    g = pd.DataFrame({"p_neu": [0.9, 0.1, 0.5]})
    result = _select_top_m(g, m_max=5)
    assert len(result) == 3


def test_select_top_m_keeps_lowest_p_neu_when_over_limit() -> None:
    g = pd.DataFrame({"p_neu": [0.9, 0.1, 0.5, 0.05, 0.7]})
    result = _select_top_m(g, m_max=2)
    assert sorted(result["p_neu"].tolist()) == [0.05, 0.1]


def test_variable_set_shapes_and_mask(tmp_path, monkeypatch) -> None:
    rows = [_fake_row("2023-01-03", 0.2, seed=i) for i in range(5)]
    rows += [_fake_row("2023-01-04", 0.2, seed=100 + i) for i in range(2)]
    _make_fake_embeddings_parquet(tmp_path, monkeypatch, rows)

    data = build_variable_set_view("FAKE")

    assert data["embeddings"].shape == (2, M_MAX, EMBEDDING_DIM)
    assert data["attention_mask"].shape == (2, M_MAX)
    assert data["n_real"].tolist() == [5, 2]
    assert data["n_truncated"].tolist() == [0, 0]

    day0_mask = data["attention_mask"][0]
    assert day0_mask[:5].all()
    assert not day0_mask[5:].any()


def test_variable_set_zero_pads_unused_slots(tmp_path, monkeypatch) -> None:
    rows = [_fake_row("2023-01-03", 0.2, seed=1)]
    _make_fake_embeddings_parquet(tmp_path, monkeypatch, rows)

    data = build_variable_set_view("FAKE")
    padded = data["embeddings"][0, 1:]
    assert (padded == 0).all()


def test_variable_set_truncates_and_keeps_marked_sentiment(tmp_path, monkeypatch) -> None:
    # 40 noticias en un día, más que M_MAX=32: solo deben quedar las 32 de
    # menor p_neu, las de sentimiento más marcado
    rows = []
    for i in range(40):
        p_neu = i / 40.0  # valores 0.0, 0.025, ..., 0.975 (todos distintos)
        rows.append(_fake_row("2023-01-05", p_neu, seed=i))
    _make_fake_embeddings_parquet(tmp_path, monkeypatch, rows)

    data = build_variable_set_view("FAKE")
    assert data["n_real"][0] == M_MAX
    assert data["n_truncated"][0] == 40 - M_MAX
    assert data["attention_mask"][0].sum() == M_MAX


def test_variable_set_session_dates_sorted(tmp_path, monkeypatch) -> None:
    rows = [_fake_row("2023-01-05", 0.2, seed=1), _fake_row("2023-01-03", 0.2, seed=2)]
    _make_fake_embeddings_parquet(tmp_path, monkeypatch, rows)

    data = build_variable_set_view("FAKE")
    dates = pd.DatetimeIndex(data["session_date"])
    assert list(dates) == sorted(dates)
