"""Pruebas de sequence_dataset.py: ventanas de L sesiones sin filas futuras,
noticias de cada sesión y construcción del dataset, descartando las fechas sin
historia suficiente o sin etiqueta.
"""

import numpy as np
import pandas as pd
import pytest

from src.training.sequence_dataset import (
    EMBEDDING_DIM,
    build_dataset,
    get_news_set_today,
    get_quant_window,
    get_text_sequence,
    get_text_today,
    position_index,
    position_index_from_array,
)


def _make_quant_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "log_return": np.arange(n, dtype="float32") * 0.01,
            "cond_vol": np.arange(n, dtype="float32") * 0.1,
            "std_resid": np.arange(n, dtype="float32") * 0.001,
        }
    )


def _make_daily_agg(n: int = 30, news_days: set[int] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if news_days is None:
        news_days = set(range(n))
    rows = []
    for i, d in enumerate(dates):
        has_news = i in news_days
        rows.append(
            {
                "date": d,
                "ticker": "FAKE",
                "n_news": 3 if has_news else 0,
                "has_news": has_news,
                "embedding": (np.full(EMBEDDING_DIM, float(i), dtype="float32") if has_news else None),
                "p_pos": 0.5 if has_news else np.nan,
            }
        )
    return pd.DataFrame(rows)


def test_quant_window_correct_slice_and_order() -> None:
    df = _make_quant_df()
    pos = position_index(df)
    end_date = df["date"].iloc[10]
    window = get_quant_window(df, pos, end_date, L=5)
    assert window.shape == (5, 3)
    # las filas 6..10 (0-indexed) -> log_return 0.06..0.10
    expected = np.arange(6, 11, dtype="float32") * 0.01
    assert np.allclose(window[:, 0], expected)


def test_quant_window_none_when_insufficient_history() -> None:
    df = _make_quant_df()
    pos = position_index(df)
    end_date = df["date"].iloc[2]  # solo 3 filas disponibles (índices 0,1,2)
    assert get_quant_window(df, pos, end_date, L=5) is None


def test_quant_window_none_when_date_missing() -> None:
    df = _make_quant_df()
    pos = position_index(df)
    missing_date = pd.Timestamp("2099-01-01")
    assert get_quant_window(df, pos, missing_date, L=5) is None


def test_quant_window_never_leaks_future_rows() -> None:
    """Cambiar las filas posteriores a end_date no debe cambiar la ventana."""
    df = _make_quant_df()
    pos = position_index(df)
    end_date = df["date"].iloc[10]

    window_before = get_quant_window(df, pos, end_date, L=5)

    df_mutated = df.copy()
    df_mutated.loc[11:, ["log_return", "cond_vol", "std_resid"]] = 9999.0
    window_after = get_quant_window(df_mutated, pos, end_date, L=5)

    assert np.allclose(window_before, window_after)
    assert not (window_after == 9999.0).any()


def test_text_today_no_news_returns_zero_and_false() -> None:
    agg = _make_daily_agg(news_days=set())
    pos = position_index(agg)
    emb, has_news = get_text_today(agg, pos, agg["date"].iloc[5])
    assert has_news is False
    assert (emb == 0).all()


def test_text_today_with_news_returns_embedding() -> None:
    agg = _make_daily_agg(news_days={5})
    pos = position_index(agg)
    emb, has_news = get_text_today(agg, pos, agg["date"].iloc[5])
    assert has_news is True
    assert np.allclose(emb, 5.0)


def test_text_sequence_mask_matches_news_days() -> None:
    agg = _make_daily_agg(news_days={6, 8})
    pos = position_index(agg)
    end_date = agg["date"].iloc[9]  # ventana de 5: días 5,6,7,8,9
    seq, mask = get_text_sequence(agg, pos, end_date, L=5)
    assert seq.shape == (5, EMBEDDING_DIM)
    assert mask.tolist() == [False, True, False, True, False]
    assert np.allclose(seq[1], 6.0)
    assert np.allclose(seq[3], 8.0)
    assert (seq[0] == 0).all()


def test_build_dataset_shapes_and_drops_invalid_dates() -> None:
    quant_df = _make_quant_df(n=30)
    agg = _make_daily_agg(n=30, news_days={10, 15, 20})

    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    targets_df = pd.DataFrame(
        {
            "date": dates,
            "d_h1": [1.0] * 25 + [np.nan] * 5,  # las 5 últimas, sin etiqueta
            "rv_h1": [0.02] * 25 + [np.nan] * 5,
        }
    )

    # fechas desde el principio (sin historia suficiente) hasta el final
    result = build_dataset(dates, quant_df, agg, targets_df, L=5, horizon=1)

    assert result is not None
    # se descartan las 4 primeras (sin historia suficiente) y las 5 últimas
    # (sin etiqueta)
    assert len(result["date"]) == 30 - 4 - 5
    assert result["quant_seq"].shape == (len(result["date"]), 5, 3)
    assert result["text_seq_emb"].shape == (len(result["date"]), 5, EMBEDDING_DIM)
    assert result["text_today_emb"].shape == (len(result["date"]), EMBEDDING_DIM)
    assert not np.isnan(result["y_direction"]).any()
    assert not np.isnan(result["y_vol"]).any()


def test_build_dataset_none_when_no_valid_samples() -> None:
    quant_df = _make_quant_df(n=3)  # nunca hay suficiente historial para L=5
    agg = _make_daily_agg(n=3)
    dates = quant_df["date"]
    targets_df = pd.DataFrame({"date": dates, "d_h1": [1.0] * 3, "rv_h1": [0.01] * 3})

    result = build_dataset(dates, quant_df, agg, targets_df, L=5, horizon=1)
    assert result is None


def _make_variable_set(m_max: int = 32) -> dict:
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    rng = np.random.default_rng(0)
    embeddings = np.zeros((3, m_max, EMBEDDING_DIM), dtype="float32")
    mask = np.zeros((3, m_max), dtype=bool)
    embeddings[0, :4] = rng.standard_normal((4, EMBEDDING_DIM))
    mask[0, :4] = True
    # día 1 (índice 1) sin ninguna noticia (todo False/cero)
    embeddings[2, :2] = rng.standard_normal((2, EMBEDDING_DIM))
    mask[2, :2] = True
    return {
        "session_date": dates.values.astype("datetime64[ns]"),
        "embeddings": embeddings,
        "attention_mask": mask,
        "n_real": np.array([4, 0, 2], dtype="int32"),
        "n_truncated": np.zeros(3, dtype="int32"),
    }


def test_get_news_set_today_returns_stored_arrays() -> None:
    vs = _make_variable_set()
    pos = position_index_from_array(vs["session_date"])
    emb, mask = get_news_set_today(vs, pos, pd.Timestamp("2020-01-01"))
    assert mask.sum() == 4
    assert np.allclose(emb[:4], vs["embeddings"][0, :4])


def test_get_news_set_today_missing_date_returns_zero() -> None:
    vs = _make_variable_set()
    pos = position_index_from_array(vs["session_date"])
    emb, mask = get_news_set_today(vs, pos, pd.Timestamp("2099-01-01"))
    assert not mask.any()
    assert (emb == 0).all()


def test_build_dataset_includes_news_set_when_provided() -> None:
    quant_df = _make_quant_df(n=30)
    agg = _make_daily_agg(n=30, news_days={10, 15, 20})
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    targets_df = pd.DataFrame(
        {"date": dates, "d_h1": [1.0] * 30, "rv_h1": [0.02] * 30}
    )

    m_max = 32
    rng = np.random.default_rng(1)
    embeddings = np.zeros((30, m_max, EMBEDDING_DIM), dtype="float32")
    mask = np.zeros((30, m_max), dtype=bool)
    embeddings[10, :3] = rng.standard_normal((3, EMBEDDING_DIM))
    mask[10, :3] = True
    variable_set = {
        "session_date": dates.values.astype("datetime64[ns]"),
        "embeddings": embeddings,
        "attention_mask": mask,
    }

    result = build_dataset(dates, quant_df, agg, targets_df, L=5, horizon=1, variable_set=variable_set)
    assert "news_set_emb" in result
    assert "news_set_mask" in result
    assert result["news_set_emb"].shape == (len(result["date"]), m_max, EMBEDDING_DIM)

    idx_10 = list(result["date"]).index(pd.Timestamp("2020-01-15"))  # sesión de índice 10 (días hábiles, contando desde 0)
    assert result["news_set_mask"][idx_10].sum() == 3


def test_build_dataset_without_variable_set_omits_news_set_keys() -> None:
    quant_df = _make_quant_df(n=10)
    agg = _make_daily_agg(n=10)
    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    targets_df = pd.DataFrame({"date": dates, "d_h1": [1.0] * 10, "rv_h1": [0.02] * 10})

    result = build_dataset(dates, quant_df, agg, targets_df, L=5, horizon=1)
    assert "news_set_emb" not in result
    assert "news_set_mask" not in result
