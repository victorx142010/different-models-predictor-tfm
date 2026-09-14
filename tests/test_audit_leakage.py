"""Auditoría de fugas de información: comprueba uno a uno los cinco puntos por
los que más fácilmente se cuela información del futuro en un proyecto de series
temporales. Dos de ellos ya tienen pruebas en otros ficheros; los otros tres se
prueban aquí.

1. Ni el GARCH ni el escalador se ajustan con datos posteriores al final del
   entrenamiento de su fold: `test_garch_fold_never_uses_future_returns` (aquí)
   y `test_scaler_fit_only_on_train_not_val` (test_train_control_variants.py).
2. Las noticias asignadas a una sesión se publicaron antes de su cierre:
   test_align_news.py, que incluye una comprobación sobre los datos reales.
3. La etiqueta y_{t,h} solo usa precios posteriores a t y hasta t+h:
   `test_targets_never_use_data_at_or_before_t` y
   `test_targets_never_use_data_beyond_t_plus_h`.
4. Con las etiquetas de entrenamiento barajadas, el acierto en validación cae
   al nivel del azar: `test_shuffled_labels_collapse_to_chance_level`.
5. La deduplicación conserva el titular más antiguo, para que una noticia
   repetida más tarde no parezca información nueva:
   `test_dedup_keeps_earliest_not_latest_occurrence`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.ingestion.clean_news import dedup_near_identical
from src.quant_module.garch_features import _garch_segment
from src.targets.targets import compute_targets
from src.validation.splitter import WalkForwardSplitter


# --------------------------------------------------------------------------
# 1. El GARCH de cada fold no usa datos posteriores al final de su
# entrenamiento
# --------------------------------------------------------------------------


def test_garch_fold_never_uses_future_returns() -> None:
    """Si se cambian los retornos posteriores al final del entrenamiento de
    un fold, por otros con una distribución muy distinta, los parámetros
    GARCH de ese fold no deben cambiar."""
    splitter = WalkForwardSplitter()
    fold = splitter.folds[0]
    sessions = splitter._sessions
    span_mask = (sessions >= fold.train_start) & (sessions <= fold.val_end)
    span_dates = sessions[span_mask]

    rng = np.random.default_rng(0)
    log_return = rng.normal(0, 0.01, size=len(span_dates))
    prices_df = pd.DataFrame(
        {"date": span_dates, "ticker": "FAKE", "log_return": log_return}
    )

    kwargs = dict(
        window_start=fold.train_start,
        window_end=fold.val_end,
        output_start=fold.train_start,
        output_end=fold.val_end,
    )
    result_before = _garch_segment(prices_df, fold, **kwargs)

    prices_mutated = prices_df.copy()
    future_mask = prices_mutated["date"] > fold.train_end
    rng2 = np.random.default_rng(1)
    prices_mutated.loc[future_mask, "log_return"] = rng2.normal(0, 5.0, size=int(future_mask.sum()))

    result_after = _garch_segment(prices_mutated, fold, **kwargs)

    for col in ["garch_omega", "garch_alpha1", "garch_beta1"]:
        assert np.isclose(result_before[col].iloc[0], result_after[col].iloc[0]), (
            f"{col} cambió al mutar datos futuros al fold: posible fuga"
        )


# --------------------------------------------------------------------------
# 3. La etiqueta y_{t,h} solo usa precios posteriores a t y hasta t+h
# --------------------------------------------------------------------------


def _make_price_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_return = rng.normal(0, 0.01, size=n)
    adj_close = 100 * np.exp(np.cumsum(log_return))
    return pd.DataFrame(
        {"date": dates, "ticker": "FAKE", "adj_close": adj_close, "log_return": log_return}
    )


def test_targets_never_use_data_at_or_before_t() -> None:
    """Cambiar los precios del día t o anteriores no debe cambiar y_{t,h}."""
    df = _make_price_df()
    targets_before = compute_targets(df)

    df_mutated = df.copy()
    t_idx = 15
    df_mutated.loc[: t_idx - 1, ["adj_close", "log_return"]] *= 1000  # se cambian todos los días anteriores a t
    df_mutated.loc[t_idx, "log_return"] = 999.0  # y también el propio día t

    targets_after = compute_targets(df_mutated)

    row_before = targets_before.iloc[t_idx]
    row_after = targets_after.iloc[t_idx]
    for h in (1, 3, 5):
        assert row_before[f"d_h{h}"] == row_after[f"d_h{h}"] or (
            pd.isna(row_before[f"d_h{h}"]) and pd.isna(row_after[f"d_h{h}"])
        )
        assert np.isclose(row_before[f"rv_h{h}"], row_after[f"rv_h{h}"], equal_nan=True)


def test_targets_never_use_data_beyond_t_plus_h() -> None:
    """Cambiar los precios posteriores a t+h no debe cambiar y_{t,h}."""
    df = _make_price_df(n=30)
    t_idx = 10
    h = 3

    targets_before = compute_targets(df)

    df_mutated = df.copy()
    df_mutated.loc[t_idx + h + 1 :, ["adj_close", "log_return"]] *= 1000

    targets_after = compute_targets(df_mutated)

    assert targets_before.iloc[t_idx][f"d_h{h}"] == targets_after.iloc[t_idx][f"d_h{h}"]
    assert np.isclose(targets_before.iloc[t_idx][f"rv_h{h}"], targets_after.iloc[t_idx][f"rv_h{h}"])


def test_targets_do_change_when_window_data_changes() -> None:
    """Control positivo: cambiar los precios dentro de (t, t+h] sí debe
    cambiar y_{t,h}. Si no cambiara, las dos pruebas anteriores pasarían
    siempre y no demostrarían nada."""
    df = _make_price_df(n=30)
    t_idx = 10
    h = 3

    targets_before = compute_targets(df)

    df_mutated = df.copy()
    df_mutated.loc[t_idx + 1 : t_idx + h, "log_return"] = 0.5  # dentro de la ventana (t, t+h]
    df_mutated.loc[t_idx + 1 : t_idx + h, "adj_close"] = (
        df_mutated.loc[t_idx, "adj_close"] * np.exp(np.cumsum(np.full(h, 0.5)))
    )

    targets_after = compute_targets(df_mutated)

    assert targets_before.iloc[t_idx][f"d_h{h}"] != targets_after.iloc[t_idx][f"d_h{h}"] or not np.isclose(
        targets_before.iloc[t_idx][f"rv_h{h}"], targets_after.iloc[t_idx][f"rv_h{h}"]
    )


# --------------------------------------------------------------------------
# 5. La deduplicación conserva el titular más antiguo
# --------------------------------------------------------------------------


def test_dedup_keeps_earliest_not_latest_occurrence() -> None:
    """De un grupo de titulares repetidos debe quedar el publicado primero.
    Si quedara una copia publicada más tarde por otra fuente, el modelo
    vería como nueva una información que el mercado ya conocía."""
    df = pd.DataFrame(
        {
            "date_publicacion": pd.to_datetime(
                ["2023-06-01", "2023-08-15", "2022-01-10", "2023-06-02"], utc=True
            ),
            "ticker": ["FAKE"] * 4,
            "titular": ["irrelevante"] * 4,
            "fuente": [None] * 4,
            "texto_limpio": [
                "Nvidia shares jump on strong earnings beat",
                "Nvidia shares jump on strong earnings beat",  # mismo titular, publicado más tarde
                "Nvidia shares jump on strong earnings beat",  # mismo titular, el publicado primero
                "Completely unrelated headline about oil prices",
            ],
        }
    )

    result, n_dropped = dedup_near_identical(df)

    assert n_dropped == 2  # de los 3 titulares repetidos se descartan 2
    kept_dates = set(result["date_publicacion"].dt.strftime("%Y-%m-%d"))
    assert "2022-01-10" in kept_dates  # sobrevive el publicado primero
    assert "2023-06-01" not in kept_dates  # repetido y posterior: se descarta
    assert "2023-08-15" not in kept_dates  # repetido y muy posterior: se descarta


# --------------------------------------------------------------------------
# 4. Con etiquetas barajadas, el acierto cae al nivel del azar
# --------------------------------------------------------------------------


def test_shuffled_labels_collapse_to_chance_level() -> None:
    """Entrena price_only en SPY (fold 2) con las etiquetas de dirección de
    entrenamiento (`d_h1`) barajadas al azar, sin tocar las de validación.
    Si el modelo siguiera acertando en validación después de aprender de
    etiquetas sin sentido, habría una fuga: estaría obteniendo la respuesta
    por otra vía. Lo esperado es un acierto cercano al 50%."""
    import pytest

    interim = Path(__file__).resolve().parents[1] / "data" / "interim"
    price_path = interim / "SPY_prices_clean_2010-01-01_2023-12-31.parquet"
    if not price_path.exists():
        pytest.skip("precios limpios de SPY no generados todavía")

    from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START
    from src.training.sequence_dataset import build_dataset, build_quant_feature_table, load_daily_agg
    from src.training.train_control_variants import (
        DATA_INTERIM_DIR,
        HORIZON,
        L,
        scale_quant_seq,
        to_tensors,
        train_one,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ticker = "SPY"
    splitter = WalkForwardSplitter()
    fold = splitter.folds[1]  # fold 2: buena cobertura de noticias

    quant_df = build_quant_feature_table(ticker)
    daily_agg = load_daily_agg(ticker)
    targets = pd.read_parquet(
        DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    )
    prices = pd.read_parquet(price_path)

    train_dates = splitter.slice_train(prices, fold)["date"]
    val_dates = splitter.slice_val(prices, fold)["date"]
    train_ds = build_dataset(train_dates, quant_df, daily_agg, targets, L, HORIZON)
    val_ds = build_dataset(val_dates, quant_df, daily_agg, targets, L, HORIZON)

    train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
    train_ds = dict(train_ds, quant_seq=train_q)
    val_ds = dict(val_ds, quant_seq=val_q)

    train_t = to_tensors(train_ds, device)
    val_t = to_tensors(val_ds, device)

    _, metrics_real = train_one("price_only", train_t, val_t, device)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_t["y_direction"]))
    train_t_shuffled = dict(train_t)
    train_t_shuffled["y_direction"] = train_t["y_direction"][perm]

    _, metrics_shuffled = train_one("price_only", train_t_shuffled, val_t, device)

    print(
        f"\n[shuffle test] accuracy real={metrics_real['accuracy']:.3f} "
        f"vs. barajado={metrics_shuffled['accuracy']:.3f} (azar~0.5)"
    )

    assert 0.35 <= metrics_shuffled["accuracy"] <= 0.65, (
        f"accuracy con etiquetas barajadas ({metrics_shuffled['accuracy']:.3f}) se aleja "
        "demasiado del nivel de azar (0.5): posible fuga de información en el pipeline"
    )
