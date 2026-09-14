"""Convierte las tablas de variables (precio y GARCH, agregación diaria de
noticias y conjunto variable) en las secuencias que recibe el modelo, sin dejar
pasar información del futuro.

La regla de todo el módulo: la muestra de la fecha t solo usa información con
fecha menor o igual que t. Las etiquetas (d_h{h}, rv_h{h}) sí dependen de
precios posteriores a t, pero son lo que se predice, nunca una entrada.

Se construyen tres tipos de secuencia:
- `quant_seq`: la ventana de L sesiones de [log_return, cond_vol, std_resid]
  que recibe la LSTM. Sale de una serie continua que une el tramo de
  calentamiento con las variables GARCH por fold, para que las primeras fechas
  de 2015 ya tengan L sesiones de historia.
- `text_today_*`: el embedding agregado del día t, para late_fusion y
  text_only.
- `text_seq_*`: el embedding agregado de cada uno de los L días de la ventana,
  para early_fusion, que concatena el texto en cada paso de la LSTM.

En los días sin noticias el relleno es un vector de ceros acompañado de la
máscara `has_news`. Todavía no es el vector nulo aprendido: es el modelo
(base_model.py) el que usa la máscara para sustituir ese relleno por su
parámetro entrenado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START
from src.quant_module.technical_indicators import compute_technical_indicators
from src.validation.splitter import WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_QUANT_DIR = PROJECT_ROOT / "features" / "quant"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"

EMBEDDING_DIM = 768


def _fold_garch_path(ticker: str, garch_suffix: str = "") -> Path:
    """Ruta del fichero de variables GARCH por fold de un activo. El rango
    de fechas forma parte del nombre, así que se reconstruye a partir del
    splitter. `garch_suffix` permite leer otra versión, como las variables
    GJR-GARCH ("gjr")."""
    splitter = WalkForwardSplitter()
    span_start = splitter.folds[0].train_start.strftime("%Y-%m-%d")
    span_end = splitter.folds[-1].val_end.strftime("%Y-%m-%d")
    infix = f"_{garch_suffix}" if garch_suffix else ""
    return FEATURES_QUANT_DIR / f"{ticker}_garch_features{infix}_{span_start}_{span_end}.parquet"


def build_quant_feature_table(
    ticker: str,
    garch_suffix: str = "",
    include_volume: bool = False,
    include_technical: bool = False,
    include_garch_forecast: bool = False,
    garch_forecast_horizon: int | None = None,
) -> pd.DataFrame:
    """Une el tramo de calentamiento con las variables GARCH por fold en una
    serie continua [date, log_return, cond_vol, std_resid], de la que
    después se recortan las ventanas de L sesiones. Con `garch_suffix="gjr"`
    usa las variables GJR-GARCH.

    Las opciones siguientes solo las usan los diagnósticos de
    feature_engineering_diagnostics.py, no el modelo final:
    - `include_volume`: añade `log_volume` = log(1 + volumen). El logaritmo
      evita que los pocos días de volumen extremo dominen la
      estandarización.
    - `include_technical`: añade `rsi_14` y `atr_14_norm`
      (technical_indicators.py). Como necesitan 14 sesiones previas, se
      descartan las primeras filas de la serie, todas de 2010.
    - `include_garch_forecast`: añade `garch_fcst_vol`, el pronóstico GARCH
      de volatilidad acumulada a `garch_forecast_horizon` sesiones, con los
      mismos parámetros por fold. Sirve para comprobar si dárselo a la LSTM
      la acerca a ARIMA-GARCH en volatilidad.
    """
    prices = pd.read_parquet(
        DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    )
    infix = f"_{garch_suffix}" if garch_suffix else ""
    warmup = pd.read_parquet(FEATURES_QUANT_DIR / f"{ticker}_garch_features{infix}_warmup.parquet")
    folds = pd.read_parquet(_fold_garch_path(ticker, garch_suffix))

    garch = pd.concat(
        [warmup[["date", "cond_vol", "std_resid"]], folds[["date", "cond_vol", "std_resid"]]],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)
    assert garch["date"].is_unique, "fechas duplicadas entre calentamiento y features por fold"

    merged = garch.merge(prices[["date", "log_return"]], on="date", how="left")
    assert merged["log_return"].notna().all(), "faltan log_return para alguna fecha con features GARCH"

    cols = ["date", "log_return", "cond_vol", "std_resid"]
    if include_volume:
        merged = merged.merge(prices[["date", "volume"]], on="date", how="left")
        assert merged["volume"].notna().all(), "faltan volume para alguna fecha con features GARCH"
        merged["log_volume"] = np.log1p(merged["volume"])
        cols.append("log_volume")

    if include_technical:
        indicators = compute_technical_indicators(prices)
        merged = merged.merge(indicators, on="date", how="left")
        cols += ["rsi_14", "atr_14_norm"]

    if include_garch_forecast:
        if garch_forecast_horizon is None:
            raise ValueError("include_garch_forecast=True requiere pasar garch_forecast_horizon")
        from src.quant_module.garch_features import compute_garch_features, compute_warmup_segment

        o = 1 if garch_suffix == "gjr" else 0
        splitter = WalkForwardSplitter()
        fresh_warmup = compute_warmup_segment(
            prices, splitter.folds[0].train_start, o=o, forecast_horizon=garch_forecast_horizon
        )
        fresh_folds = compute_garch_features(prices, splitter, o=o, forecast_horizon=garch_forecast_horizon)
        fcst = pd.concat(
            [fresh_warmup[["date", "garch_fcst_vol"]], fresh_folds[["date", "garch_fcst_vol"]]], ignore_index=True
        )
        merged = merged.merge(fcst, on="date", how="left")
        assert merged["garch_fcst_vol"].notna().all(), "faltan garch_fcst_vol para alguna fecha con features GARCH"
        cols.append("garch_fcst_vol")

    result = merged[cols]
    if include_technical:
        result = result.dropna(subset=["rsi_14", "atr_14_norm"]).reset_index(drop=True)
    return result


def targets_with_horizon(ticker: str, horizon: int) -> pd.DataFrame:
    """Etiquetas de un activo para el horizonte pedido. Los horizontes 1, 3
    y 5 ya están guardados; los demás (20, 60...) se calculan en memoria con
    `compute_targets`."""
    targets_path = DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    canonical = pd.read_parquet(targets_path)
    if f"d_h{horizon}" in canonical.columns:
        return canonical
    from src.targets.targets import compute_targets

    prices = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet")
    extra = compute_targets(prices, horizons=(horizon,))
    return canonical.merge(extra, on=["date", "ticker"], how="left")


def load_daily_agg(ticker: str) -> pd.DataFrame:
    """Carga la agregación diaria (un vector de texto por sesión) de un
    activo."""
    return pd.read_parquet(FEATURES_NLP_DIR / f"{ticker}_news_daily_agg_simple.parquet")


def load_variable_set(ticker: str) -> dict[str, np.ndarray]:
    """Carga el conjunto variable de noticias de un activo. Solo contiene
    las sesiones con al menos una noticia."""
    data = np.load(FEATURES_NLP_DIR / f"{ticker}_news_embeddings_variable_set.npz")
    return {k: data[k] for k in data.files}


def get_news_set_today(
    variable_set: dict[str, np.ndarray], pos_index: dict, date: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (embeddings [M_MAX, 768], mask [M_MAX]) de las noticias de
    `date`. Si ese día no tiene noticias, devuelve ceros y una máscara en
    False; el modelo sustituye esos días por su contexto nulo aprendido."""
    pos = pos_index.get(date)
    m_max = variable_set["embeddings"].shape[1]
    if pos is None:
        return np.zeros((m_max, EMBEDDING_DIM), dtype="float32"), np.zeros(m_max, dtype=bool)
    return variable_set["embeddings"][pos], variable_set["attention_mask"][pos]


def position_index(df: pd.DataFrame, date_col: str = "date") -> dict[pd.Timestamp, int]:
    """Diccionario fecha -> posición en un DataFrame ordenado por fecha.
    Permite recortar ventanas por posición, mucho más rápido que comparar
    fechas en cada llamada."""
    return {d: i for i, d in enumerate(df[date_col])}


def position_index_from_array(dates: np.ndarray) -> dict[pd.Timestamp, int]:
    """Como `position_index`, pero sobre un array de fechas (por ejemplo, el
    `session_date` del conjunto variable)."""
    return {pd.Timestamp(d): i for i, d in enumerate(dates)}


def get_quant_window(
    quant_df: pd.DataFrame, pos_index: dict, end_date: pd.Timestamp, L: int
) -> np.ndarray | None:
    """Ventana de L sesiones que termina en `end_date`, incluida. Devuelve
    None si la fecha no está o no hay historia suficiente, en lugar de
    devolver una ventana más corta. Toma todas las columnas de `quant_df`
    salvo la fecha, así que funciona con cualquier número de variables."""
    pos = pos_index.get(end_date)
    if pos is None:
        return None
    start_pos = pos - L + 1
    if start_pos < 0:
        return None
    feature_cols = [c for c in quant_df.columns if c != "date"]
    window = quant_df.iloc[start_pos : pos + 1][feature_cols]
    return window.to_numpy(dtype="float32")


def get_text_today(
    daily_agg_df: pd.DataFrame, pos_index: dict, date: pd.Timestamp
) -> tuple[np.ndarray, bool]:
    """Devuelve (embedding, has_news) solo del día `date`, para las
    variantes que usan únicamente el texto del día."""
    pos = pos_index.get(date)
    if pos is None:
        return np.zeros(EMBEDDING_DIM, dtype="float32"), False
    row = daily_agg_df.iloc[pos]
    if not bool(row["has_news"]):
        return np.zeros(EMBEDDING_DIM, dtype="float32"), False
    return np.asarray(row["embedding"], dtype="float32"), True


def get_text_sequence(
    daily_agg_df: pd.DataFrame, pos_index: dict, end_date: pd.Timestamp, L: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Devuelve (embeddings [L, 768], has_news_mask [L]) de los L días que
    terminan en `end_date`: cada paso lleva el embedding de su propio día,
    como necesita early_fusion."""
    pos = pos_index.get(end_date)
    if pos is None:
        return None
    start_pos = pos - L + 1
    if start_pos < 0:
        return None

    seq = np.zeros((L, EMBEDDING_DIM), dtype="float32")
    mask = np.zeros(L, dtype="bool")
    sub = daily_agg_df.iloc[start_pos : pos + 1]
    for i, has_news, emb in zip(range(L), sub["has_news"].to_numpy(), sub["embedding"].to_numpy()):
        if has_news:
            seq[i] = np.asarray(emb, dtype="float32")
            mask[i] = True
    return seq, mask


def build_dataset(
    dates: pd.Series,
    quant_df: pd.DataFrame,
    daily_agg_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    L: int,
    horizon: int,
    variable_set: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray] | None:
    """Construye el dataset (secuencias y etiquetas) para una lista de
    fechas. Cada fecha se descarta si le falta alguna pieza, como una
    ventana incompleta o una etiqueta en NaN, así que puede haber menos
    muestras que fechas.

    Si se pasa `variable_set`, añade `news_set_emb` [N, M_MAX, 768] y
    `news_set_mask` [N, M_MAX], el conjunto de noticias del día que necesita
    cross_attention."""
    quant_pos = position_index(quant_df)
    text_pos = position_index(daily_agg_df)
    targets_by_date = targets_df.set_index("date")
    var_set_pos = position_index_from_array(variable_set["session_date"]) if variable_set else None

    kept_dates: list = []
    quant_seqs, text_today_emb, text_today_has_news = [], [], []
    text_seq_emb, text_seq_mask = [], []
    news_set_emb, news_set_mask = [], []
    y_dir, y_vol = [], []

    for date in dates:
        qw = get_quant_window(quant_df, quant_pos, date, L)
        if qw is None or date not in targets_by_date.index:
            continue

        # la etiqueta es NaN en las últimas `horizon` sesiones de la serie,
        # porque no hay días futuros suficientes; esas fechas se descartan
        # igual que las que no tienen ventana completa
        row_t = targets_by_date.loc[date]
        d_val, rv_val = row_t[f"d_h{horizon}"], row_t[f"rv_h{horizon}"]
        if pd.isna(d_val) or pd.isna(rv_val):
            continue

        seq_result = get_text_sequence(daily_agg_df, text_pos, date, L)
        if seq_result is None:
            continue
        seq_emb, seq_mask = seq_result
        today_emb, today_has_news = get_text_today(daily_agg_df, text_pos, date)

        kept_dates.append(date)
        quant_seqs.append(qw)
        text_today_emb.append(today_emb)
        text_today_has_news.append(today_has_news)
        text_seq_emb.append(seq_emb)
        text_seq_mask.append(seq_mask)
        y_dir.append(float(d_val))
        y_vol.append(float(rv_val))

        if variable_set is not None:
            ns_emb, ns_mask = get_news_set_today(variable_set, var_set_pos, date)
            news_set_emb.append(ns_emb)
            news_set_mask.append(ns_mask)

    if not kept_dates:
        return None

    result = {
        "date": np.array(kept_dates),
        "quant_seq": np.stack(quant_seqs),
        "text_today_emb": np.stack(text_today_emb),
        "text_today_has_news": np.array(text_today_has_news, dtype=bool),
        "text_seq_emb": np.stack(text_seq_emb),
        "text_seq_mask": np.stack(text_seq_mask),
        "y_direction": np.array(y_dir, dtype="float32"),
        "y_vol": np.array(y_vol, dtype="float32"),
    }
    if variable_set is not None:
        result["news_set_emb"] = np.stack(news_set_emb)
        result["news_set_mask"] = np.stack(news_set_mask)
    return result
