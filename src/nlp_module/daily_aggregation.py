"""Resume las noticias de cada sesión en un único vector, para las variantes
que usan un embedding por día: text_only, early_fusion y late_fusion.

Cuando una sesión tiene varias noticias, cada una se pondera por 1 - p_neu, la
probabilidad de que no sea neutra. Así, las noticias con un sentimiento marcado
pesan más que las neutras. Si todas son neutras, la suma de pesos es
prácticamente cero y se usa una media simple.

En las sesiones sin noticias (`has_news=False`), el embedding y las columnas de
sentimiento se dejan vacíos (None o NaN) a propósito, sin rellenar con ceros.
El vector que representa la ausencia de noticias lo aprende el propio modelo
(`NullEmbedding` en base_model.py): decidirlo aquí mezclaría una decisión de
arquitectura con el preprocesado.

Salida: features/nlp/{ticker}_news_daily_agg_simple.parquet, con las columnas
date, ticker, n_news, has_news, p_pos, p_neg, p_neu, net_sentiment y embedding
(vector de 768 o None).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"


def _weighted_agg_group(g: pd.DataFrame) -> dict:
    """Media del embedding y de las probabilidades de las noticias de una
    sesión, ponderando cada noticia por 1 - p_neu."""
    w = (1 - g["p_neu"]).to_numpy()
    if w.sum() <= 1e-9:
        w = np.ones(len(g))
    w = w / w.sum()

    emb_stack = np.stack(g["embedding"].to_numpy())
    emb_avg = (w[:, None] * emb_stack).sum(axis=0).astype("float32")

    return {
        "embedding": emb_avg,
        "p_pos": float(np.average(g["p_pos"], weights=w)),
        "p_neg": float(np.average(g["p_neg"], weights=w)),
        "p_neu": float(np.average(g["p_neu"], weights=w)),
    }


def build_simple_daily_aggregation(ticker: str) -> pd.DataFrame:
    """Agrupa los embeddings de un activo por sesión y los resume con
    `_weighted_agg_group`. Las sesiones sin noticias se completan con
    valores vacíos, usando el índice diario como referencia de las sesiones
    existentes."""
    emb_df = pd.read_parquet(FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet")
    emb_df = emb_df.dropna(subset=["session_date"])

    rows = []
    for session_date, g in emb_df.groupby("session_date"):
        agg = _weighted_agg_group(g)
        agg["date"] = session_date
        rows.append(agg)
    agg_df = pd.DataFrame(rows)
    agg_df["net_sentiment"] = agg_df["p_pos"] - agg_df["p_neg"]

    daily_index = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_news_daily_index.parquet")
    full = daily_index[["date", "ticker", "n_news", "has_news"]].merge(
        agg_df, on="date", how="left"
    )

    no_news = ~full["has_news"]
    full.loc[no_news, ["p_pos", "p_neg", "p_neu", "net_sentiment"]] = np.nan
    full.loc[no_news, "embedding"] = None

    return full[
        ["date", "ticker", "n_news", "has_news", "p_pos", "p_neg", "p_neu", "net_sentiment", "embedding"]
    ]


def main() -> None:
    """Construye y guarda la agregación diaria de los activos indicados (por
    defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a agregar (por defecto: TICKERS)")
    args = parser.parse_args()

    FEATURES_NLP_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        print(f"Agregando noticias por sesión para {ticker}...")
        df = build_simple_daily_aggregation(ticker)
        out_path = FEATURES_NLP_DIR / f"{ticker}_news_daily_agg_simple.parquet"
        df.to_parquet(out_path, index=False)
        n_with_news = int(df["has_news"].sum())
        print(f"  {len(df)} sesiones ({n_with_news} con noticia agregada) -> {out_path}")


if __name__ == "__main__":
    main()
