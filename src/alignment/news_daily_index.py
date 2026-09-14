"""Resumen diario de la cobertura de noticias de cada activo: convierte la
lista de noticias en una serie alineada con el calendario de precios.

El índice cubre todas las sesiones de 2010 a 2023, no solo las que tienen
noticias. Así, los periodos sin cobertura en FNSPID quedan marcados de forma
explícita con `has_news=False`, en lugar de desaparecer y poder confundirse con
un error en los datos.

Columnas por sesión:
- `n_news`: número de noticias asignadas a la sesión (ver align_news.py).
- `n_news_log1p`: log(1 + n_news). La mayoría de días tiene pocas noticias o
  ninguna y algunos tienen muchas; el logaritmo suaviza esa cola. La
  estandarización se hace después, dentro de cada fold.
- `has_news`: si la sesión tiene alguna noticia. Decide si el modelo usa el
  embedding real o el vector nulo aprendido.
- `dias_desde_ultima_noticia`: sesiones transcurridas desde la última noticia
  (0 si hay noticia ese día). Queda en NaN hasta que aparece la primera noticia
  del activo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"


def build_daily_news_index(ticker: str) -> pd.DataFrame:
    """Convierte las noticias de un activo, ya asignadas a su sesión, en una
    fila por sesión con el número de noticias y las columnas derivadas."""
    news = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet")
    prices = pd.read_parquet(
        DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    )

    sessions = prices[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

    counts = (
        news.dropna(subset=["session_date"])
        .groupby("session_date")
        .size()
        .rename("n_news")
    )

    daily = sessions.merge(counts, left_on="date", right_index=True, how="left")
    daily["n_news"] = daily["n_news"].fillna(0).astype(int)
    daily["has_news"] = daily["n_news"] > 0
    daily["n_news_log1p"] = np.log1p(daily["n_news"])

    # Para cada fila, posición de la sesión con la última noticia: se marca -1
    # donde no hay noticia, se convierte en NaN y se rellena hacia delante con
    # la última posición válida. Restándola a la posición actual se obtienen
    # las sesiones transcurridas desde la última noticia.
    news_session_pos = np.where(daily["has_news"].values, np.arange(len(daily)), -1)
    last_news_pos = pd.Series(news_session_pos, dtype="float64").replace(-1, np.nan).ffill()
    daily["dias_desde_ultima_noticia"] = np.arange(len(daily)) - last_news_pos.values

    daily["ticker"] = ticker
    return daily[
        ["date", "ticker", "n_news", "has_news", "n_news_log1p", "dias_desde_ultima_noticia"]
    ]


def main() -> None:
    """Construye y guarda el índice diario de noticias de los activos
    indicados (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a procesar (por defecto: TICKERS)")
    args = parser.parse_args()

    for ticker in args.tickers:
        daily = build_daily_news_index(ticker)
        out_path = DATA_INTERIM_DIR / f"{ticker}_news_daily_index.parquet"
        daily.to_parquet(out_path, index=False)

        n_with_news = int(daily["has_news"].sum())
        first_news_date = daily.loc[daily["has_news"], "date"].min()
        print(
            f"{ticker}: {len(daily)} sesiones totales, {n_with_news} con noticia "
            f"({n_with_news / len(daily):.1%}), primera noticia en {first_news_date.date()} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
