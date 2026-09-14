"""Descarga los precios diarios (OHLCV) de los activos.

La fuente principal es yfinance. Si falla, por ejemplo por un límite de
peticiones o un corte de red, se reintenta con stooq a través de
pandas_datareader.

Aunque el estudio empieza en 2015, se descarga desde 2010 para disponer del
tramo de calentamiento del GARCH y de las primeras ventanas de la LSTM.

`TICKERS` es solo el valor por defecto: este y los demás scripts aceptan
cualquier lista de activos por línea de comandos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TICKERS = ["SPY", "QQQ", "KO", "GS"]
DOWNLOAD_START = "2010-01-01"
DOWNLOAD_END = "2023-12-31"

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Descarga con yfinance y renombra las columnas al esquema del
    proyecto."""
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        raise ValueError(f"yfinance devolvió vacío para {ticker}")

    df = df.copy()
    df.columns = df.columns.get_level_values(0)  # yfinance devuelve columnas con dos niveles; se quita el del ticker
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df.index.name = "date"
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def _download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fuente alternativa cuando yfinance falla. stooq ya devuelve el precio
    ajustado como `close`, así que `adj_close` se copia de esa columna."""
    import pandas_datareader.data as web

    df = web.DataReader(ticker, "stooq", start=start, end=end)
    df = df.sort_index()  # stooq devuelve orden descendente
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    # stooq no distingue adj_close de close (ya viene ajustado por defecto)
    df["adj_close"] = df["close"]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def download_ticker(
    ticker: str, start: str = DOWNLOAD_START, end: str = DOWNLOAD_END
) -> pd.DataFrame:
    """Descarga el OHLCV de un activo con yfinance y, si falla, con stooq.
    La columna `source` indica qué fuente se usó."""
    try:
        df = _download_yfinance(ticker, start, end)
        df["source"] = "yfinance"
    except Exception as exc:  # noqa: BLE001 - fallback deliberado ante cualquier fallo de yfinance
        print(f"  [WARN] yfinance falló para {ticker} ({exc}); probando stooq...")
        df = _download_stooq(ticker, start, end)
        df["source"] = "stooq"

    df = df.reset_index()
    df.insert(1, "ticker", ticker)
    return df


def main() -> None:
    """Descarga y guarda en data/raw los precios sin procesar de los activos
    indicados (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a descargar (por defecto: TICKERS)")
    args = parser.parse_args()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        print(f"Descargando {ticker} ({DOWNLOAD_START} a {DOWNLOAD_END})...")
        df = download_ticker(ticker)
        out_path = DATA_RAW_DIR / f"{ticker}_prices_raw_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        df.to_parquet(out_path, index=False)
        print(
            f"  {len(df)} filas, {df['date'].min().date()} a {df['date'].max().date()}, "
            f"fuente={df['source'].iloc[0]} -> {out_path}"
        )


if __name__ == "__main__":
    main()
