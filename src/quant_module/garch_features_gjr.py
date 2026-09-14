"""Variables GARCH calculadas con GJR-GARCH(1,1,1), la versión asimétrica, y
guardadas aparte de las simétricas.

El GJR-GARCH añade un término que permite que una caída suba la volatilidad más
que una subida del mismo tamaño (efecto apalancamiento). El script reutiliza
las funciones de garch_features.py con `o=1` y solo cambia el nombre de los
ficheros, para no sobrescribir las variables simétricas, que son las que usan
los resultados principales. Dentro del pipeline, `main.py --garch gjr` hace lo
mismo.

Salida:
    features/quant/{ticker}_garch_features_gjr_2015-01-02_2021-12-31.parquet
    features/quant/{ticker}_garch_features_gjr_warmup.parquet

Uso:
    python -m src.quant_module.garch_features_gjr
    python -m src.quant_module.garch_features_gjr AAPL MSFT
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.quant_module.garch_features import compute_garch_features, compute_warmup_segment
from src.validation.splitter import WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_QUANT_DIR = PROJECT_ROOT / "features" / "quant"


def compute_and_save_gjr_features(ticker: str, splitter: WalkForwardSplitter) -> tuple[Path, Path]:
    """Calcula y guarda las variables GJR-GARCH de un activo (por fold y de
    calentamiento) y devuelve las rutas de los dos ficheros."""
    clean_path = DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    prices_df = pd.read_parquet(clean_path)

    span_start = splitter.folds[0].train_start.strftime("%Y-%m-%d")
    span_end = splitter.folds[-1].val_end.strftime("%Y-%m-%d")

    features_df = compute_garch_features(prices_df, splitter, o=1)
    garch_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features_gjr_{span_start}_{span_end}.parquet"
    features_df.to_parquet(garch_path, index=False)

    warmup_df = compute_warmup_segment(prices_df, splitter.folds[0].train_start, o=1)
    warmup_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features_gjr_warmup.parquet"
    warmup_df.to_parquet(warmup_path, index=False)

    return garch_path, warmup_path


def main() -> None:
    """Calcula y guarda las variables GJR-GARCH de los activos indicados
    (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a procesar (por defecto: TICKERS)")
    args = parser.parse_args()

    FEATURES_QUANT_DIR.mkdir(parents=True, exist_ok=True)
    splitter = WalkForwardSplitter()

    for ticker in args.tickers:
        print(f"{ticker}: ajustando GJR-GARCH(1,1,1) por fold...")
        garch_path, warmup_path = compute_and_save_gjr_features(ticker, splitter)
        features_df = pd.read_parquet(garch_path)
        by_fold = features_df.groupby("fold").agg(
            n=("date", "size"), dist=("garch_dist", "first"),
            alpha1=("garch_alpha1", "first"), beta1=("garch_beta1", "first"),
            gamma1=("garch_gamma1", "first"),
        )
        print(f"  {len(features_df)} filas -> {garch_path.name}")
        print(by_fold.to_string())
        print(f"  -> {warmup_path.name}\n")


if __name__ == "__main__":
    main()
