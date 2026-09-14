"""Variables objetivo del modelo: la dirección del precio y la volatilidad
realizada.

Para cada sesión t y cada horizonte h:

    d_{t,h}  = 1 si ln(P_{t+h} / P_t) > 0, y 0 en caso contrario
    rv_{t,h} = sqrt( suma de r_{t+k}^2 para k = 1..h )

donde P es el precio de cierre ajustado y r el retorno logarítmico diario. La
volatilidad realizada solo suma retornos posteriores a t (de t+1 a t+h): el
retorno del propio día t ya es una entrada del modelo, e incluirlo en la
etiqueta sería una fuga de información.

Las últimas h filas de cada serie quedan en NaN, porque no hay sesiones futuras
con las que calcular su etiqueta; se descartan al construir los datasets.

Este script guarda los horizontes 1, 3 y 5. Los de 20 y 60 sesiones se calculan
en memoria con `compute_targets` cuando hacen falta (ver `targets_with_horizon`
en sequence_dataset.py).

Salida: data/interim/{ticker}_targets_{inicio}_{fin}.parquet, con las columnas
date, ticker, d_h1, d_h3, d_h5, rv_h1, rv_h3 y rv_h5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

HORIZONS = (1, 3, 5)


def compute_targets(df: pd.DataFrame, horizons: tuple[int, ...] = HORIZONS) -> pd.DataFrame:
    """A partir de los precios limpios de un activo (columnas date,
    adj_close y log_return), calcula la dirección y la volatilidad realizada
    para cada horizonte de `horizons`."""
    df = df.sort_values("date").reset_index(drop=True)
    price = df["adj_close"]
    log_ret = df["log_return"]

    out = {"date": df["date"], "ticker": df["ticker"]}
    for h in horizons:
        r_fwd = np.log(price.shift(-h) / price)
        direction = np.where(r_fwd.isna(), np.nan, (r_fwd > 0).astype(float))
        out[f"d_h{h}"] = direction

        sq_sum = sum((log_ret.shift(-k) ** 2 for k in range(1, h + 1)), start=pd.Series(0.0, index=df.index))
        out[f"rv_h{h}"] = np.sqrt(sq_sum)

    return pd.DataFrame(out)


def main() -> None:
    """Calcula y guarda las etiquetas de los activos indicados por línea de
    comandos (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a procesar (por defecto: TICKERS)")
    args = parser.parse_args()

    for ticker in args.tickers:
        clean_path = (
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        clean_df = pd.read_parquet(clean_path)

        targets_df = compute_targets(clean_df)
        out_path = (
            DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        targets_df.to_parquet(out_path, index=False)

        valid_h1 = targets_df["d_h1"].notna().sum()
        pct_up_h1 = targets_df["d_h1"].mean()
        print(
            f"{ticker}: {len(targets_df)} filas, {valid_h1} con d_h1 válido "
            f"(% subida h=1: {pct_up_h1:.3f}) -> {out_path}"
        )


if __name__ == "__main__":
    main()
