"""Junta en una sola tabla las métricas de desarrollo de las tres líneas base
(camino aleatorio, persistente y ARIMA-GARCH), que naive_baselines.py y
arima_garch.py guardan por separado al ejecutarse como scripts. Hay que
ejecutar antes esos dos scripts con los mismos activos.

Salida (en results/):
    all_baselines_metrics_{activos}.parquet      todas las filas
    baselines_summary_{activos}.csv              media por modelo y horizonte
    baselines_summary_by_ticker_{activos}.csv    media por activo y modelo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingestion.download_prices import TICKERS

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def main() -> pd.DataFrame:
    """Junta las métricas de las tres líneas base y guarda la tabla completa
    y los dos resúmenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Activos que se juntan (por defecto: TICKERS); deben ser los mismos que se usaron con naive_baselines.py y arima_garch.py")
    args = parser.parse_args()
    tickers_tag = "_".join(t.lower() for t in args.tickers)

    simple = pd.read_parquet(RESULTS_DIR / f"baselines_simples_metrics_{tickers_tag}.parquet")
    arima_garch = pd.read_parquet(RESULTS_DIR / f"arima_garch_metrics_{tickers_tag}.parquet")

    all_metrics = pd.concat([simple, arima_garch], ignore_index=True)
    out_path = RESULTS_DIR / f"all_baselines_metrics_{tickers_tag}.parquet"
    all_metrics.to_parquet(out_path, index=False)

    # media de todos los activos y folds, por modelo y horizonte
    summary = (
        all_metrics.groupby(["model", "horizon", "metric"])["value"]
        .mean()
        .unstack("metric")[["accuracy", "f1", "rmse_rv", "mae_rv"]]
        .round(4)
        .sort_index()
    )
    summary_path = RESULTS_DIR / f"baselines_summary_{tickers_tag}.csv"
    summary.to_csv(summary_path)

    print(f"{len(all_metrics)} filas consolidadas -> {out_path}")
    print(f"Resumen (promedio ticker x fold) -> {summary_path}\n")
    print(summary.to_string())

    # media por activo, para ver si alguna línea base se comporta distinto en
    # un activo concreto
    by_ticker = (
        all_metrics.groupby(["ticker", "model", "metric"])["value"]
        .mean()
        .unstack("metric")[["accuracy", "f1", "rmse_rv", "mae_rv"]]
        .round(4)
        .sort_index()
    )
    by_ticker_path = RESULTS_DIR / f"baselines_summary_by_ticker_{tickers_tag}.csv"
    by_ticker.to_csv(by_ticker_path)
    print(f"\nDesglose por ticker (promedio de horizontes/folds) -> {by_ticker_path}")
    print(by_ticker.to_string())

    return all_metrics


if __name__ == "__main__":
    main()
