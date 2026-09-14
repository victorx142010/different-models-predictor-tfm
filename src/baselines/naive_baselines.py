"""Dos líneas base muy simples, sin parámetros que aprender: si el modelo no
las supera, no aporta nada.

- Camino aleatorio con deriva: una predicción constante por fold, calculada
  solo con el tramo de entrenamiento. La dirección es el signo de h · r_medio,
  donde r_medio es el retorno medio de entrenamiento: si es positivo, como
  ocurre en la evaluación sobre el holdout, predice siempre «sube». La
  volatilidad es la media de la volatilidad realizada de entrenamiento.
  Representa la idea de que, sin información del día, lo mejor es repetir la
  media histórica.
- Persistente (naïve): cambia cada día, pero tampoco estima nada. La dirección
  es el signo del último retorno observado y la volatilidad, la realizada en el
  periodo anterior equivalente. Representa la idea de que mañana se parecerá a
  hoy.

Este script las evalúa en los cuatro folds de desarrollo. La evaluación sobre
el holdout está en holdout_baseline_significance.py, que reutiliza estas
funciones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.targets.targets import HORIZONS
from src.validation.splitter import Fold, WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def random_walk_predictions(
    prices_train: pd.DataFrame, targets_train: pd.DataFrame, val_dates: pd.Series, h: int
) -> tuple[pd.Series, pd.Series]:
    """Predicciones del camino aleatorio con deriva: la misma dirección y la
    misma volatilidad para todas las fechas de validación, calculadas solo
    con el entrenamiento del fold."""
    r_bar = prices_train["log_return"].mean()
    r_hat = h * r_bar
    direction_pred = pd.Series(1.0 if r_hat > 0 else 0.0, index=val_dates.index)

    rv_bar = targets_train[f"rv_h{h}"].mean()
    rv_pred = pd.Series(rv_bar, index=val_dates.index)
    return direction_pred, rv_pred


def naive_predictions(
    prices_full: pd.DataFrame, targets_full: pd.DataFrame, val_dates: pd.Series, h: int
) -> tuple[pd.Series, pd.Series]:
    """Predicciones persistentes: en cada fecha de validación repite lo
    último conocido, el signo del retorno de ese día para la dirección y la
    volatilidad realizada del periodo anterior equivalente."""
    # dirección: signo del último retorno observado en t, la fecha de la fila
    price_by_date = prices_full.set_index("date")["log_return"].sort_index()
    r_t = price_by_date.reindex(val_dates.values)
    direction_pred = pd.Series((r_t > 0).astype(float).values, index=val_dates.index)

    # volatilidad: la del periodo equivalente anterior, RV_{t-h,h}. Como el
    # calendario de sesiones no tiene huecos, desplazar h posiciones equivale a
    # retroceder h sesiones.
    rv_sorted = targets_full.sort_values("date").set_index("date")[f"rv_h{h}"]
    rv_lagged = rv_sorted.shift(h)
    rv_pred = rv_lagged.reindex(val_dates.values)
    rv_pred.index = val_dates.index
    return direction_pred, rv_pred


def evaluate_fold(
    ticker: str,
    fold: Fold,
    prices_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    splitter: WalkForwardSplitter,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict]:
    """Calcula las predicciones de las dos líneas base en un fold para cada
    horizonte y devuelve las métricas (accuracy, f1, rmse_rv y mae_rv) en
    formato largo: una fila por activo, fold, horizonte, modelo y métrica.
    `targets_df` debe incluir las columnas d_h{h} y rv_h{h} de cada
    horizonte (ver `targets_with_horizon` en sequence_dataset.py)."""
    prices_train = splitter.slice_train(prices_df, fold)
    targets_train = splitter.slice_train(targets_df, fold)
    targets_val = splitter.slice_val(targets_df, fold)
    val_dates = targets_val["date"]

    rows = []
    for h in horizons:
        y_dir = targets_val[f"d_h{h}"].values
        y_rv = targets_val[f"rv_h{h}"].values

        rw_dir, rw_rv = random_walk_predictions(prices_train, targets_train, val_dates, h)
        naive_dir, naive_rv = naive_predictions(prices_df, targets_df, val_dates, h)

        for model_name, dir_pred, rv_pred in [
            ("random_walk", rw_dir, rw_rv),
            ("naive_persistente", naive_dir, naive_rv),
        ]:
            mask = ~np.isnan(y_dir) & ~dir_pred.isna().values
            rows.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "horizon": h,
                    "model": model_name,
                    "metric": "accuracy",
                    "value": accuracy_score(y_dir[mask], dir_pred.values[mask]),
                }
            )
            rows.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "horizon": h,
                    "model": model_name,
                    "metric": "f1",
                    "value": f1_score(y_dir[mask], dir_pred.values[mask], zero_division=0),
                }
            )

            mask_rv = ~np.isnan(y_rv) & ~rv_pred.isna().values
            rows.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "horizon": h,
                    "model": model_name,
                    "metric": "rmse_rv",
                    "value": _rmse(y_rv[mask_rv], rv_pred.values[mask_rv]),
                }
            )
            rows.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "horizon": h,
                    "model": model_name,
                    "metric": "mae_rv",
                    "value": mean_absolute_error(y_rv[mask_rv], rv_pred.values[mask_rv]),
                }
            )
    return rows


def main() -> pd.DataFrame:
    """Evalúa las dos líneas base en los folds de desarrollo de los activos
    indicados (por defecto, `TICKERS`) y guarda las métricas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a evaluar (por defecto: TICKERS)")
    args = parser.parse_args()

    splitter = WalkForwardSplitter()
    all_rows: list[dict] = []

    for ticker in args.tickers:
        prices_df = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        targets_df = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        for fold in splitter.folds:
            all_rows.extend(evaluate_fold(ticker, fold, prices_df, targets_df, splitter))

    metrics_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_path = RESULTS_DIR / f"baselines_simples_metrics_{tickers_tag}.parquet"
    metrics_df.to_parquet(out_path, index=False)
    print(f"{len(metrics_df)} filas de métricas -> {out_path}")

    summary = (
        metrics_df.groupby(["model", "horizon", "metric"])["value"].mean().unstack("metric").round(4)
    )
    print("\nPromedio across tickers/folds:")
    print(summary.to_string())
    return metrics_df


if __name__ == "__main__":
    main()
