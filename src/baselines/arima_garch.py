"""Línea base ARIMA-GARCH, la referencia clásica de la econometría financiera:
si el modelo propuesto no la supera, difícilmente justifica su complejidad.

Cómo se construye:
- El orden (p, d, q) del ARIMA lo elige `pmdarima.auto_arima` sobre el retorno
  logarítmico, sin componente estacional. Se reestima en cada fold solo con su
  entrenamiento.
- La volatilidad de los residuos del ARIMA se modela con un GARCH(1,1),
  eligiendo entre distribución Normal y t de Student por AIC, como en
  garch_features.py.
- En validación se avanza sesión a sesión sin reestimar los parámetros: cada
  nuevo retorno observado solo actualiza el estado del modelo (filtrado).
- Dirección: signo de la suma de los retornos previstos para las h sesiones
  siguientes, coherente con la etiqueta, que es un retorno acumulado.
- Volatilidad: raíz de la suma de las varianzas previstas por el GARCH para
  esas h sesiones.

Este script la evalúa en los folds de desarrollo; su evaluación sobre el
holdout está en holdout_baseline_significance.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from arch import arch_model
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.quant_module.garch_features import RETURN_SCALE, fit_best_garch
from src.targets.targets import HORIZONS
from src.validation.splitter import Fold, WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

MAX_H = max(HORIZONS)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_arima_fold(train_returns: np.ndarray) -> pm.ARIMA:
    """Ajusta un ARIMA a los retornos de entrenamiento de un fold, con el
    orden (p, d, q) elegido automáticamente por `auto_arima`."""
    return pm.auto_arima(
        train_returns,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )


def arima_walk_forward(
    arima_model: pm.ARIMA, val_returns: np.ndarray, max_h: int = MAX_H
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Recorre la validación sesión a sesión: en cada t pronostica hasta
    `max_h` pasos con los parámetros fijos del fold y después incorpora el
    retorno real de t sin reestimar (`refit=False`). Devuelve los
    pronósticos por horizonte y el residuo a un paso, que alimenta el GARCH."""
    res = arima_model.arima_res_
    point_forecasts = {h: [] for h in range(1, max_h + 1)}
    for r_t in val_returns:
        fc = res.get_forecast(steps=max_h).predicted_mean
        for h in range(1, max_h + 1):
            point_forecasts[h].append(fc[h - 1])
        res = res.append([r_t], refit=False)

    one_step = np.array(point_forecasts[1])
    val_resid = val_returns - one_step
    return {h: np.array(v) for h, v in point_forecasts.items()}, val_resid


def garch_walk_forward_rv(
    train_resid: np.ndarray, val_resid: np.ndarray, horizons: tuple[int, ...] = HORIZONS
) -> dict[int, np.ndarray]:
    """Ajusta un GARCH(1,1) a los residuos de entrenamiento del ARIMA y, con
    esos parámetros fijos, pronostica la varianza sobre la validación.
    Devuelve, para cada horizonte, la raíz de la suma de las varianzas
    previstas."""
    train_series = pd.Series(train_resid) * RETURN_SCALE
    train_res, dist = fit_best_garch(train_series)

    full_series = pd.Series(np.concatenate([train_resid, val_resid])) * RETURN_SCALE
    am_full = arch_model(full_series, mean="Zero", vol="GARCH", p=1, q=1, dist=dist)
    fixed = am_full.fix(train_res.params)

    fc = fixed.forecast(horizon=max(horizons), start=len(train_resid), reindex=False)
    var = fc.variance  # columnas h.1 ... h.{max_h}
    # arch rellena el número de paso con ceros a la izquierda según el ancho de
    # max_h (con horizonte 20 son «h.01» ... «h.20»), así que el ancho se
    # calcula del mismo modo
    width = len(str(max(horizons)))

    rv_forecasts = {}
    for h in horizons:
        cols = [f"h.{k:0{width}d}" for k in range(1, h + 1)]
        rv_forecasts[h] = np.sqrt(var[cols].sum(axis=1).values) / RETURN_SCALE
    return rv_forecasts


def evaluate_fold(
    ticker: str,
    fold: Fold,
    prices_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    splitter: WalkForwardSplitter,
    horizons: tuple[int, ...] = HORIZONS,
) -> list[dict]:
    """Ajusta ARIMA y GARCH con el entrenamiento de un fold, genera los
    pronósticos de validación y calcula las métricas para cada horizonte.
    `targets_df` debe incluir d_h{h} y rv_h{h}; los pronósticos se extienden
    hasta `max(horizons)` pasos."""
    prices_train = splitter.slice_train(prices_df, fold)
    prices_val = splitter.slice_val(prices_df, fold)
    targets_val = splitter.slice_val(targets_df, fold)

    train_returns = prices_train["log_return"].dropna().values
    val_returns = prices_val["log_return"].values

    arima_model = fit_arima_fold(train_returns)
    point_forecasts, val_resid = arima_walk_forward(arima_model, val_returns, max_h=max(horizons))

    train_resid = arima_model.arima_res_.resid
    rv_forecasts = garch_walk_forward_rv(train_resid, val_resid, horizons=horizons)

    rows = []
    for h in horizons:
        r_hat_cum = np.cumsum([point_forecasts[k] for k in range(1, h + 1)], axis=0)[-1]
        dir_pred = (r_hat_cum > 0).astype(float)
        rv_pred = rv_forecasts[h]

        y_dir = targets_val[f"d_h{h}"].values
        y_rv = targets_val[f"rv_h{h}"].values

        mask = ~np.isnan(y_dir)
        mask_rv = ~np.isnan(y_rv)

        rows.append(
            {
                "ticker": ticker, "fold": fold.name, "horizon": h, "model": "arima_garch",
                "metric": "accuracy", "value": accuracy_score(y_dir[mask], dir_pred[mask]),
            }
        )
        rows.append(
            {
                "ticker": ticker, "fold": fold.name, "horizon": h, "model": "arima_garch",
                "metric": "f1", "value": f1_score(y_dir[mask], dir_pred[mask], zero_division=0),
            }
        )
        rows.append(
            {
                "ticker": ticker, "fold": fold.name, "horizon": h, "model": "arima_garch",
                "metric": "rmse_rv", "value": _rmse(y_rv[mask_rv], rv_pred[mask_rv]),
            }
        )
        rows.append(
            {
                "ticker": ticker, "fold": fold.name, "horizon": h, "model": "arima_garch",
                "metric": "mae_rv", "value": mean_absolute_error(y_rv[mask_rv], rv_pred[mask_rv]),
            }
        )
    return rows


def main() -> pd.DataFrame:
    """Evalúa ARIMA-GARCH en los folds de desarrollo de los activos
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
            print(f"  ARIMA-GARCH {ticker} {fold.name}...")
            all_rows.extend(evaluate_fold(ticker, fold, prices_df, targets_df, splitter))

    metrics_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_path = RESULTS_DIR / f"arima_garch_metrics_{tickers_tag}.parquet"
    metrics_df.to_parquet(out_path, index=False)
    print(f"\n{len(metrics_df)} filas de métricas -> {out_path}")

    summary = metrics_df.groupby(["horizon", "metric"])["value"].mean().unstack("metric").round(4)
    print("\nPromedio ARIMA-GARCH across tickers/folds:")
    print(summary.to_string())
    return metrics_df


if __name__ == "__main__":
    main()
