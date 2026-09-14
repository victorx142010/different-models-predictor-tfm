"""Volatilidad condicional GARCH(1,1) y residuo estandarizado de cada sesión:
dos de las tres entradas de la rama cuantitativa.

El GARCH no se ajusta una sola vez sobre toda la serie, sino dentro de cada
fold y solo con sus datos de entrenamiento. Ajustado con toda la serie, los
parámetros usados para 2018 contendrían información de 2021, lo que sería una
fuga de información.

Para no repetir fechas, la serie 2015-2021 se construye así:

- El fold 1 aporta su entrenamiento (2015-2017) y su validación (2018).
- Los folds 2, 3 y 4 solo aportan su año de validación (2019, 2020 y 2021),
  calculado con los parámetros reestimados con todo el histórico hasta ese
  punto.

En cada fold, los parámetros se estiman con el entrenamiento y después se
aplican, fijos, sobre la ventana continua entrenamiento + embargo + validación,
para que la recursión de la varianza no tenga saltos. Las filas del embargo se
descartan.

Para las innovaciones se prueban la distribución Normal y la t de Student, y se
elige la de menor AIC.

El tramo de calentamiento (2010-2014) se calcula con `compute_warmup_segment`,
y el holdout se completa en holdout_evaluation.py.

Salida: features/quant/{ticker}_garch_features_2015-01-02_2021-12-31.parquet,
con las columnas date, ticker, cond_vol, std_resid, garch_omega, garch_alpha1,
garch_beta1, garch_dist y fold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.validation.splitter import Fold, WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_QUANT_DIR = PROJECT_ROOT / "features" / "quant"

RETURN_SCALE = 100  # retornos en %: evita problemas numéricos al ajustar el GARCH


def fit_best_garch(returns_pct: pd.Series, o: int = 0) -> tuple:
    """Ajusta un GARCH(1,1) con innovaciones Normal y t de Student y se
    queda con el de menor AIC. Devuelve el modelo ajustado y el nombre de la
    distribución elegida.

    Con `o=1` ajusta un GJR-GARCH, que añade un término asimétrico: una
    caída sube la volatilidad futura más que una subida del mismo tamaño
    (efecto apalancamiento)."""
    candidates = []
    for dist in ("normal", "t"):
        am = arch_model(returns_pct, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist)
        res = am.fit(disp="off")
        candidates.append((res.aic, dist, res))
    candidates.sort(key=lambda x: x[0])
    _, best_dist, best_res = candidates[0]
    return best_res, best_dist


def _forecast_vol_column(fixed_res, horizon: int) -> "np.ndarray":
    """Pronóstico GARCH de volatilidad acumulada a `horizon` sesiones en
    cada fecha: la raíz de la suma de las varianzas previstas a 1, 2, ...,
    `horizon` pasos. En cada fecha solo usa información hasta ese día, así
    que no introduce fugas. Solo lo usa el diagnóstico de
    feature_engineering_diagnostics.py."""
    fc = fixed_res.forecast(horizon=horizon, start=0, reindex=False)
    width = len(str(horizon))
    cols = [f"h.{k:0{width}d}" for k in range(1, horizon + 1)]
    return np.sqrt(fc.variance[cols].sum(axis=1).values) / RETURN_SCALE


def _garch_segment(
    prices_df: pd.DataFrame,
    fold: Fold,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
    o: int = 0,
    forecast_horizon: int | None = None,
) -> pd.DataFrame:
    """Calcula las variables GARCH de un fold.

    Estima los parámetros solo con el entrenamiento del fold y los aplica,
    sin reestimar, sobre la ventana continua [window_start, window_end], que
    incluye el embargo y la validación. Así se reproduce lo que ocurriría en
    producción, donde no se puede reajustar el modelo con datos que aún no
    existen. Devuelve solo las filas de [output_start, output_end], sin las
    del embargo.

    Si se indica `forecast_horizon`, añade la columna `garch_fcst_vol` (ver
    `_forecast_vol_column`)."""
    train_mask = (prices_df["date"] >= fold.train_start) & (prices_df["date"] <= fold.train_end)
    train_returns = prices_df.loc[train_mask, "log_return"].dropna() * RETURN_SCALE

    train_res, dist = fit_best_garch(train_returns, o=o)

    window_mask = (prices_df["date"] >= window_start) & (prices_df["date"] <= window_end)
    window_df = prices_df.loc[window_mask].dropna(subset=["log_return"]).reset_index(drop=True)
    window_returns = window_df["log_return"] * RETURN_SCALE

    # `.fix()` en lugar de `.fit()`: aplica la recursión con los parámetros ya
    # estimados en el entrenamiento, sin volver a optimizar. `o` debe coincidir
    # con el del ajuste para que los parámetros encajen.
    am_full = arch_model(window_returns, mean="Zero", vol="GARCH", p=1, o=o, q=1, dist=dist)
    fixed_res = am_full.fix(train_res.params)

    out_dict = {
        "date": window_df["date"].values,
        "ticker": window_df["ticker"].values,
        "cond_vol": fixed_res.conditional_volatility.values / RETURN_SCALE,
        "std_resid": fixed_res.std_resid.values,
        "garch_omega": train_res.params["omega"],
        "garch_alpha1": train_res.params["alpha[1]"],
        "garch_beta1": train_res.params["beta[1]"],
        "garch_dist": dist,
        "fold": fold.name,
    }
    if o != 0:
        out_dict["garch_gamma1"] = train_res.params["gamma[1]"]
    if forecast_horizon is not None:
        out_dict["garch_fcst_vol"] = _forecast_vol_column(fixed_res, forecast_horizon)
    out = pd.DataFrame(out_dict)
    in_output_range = (out["date"] >= output_start) & (out["date"] <= output_end)
    in_embargo = (out["date"] >= fold.embargo_start) & (out["date"] <= fold.embargo_end)
    return out.loc[in_output_range & ~in_embargo].reset_index(drop=True)


def compute_warmup_segment(
    prices_df: pd.DataFrame, warmup_end: pd.Timestamp, o: int = 0, forecast_horizon: int | None = None
) -> pd.DataFrame:
    """Calcula las variables GARCH del tramo de calentamiento: todo lo
    anterior a `warmup_end`, que es el inicio del primer fold (2015-01-02).

    La LSTM necesita las L sesiones anteriores a cada predicción, así que
    las primeras sesiones de 2015 necesitan variables GARCH de finales de
    2014. Por eso se descargan precios desde 2010. Este tramo nunca se usa
    como etiqueta ni se evalúa, y como es anterior a todos los folds basta
    con un único ajuste, sin ir fold a fold."""
    mask = prices_df["date"] < warmup_end
    warmup_returns = prices_df.loc[mask, "log_return"].dropna() * RETURN_SCALE
    if len(warmup_returns) < 30:
        raise ValueError(
            f"Histórico insuficiente antes de {warmup_end.date()} para el calentamiento GARCH "
            f"({len(warmup_returns)} obs.)"
        )

    res, dist = fit_best_garch(warmup_returns, o=o)
    warmup_df = prices_df.loc[mask].dropna(subset=["log_return"]).reset_index(drop=True)

    out_dict = {
        "date": warmup_df["date"].values,
        "ticker": warmup_df["ticker"].values,
        "cond_vol": res.conditional_volatility.values / RETURN_SCALE,
        "std_resid": res.std_resid.values,
        "garch_omega": res.params["omega"],
        "garch_alpha1": res.params["alpha[1]"],
        "garch_beta1": res.params["beta[1]"],
        "garch_dist": dist,
        "fold": "warmup",
    }
    if o != 0:
        out_dict["garch_gamma1"] = res.params["gamma[1]"]
    if forecast_horizon is not None:
        out_dict["garch_fcst_vol"] = _forecast_vol_column(res, forecast_horizon)
    return pd.DataFrame(out_dict)


def compute_garch_features(
    prices_df: pd.DataFrame, splitter: WalkForwardSplitter, o: int = 0, forecast_horizon: int | None = None
) -> pd.DataFrame:
    """Recorre los folds y concatena el tramo que aporta cada uno (ver el
    docstring del módulo) hasta cubrir 2015-2021 sin huecos ni fechas
    repetidas. Con `o=1` calcula GJR-GARCH."""
    segments = []
    for i, fold in enumerate(splitter.folds):
        if i == 0:
            # el fold 1 aporta su entrenamiento (2015-2017) y su validación
            # (2018)
            output_start = fold.train_start
        else:
            # los folds 2-4 solo aportan su año de validación: su entrenamiento
            # ya lo cubre el fold anterior
            output_start = fold.val_start
        segments.append(
            _garch_segment(
                prices_df,
                fold,
                window_start=fold.train_start,
                window_end=fold.val_end,
                output_start=output_start,
                output_end=fold.val_end,
                o=o,
                forecast_horizon=forecast_horizon,
            )
        )
    result = pd.concat(segments, ignore_index=True).sort_values("date").reset_index(drop=True)
    assert result["date"].is_unique, "fechas duplicadas entre folds: revisar límites de segmentos"
    return result


def main() -> None:
    """Calcula y guarda las variables GARCH y el tramo de calentamiento de
    los activos indicados (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a procesar (por defecto: TICKERS)")
    args = parser.parse_args()

    FEATURES_QUANT_DIR.mkdir(parents=True, exist_ok=True)
    splitter = WalkForwardSplitter()
    span_start = splitter.folds[0].train_start.strftime("%Y-%m-%d")
    span_end = splitter.folds[-1].val_end.strftime("%Y-%m-%d")

    for ticker in args.tickers:
        clean_path = (
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        prices_df = pd.read_parquet(clean_path)

        features_df = compute_garch_features(prices_df, splitter)
        out_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features_{span_start}_{span_end}.parquet"
        features_df.to_parquet(out_path, index=False)

        by_fold = features_df.groupby("fold").agg(
            n=("date", "size"),
            dist=("garch_dist", "first"),
            alpha1=("garch_alpha1", "first"),
            beta1=("garch_beta1", "first"),
        )
        print(f"{ticker}: {len(features_df)} filas ({features_df['date'].min().date()} a "
              f"{features_df['date'].max().date()}) -> {out_path}")
        print(by_fold.to_string())

        warmup_df = compute_warmup_segment(prices_df, splitter.folds[0].train_start)
        warmup_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features_warmup.parquet"
        warmup_df.to_parquet(warmup_path, index=False)
        print(
            f"  warmup: {len(warmup_df)} filas ({warmup_df['date'].min().date()} a "
            f"{warmup_df['date'].max().date()}) -> {warmup_path}"
        )


if __name__ == "__main__":
    main()
