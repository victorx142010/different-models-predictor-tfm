"""Evalúa las tres líneas base (camino aleatorio, persistente y ARIMA-GARCH)
sobre el mismo holdout 2022-2023 que las cinco variantes del modelo. `main.py
--baselines` solo las evalúa en los folds de desarrollo.

Reutiliza las funciones de predicción de naive_baselines.py y arima_garch.py,
pasándoles el fold del holdout. Su función `baseline_predictions_aligned` la
usa consolidate_holdout.py para las tablas finales.

Ejecutado como script, además, reentrena las cinco variantes y compara:
- Dirección: test de Pesaran-Timmermann de cada línea base.
- Volatilidad: test de Diebold-Mariano de cada variante frente a cada línea
  base.

Las tablas finales no salen de este script, sino de consolidate_holdout.py y
rebuild_chapter5_tables.py, que parten de los modelos ya guardados.

Salida del script: results/holdout_baseline_metrics_{activos}.csv y
results/holdout_baseline_dm_{activos}.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.baselines.arima_garch import arima_walk_forward, fit_arima_fold, garch_walk_forward_rv
from src.baselines.naive_baselines import naive_predictions, random_walk_predictions
from src.evaluation.financial_metrics import TRADING_DAYS_PER_YEAR, compute_all_financial_metrics, diebold_mariano_test, pesaran_timmermann_test
from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.training.hpo import TENSOR_KEYS_ALL
from src.training.holdout_evaluation import build_holdout_fold, extend_quant_features_to_holdout
from src.training.hyperparameters_io import require_hyperparameters_for
from src.training.sequence_dataset import build_dataset, load_daily_agg, load_variable_set, targets_with_horizon
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import to_tensors
from src.training.train_final_models import train_and_eval_final
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"
L = 20

# horizontes de referencia: con ellos, los ficheros de salida no llevan el
# horizonte en el nombre; con cualquier otro, sí
HORIZONS_REFERENCIA = (5, 20, 60)

BASELINE_NAMES = ("random_walk", "naive_persistente", "arima_garch")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def baseline_predictions_aligned(
    ticker: str, horizon: int, fold, prices_df: pd.DataFrame, targets_df: pd.DataFrame, common_dates: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Calcula las predicciones de las tres líneas base sobre el holdout y
    las alinea por fecha, no por posición, con `common_dates`, las fechas
    que usan las variantes tras el filtrado de `build_dataset`. Así la
    comparación es muestra a muestra."""
    mask_train = (prices_df["date"] >= fold.train_start) & (prices_df["date"] <= fold.train_end)
    mask_val = (prices_df["date"] >= fold.val_start) & (prices_df["date"] <= fold.val_end)
    prices_train = prices_df.loc[mask_train].reset_index(drop=True)
    prices_val = prices_df.loc[mask_val].reset_index(drop=True)

    tmask_train = (targets_df["date"] >= fold.train_start) & (targets_df["date"] <= fold.train_end)
    targets_train = targets_df.loc[tmask_train].reset_index(drop=True)

    all_val_dates = prices_val["date"]

    rw_dir, rw_rv = random_walk_predictions(prices_train, targets_train, all_val_dates, horizon)
    naive_dir, naive_rv = naive_predictions(prices_df, targets_df, all_val_dates, horizon)

    arima_model = fit_arima_fold(prices_train["log_return"].dropna().values)
    point_forecasts, val_resid = arima_walk_forward(arima_model, prices_val["log_return"].values, max_h=horizon)
    train_resid = arima_model.arima_res_.resid
    rv_forecasts = garch_walk_forward_rv(train_resid, val_resid, horizons=(horizon,))
    r_hat_cum = np.cumsum([point_forecasts[k] for k in range(1, horizon + 1)], axis=0)[-1]
    arima_dir = pd.Series((r_hat_cum > 0).astype(float), index=all_val_dates.values)
    arima_rv = pd.Series(rv_forecasts[horizon], index=all_val_dates.values)

    rw_dir = pd.Series(rw_dir.values, index=all_val_dates.values)
    rw_rv = pd.Series(rw_rv.values, index=all_val_dates.values)
    naive_dir = pd.Series(naive_dir.values, index=all_val_dates.values)
    naive_rv = pd.Series(naive_rv.values, index=all_val_dates.values)

    out = {}
    for name, dirs, rvs in [
        ("random_walk", rw_dir, rw_rv),
        ("naive_persistente", naive_dir, naive_rv),
        ("arima_garch", arima_dir, arima_rv),
    ]:
        out[name] = (dirs.reindex(common_dates).values, rvs.reindex(common_dates).values)
    return out


def run(tickers: tuple[str, ...], horizons: tuple[int, ...]) -> None:
    """Evalúa las líneas base y las compara con las cinco variantes para los
    activos y horizontes indicados, y guarda los dos ficheros de resultados."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline_rows = []
    dm_rows = []

    for horizon in horizons:
        # hiperparámetros del propio horizonte; si no existen, falla indicando
        # cómo obtenerlos
        _hp_path, best_hp = require_hyperparameters_for(horizon, RESULTS_DIR, tickers=tickers)
        embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
        fold = build_holdout_fold(embargo)
        print(f"\n=== horizonte h={horizon} ===")

        for ticker in tickers:
            quant_df = extend_quant_features_to_holdout(ticker, fold)
            daily_agg = load_daily_agg(ticker)
            variable_set = load_variable_set(ticker)
            targets = targets_with_horizon(ticker, horizon)
            prices = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet")

            train_dates = prices.loc[(prices["date"] >= fold.train_start) & (prices["date"] <= fold.train_end), "date"]
            val_dates = prices.loc[(prices["date"] >= fold.val_start) & (prices["date"] <= fold.val_end), "date"]

            train_ds = build_dataset(train_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set)
            val_ds = build_dataset(val_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set)
            common_dates = val_ds["date"]

            train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
            train_t = to_tensors(dict(train_ds, quant_seq=train_q), device)
            val_t = to_tensors(dict(val_ds, quant_seq=val_q), device)
            y_dir_true = val_t["y_direction"].cpu().numpy()
            y_vol_true = val_t["y_vol"].cpu().numpy()

            print(f"\n{ticker} (h={horizon}): {len(common_dates)} muestras de holdout comunes")

            # --- líneas base, en las mismas fechas que las variantes ---
            print("  ajustando baselines (random_walk, naive_persistente, ARIMA-GARCH)...")
            baselines = baseline_predictions_aligned(ticker, horizon, fold, prices, targets, common_dates)

            for name, (b_dir, b_rv) in baselines.items():
                mask = ~np.isnan(b_dir) & ~np.isnan(y_dir_true)
                mask_rv = ~np.isnan(b_rv) & ~np.isnan(y_vol_true)
                acc = accuracy_score(y_dir_true[mask], b_dir[mask])
                f1 = f1_score(y_dir_true[mask], b_dir[mask], zero_division=0)
                kappa = cohen_kappa_score(y_dir_true[mask], b_dir[mask])
                pt = pesaran_timmermann_test(b_dir[mask], y_dir_true[mask])
                rmse = _rmse(y_vol_true[mask_rv], b_rv[mask_rv])
                mae = mean_absolute_error(y_vol_true[mask_rv], b_rv[mask_rv])
                baseline_rows.append(
                    {
                        "ticker": ticker, "horizon": horizon, "model": name,
                        "accuracy": acc, "f1": f1, "cohen_kappa": kappa,
                        "pt_statistic": pt["statistic"], "pt_p_value": pt["p_value"],
                        "rmse_rv": rmse, "mae_rv": mae,
                    }
                )
                print(f"    {name}: acc={acc:.3f} kappa={kappa:.3f} pt_p={pt['p_value']} rmse_rv={rmse:.4f}")

            # --- las 5 variantes, entrenadas una vez, en las mismas fechas ---
            variant_vol_errors: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                params = best_hp[variant]["best_params"]
                model, metrics = train_and_eval_final(variant, params, train_t, val_t, device)
                model.eval()
                with torch.no_grad():
                    val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
                    vol_pred = model(val_batch)["vol_pred"].cpu().numpy()
                variant_vol_errors[variant] = vol_pred - y_vol_true
                print(f"    {variant}: acc={metrics['accuracy']:.3f} rmse_rv={metrics['rmse_rv']:.4f}")

            # --- Diebold-Mariano: cada variante frente a cada línea base ---
            for variant in VARIANTS:
                for name, (_, b_rv) in baselines.items():
                    baseline_vol_error = b_rv - y_vol_true
                    valid = ~np.isnan(baseline_vol_error) & ~np.isnan(variant_vol_errors[variant])
                    dm = diebold_mariano_test(
                        variant_vol_errors[variant][valid], baseline_vol_error[valid], h=horizon
                    )
                    dm_rows.append(
                        {
                            "ticker": ticker, "horizon": horizon, "variant": variant, "baseline": name,
                            "dm_statistic": dm["statistic"], "dm_p_value": dm["p_value"],
                            "mean_loss_diff": dm["mean_loss_diff"],
                        }
                    )

    tickers_tag = "_".join(t.lower() for t in tickers)
    # con otros horizontes se añaden al nombre, para no sobrescribir los
    # ficheros de referencia
    if tuple(horizons) != HORIZONS_REFERENCIA:
        tickers_tag += "_" + "_".join(f"h{h}" for h in horizons)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = RESULTS_DIR / f"holdout_baseline_metrics_{tickers_tag}.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"\n-> {baseline_path}")

    dm_df = pd.DataFrame(dm_rows)
    dm_path = RESULTS_DIR / f"holdout_baseline_dm_{tickers_tag}.csv"
    dm_df.to_csv(dm_path, index=False)
    print(f"-> {dm_path}")


def main() -> None:
    """Punto de entrada: activos y horizontes por línea de comandos (por
    defecto, los cuatro activos y los horizontes 5, 20 y 60)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    args = parser.parse_args()
    run(tuple(t.upper() for t in args.tickers), tuple(args.horizons))


if __name__ == "__main__":
    main()
