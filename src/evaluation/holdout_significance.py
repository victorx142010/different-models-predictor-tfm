"""Contrastes de significancia sobre el holdout 2022-2023 entre las propias
variantes: Pesaran-Timmermann de cada variante y Diebold-Mariano de tres pares
(cross_attention, early_fusion y text_only frente a price_only).

Entrena cada variante una vez por activo y horizonte, como
holdout_evaluation.py, y guarda sus errores de volatilidad para compararlos con
Diebold-Mariano sin volver a entrenar. Como reentrena, sus cifras pueden
diferir un poco de las de consolidate_holdout.py y rebuild_chapter5_tables.py,
que parten de los modelos ya guardados y son las que dan las tablas finales.

Salida:
    results/holdout_significance_metrics_{activos}.csv
        una fila por activo, horizonte y variante: accuracy, F1, RMSE, MAE
        y las métricas de financial_metrics.py
    results/holdout_significance_dm_{activos}.csv
        una fila por activo, horizonte y par de variantes: estadístico y
        p-valor de Diebold-Mariano sobre el error de volatilidad al cuadrado
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.financial_metrics import TRADING_DAYS_PER_YEAR, compute_all_financial_metrics, diebold_mariano_test
from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.training.hpo import TENSOR_KEYS_ALL
from src.training.holdout_evaluation import build_holdout_fold, extend_quant_features_to_holdout
from src.training.hyperparameters_io import hyperparameters_path_for, load_best_hyperparameters
from src.training.sequence_dataset import build_dataset, load_daily_agg, load_variable_set, targets_with_horizon
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import to_tensors
from src.training.train_final_models import train_and_eval_final
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

L = 20
# pares para Diebold-Mariano: cross_attention y dos variantes con texto frente
# a price_only, para ver si su diferencia de RMSE en volatilidad es
# estadísticamente significativa
DM_PAIRS = [("cross_attention", "price_only"), ("early_fusion", "price_only"), ("text_only", "price_only")]


def run(tickers: tuple[str, ...], horizons: tuple[int, ...], hp_paths: dict[int, Path]) -> None:
    """Entrena las cinco variantes sobre el holdout para cada activo y
    horizonte, calcula sus métricas y los contrastes de Diebold-Mariano, y
    guarda los dos ficheros."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric_rows = []
    dm_rows = []

    for horizon in horizons:
        hp_path = hp_paths[horizon]
        best_hp = load_best_hyperparameters(hp_path)
        embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
        fold = build_holdout_fold(embargo)
        print(f"\n=== horizonte h={horizon} (hiperparámetros: {hp_path.name}) ===")
        print(f"train {fold.train_start.date()} -> {fold.train_end.date()} | holdout {fold.val_start.date()} -> {fold.val_end.date()}")

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

            train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
            train_t = to_tensors(dict(train_ds, quant_seq=train_q), device)
            val_t = to_tensors(dict(val_ds, quant_seq=val_q), device)

            price_series = prices.set_index("date")["adj_close"]
            fwd_log_return = np.log(price_series.shift(-horizon) / price_series)
            actual_fwd_return = fwd_log_return.reindex(val_ds["date"]).to_numpy()
            y_vol_true = val_t["y_vol"].cpu().numpy()

            vol_errors_by_variant: dict[str, np.ndarray] = {}

            print(f"\n{ticker} (h={horizon}): {len(train_ds['date'])} train / {len(val_ds['date'])} holdout")
            for variant in VARIANTS:
                params = best_hp[variant]["best_params"]
                model, metrics = train_and_eval_final(variant, params, train_t, val_t, device)

                model.eval()
                with torch.no_grad():
                    val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
                    out = model(val_batch)
                    dir_pred_proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
                    vol_pred = out["vol_pred"].cpu().numpy()
                y_dir_true = val_t["y_direction"].cpu().numpy()

                fin_metrics = compute_all_financial_metrics(
                    dir_pred_proba, y_dir_true, actual_fwd_return, periods_per_year=TRADING_DAYS_PER_YEAR / horizon
                )
                metric_rows.append({"ticker": ticker, "horizon": horizon, "variant": variant, **metrics, **fin_metrics})
                vol_errors_by_variant[variant] = vol_pred - y_vol_true

                print(
                    f"  {variant}: acc={metrics['accuracy']:.3f} rmse_rv={metrics['rmse_rv']:.4f} "
                    f"pt_p={fin_metrics['pt_p_value']:.4f}"
                )

            for variant_a, variant_b in DM_PAIRS:
                dm = diebold_mariano_test(vol_errors_by_variant[variant_a], vol_errors_by_variant[variant_b], h=horizon)
                dm_rows.append(
                    {
                        "ticker": ticker, "horizon": horizon,
                        "variant_a": variant_a, "variant_b": variant_b,
                        "dm_statistic": dm["statistic"], "dm_p_value": dm["p_value"],
                        "mean_loss_diff": dm["mean_loss_diff"],
                    }
                )
                sig = "significativo (p<0.05)" if dm["p_value"] < 0.05 else "no significativo"
                print(f"    DM {variant_a} vs {variant_b}: stat={dm['statistic']:.3f} p={dm['p_value']:.4f} ({sig})")

    tickers_tag = "_".join(t.lower() for t in tickers)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = RESULTS_DIR / f"holdout_significance_metrics_{tickers_tag}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n-> {metrics_path}")

    dm_df = pd.DataFrame(dm_rows)
    dm_path = RESULTS_DIR / f"holdout_significance_dm_{tickers_tag}.csv"
    dm_df.to_csv(dm_path, index=False)
    print(f"-> {dm_path}")


def main() -> None:
    """Punto de entrada: activos y horizontes por línea de comandos (por
    defecto, los cuatro activos y los horizontes 5, 20 y 60)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20, 60])
    args = parser.parse_args()

    tickers = tuple(t.upper() for t in args.tickers)
    hp_paths = {h: hyperparameters_path_for(h, RESULTS_DIR) for h in args.horizons}
    run(tickers, tuple(args.horizons), hp_paths)


if __name__ == "__main__":
    main()
