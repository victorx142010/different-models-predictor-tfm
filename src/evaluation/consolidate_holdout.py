"""Vuelve a predecir el holdout con los modelos ya entrenados y guarda las
predicciones sesión a sesión. Es la fuente única de la que salen todas las
tablas y figuras de resultados.

Volver a entrenar no reproduce exactamente los mismos modelos en h=20 y h=60,
así que calcular unas tablas con un entrenamiento y otras con otro haría que no
cuadraran entre sí. Por eso aquí no se entrena nada: se cargan los checkpoints
que guardó recalibrate_holdout_worker.py
(models/{variante}_{activo}_holdout_h{H}_L20.pt), se hace una única pasada de
inferencia y se guarda lo que predice cada modelo en cada sesión. Después,
rebuild_chapter5_tables.py calcula todas las métricas y contrastes a partir de
estas predicciones.

Las tres líneas base no dependen del entrenamiento, pero se recalculan
igualmente, alineadas por fecha con las mismas sesiones que las variantes.

Salida:
    results/consolidado_predicciones_variantes.csv
        una fila por activo, horizonte, variante y sesión: probabilidad
        de dirección, volatilidad predicha, etiquetas y retorno a h sesiones
    results/consolidado_predicciones_baselines.csv
        lo mismo para las tres líneas base
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.holdout_baseline_significance import baseline_predictions_aligned
from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START
from src.training.holdout_evaluation import build_holdout_fold, extend_quant_features_to_holdout
from src.training.hpo import TENSOR_KEYS_ALL, build_model
from src.training.sequence_dataset import (
    build_dataset,
    load_daily_agg,
    load_variable_set,
    targets_with_horizon,
)
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import to_tensors
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

TICKERS = ("SPY", "QQQ", "KO", "GS")
HORIZONS = (5, 20, 60)
L = 20


def main() -> None:
    """Carga los checkpoints del holdout, predice con cada uno, calcula las
    predicciones de las líneas base en las mismas fechas y guarda los dos
    ficheros."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    var_rows: list[pd.DataFrame] = []
    base_rows: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
        fold = build_holdout_fold(embargo)
        print(f"\n=== h={horizon} | holdout {fold.val_start.date()} -> {fold.val_end.date()} ===")

        for ticker in TICKERS:
            quant_df = extend_quant_features_to_holdout(ticker, fold)
            daily_agg = load_daily_agg(ticker)
            variable_set = load_variable_set(ticker)
            targets = targets_with_horizon(ticker, horizon)
            prices = pd.read_parquet(
                DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
            )

            train_dates = prices.loc[
                (prices["date"] >= fold.train_start) & (prices["date"] <= fold.train_end), "date"
            ]
            val_dates = prices.loc[
                (prices["date"] >= fold.val_start) & (prices["date"] <= fold.val_end), "date"
            ]

            train_ds = build_dataset(train_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set)
            val_ds = build_dataset(val_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set)
            common_dates = val_ds["date"]

            train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
            val_t = to_tensors(dict(val_ds, quant_seq=val_q), device)

            y_dir_true = val_t["y_direction"].cpu().numpy()
            y_vol_true = val_t["y_vol"].cpu().numpy()
            quant_input_dim = val_t["quant_seq"].shape[-1]

            # el retorno a h sesiones se calcula igual que en
            # holdout_evaluation.main, porque es el que convierte la dirección
            # en el retorno de la estrategia; si difiriera, Sharpe y Sortino no
            # serían comparables
            price_series = prices.set_index("date")["adj_close"]
            fwd_ret = np.log(price_series.shift(-horizon) / price_series).reindex(common_dates).to_numpy()

            print(f"  {ticker}: {len(common_dates)} sesiones de holdout")

            # aquí no se entrena: se cargan los pesos que guardó
            # recalibrate_holdout_worker.py y solo se predice, para que las
            # predicciones sean exactamente las de aquella ejecución
            for variant in VARIANTS:
                ckpt_path = MODELS_DIR / f"{variant}_{ticker}_holdout_h{horizon}_L{L}.pt"
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = build_model(variant, ckpt["params"], quant_input_dim=quant_input_dim).to(device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                with torch.no_grad():
                    batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
                    out = model(batch)
                    proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
                    vol_pred = out["vol_pred"].cpu().numpy()

                var_rows.append(
                    pd.DataFrame(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "variant": variant,
                            "date": common_dates,
                            "proba_dir": proba,
                            "y_dir": y_dir_true,
                            "vol_pred": vol_pred,
                            "y_vol": y_vol_true,
                            "fwd_return": fwd_ret,
                        }
                    )
                )

            # las líneas base se calculan sobre `common_dates`, las fechas que
            # quedan tras el filtrado de build_dataset; si no, Diebold-Mariano
            # compararía series de distinta longitud
            print("    ajustando baselines (ARIMA-GARCH incluido)...")
            baselines = baseline_predictions_aligned(ticker, horizon, fold, prices, targets, common_dates)
            for name, (b_dir, b_rv) in baselines.items():
                base_rows.append(
                    pd.DataFrame(
                        {
                            "ticker": ticker,
                            "horizon": horizon,
                            "model": name,
                            "date": common_dates,
                            "dir_pred": b_dir,
                            "vol_pred": b_rv,
                            "y_dir": y_dir_true,
                            "y_vol": y_vol_true,
                            "fwd_return": fwd_ret,
                        }
                    )
                )

    var_df = pd.concat(var_rows, ignore_index=True)
    base_df = pd.concat(base_rows, ignore_index=True)

    out_v = RESULTS_DIR / "consolidado_predicciones_variantes.csv"
    out_b = RESULTS_DIR / "consolidado_predicciones_baselines.csv"
    var_df.to_csv(out_v, index=False)
    base_df.to_csv(out_b, index=False)
    print(f"\n-> {out_v} ({len(var_df)} filas)")
    print(f"-> {out_b} ({len(base_df)} filas)")


if __name__ == "__main__":
    main()
