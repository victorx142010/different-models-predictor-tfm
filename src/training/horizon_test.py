"""Diagnóstico exploratorio: ¿mejora la accuracy de dirección de los modelos al
predecir a más largo plazo, como ocurre con las líneas base?

Entrena las cinco variantes en los folds de desarrollo a una semana (h=5) y a
un mes (h=21 sesiones), con los hiperparámetros de h=5 en los dos casos, para
que solo cambie el horizonte. Para h=21 el embargo se amplía a 21 sesiones: con
el de 5, las últimas etiquetas de entrenamiento usarían precios del periodo de
validación. Las etiquetas de h=21 se calculan en memoria, sin modificar el
fichero de etiquetas guardado.

Si existe results/final_models_metrics_{activos}.parquet (h=1, de
train_final_models.py), lo añade para comparar los tres horizontes en una misma
tabla.

Salida: results/horizon_diagnostic_metrics_{activos}.parquet y, si se combina
con h=1, horizon_diagnostic_combined_{activos}.parquet y
horizon_diagnostic_summary_{activos}.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.targets.targets import compute_targets
from src.training.sequence_dataset import (
    build_dataset,
    build_quant_feature_table,
    load_daily_agg,
    load_variable_set,
)
from src.training.hpo import L
from src.training.hyperparameters_io import load_best_hyperparameters, require_hyperparameters_for
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import to_tensors
from src.training.train_final_models import train_and_eval_final
from src.validation.splitter import WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

# horizonte cuyos hiperparámetros se usan en todos los horizontes comparados:
# así solo cambia el horizonte y no el modelo
BASE_HORIZON = 5

# nombre -> (horizonte en sesiones, sesiones de embargo)
HORIZONS_TO_TEST = {
    "1_semana_h5": (5, 5),
    "1_mes_h21": (21, 21),
}


def build_targets_with_horizon(ticker: str, horizon: int) -> pd.DataFrame:
    """Carga las etiquetas guardadas (h=1, 3 y 5) y, si el horizonte pedido
    no está entre ellas, lo calcula en memoria sin escribir nada en disco."""
    canonical = pd.read_parquet(
        DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    )
    if f"d_h{horizon}" in canonical.columns:
        return canonical

    prices = pd.read_parquet(
        DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    )
    extra = compute_targets(prices, horizons=(horizon,))
    return canonical.merge(extra, on=["date", "ticker"], how="left")


def precompute_data_for_horizon(
    horizon: int, embargo_sessions: int, device: str, tickers: tuple[str, ...] = tuple(TICKERS)
) -> list[dict]:
    """Construye los pares de entrenamiento y validación de cada activo y
    fold para un horizonte y un embargo concretos. El embargo es un
    parámetro porque con horizontes largos hay que ampliarlo."""
    splitter = WalkForwardSplitter(embargo_sessions=embargo_sessions)
    all_data = []
    for ticker in tickers:
        quant_df = build_quant_feature_table(ticker)
        daily_agg = load_daily_agg(ticker)
        variable_set = load_variable_set(ticker)
        targets = build_targets_with_horizon(ticker, horizon)
        prices = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )

        for fold in splitter.folds:
            train_dates = splitter.slice_train(prices, fold)["date"]
            val_dates = splitter.slice_val(prices, fold)["date"]
            train_ds = build_dataset(
                train_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set
            )
            val_ds = build_dataset(
                val_dates, quant_df, daily_agg, targets, L, horizon, variable_set=variable_set
            )
            if train_ds is None or val_ds is None:
                print(f"  [WARN] sin muestras para {ticker} {fold.name} h={horizon}")
                continue

            train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
            train_ds = dict(train_ds, quant_seq=train_q)
            val_ds = dict(val_ds, quant_seq=val_q)

            all_data.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "train": to_tensors(train_ds, device),
                    "val": to_tensors(val_ds, device),
                }
            )
    return all_data


def main() -> None:
    """Entrena las cinco variantes en los horizontes de HORIZONS_TO_TEST con
    los mismos hiperparámetros y guarda las métricas. Si existen las de h=1
    para los mismos activos, genera además la tabla comparativa."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a probar (por defecto: TICKERS)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Prueba de horizonte en {device}")

    # los mismos hiperparámetros para todos los horizontes, a propósito (ver
    # BASE_HORIZON)
    _hp_path, best_hp = require_hyperparameters_for(BASE_HORIZON, RESULTS_DIR)
    print(f"Hiperparámetros (reutilizados en todos los horizontes): {_hp_path.name}")

    all_rows = []
    for label, (horizon, embargo) in HORIZONS_TO_TEST.items():
        print(f"\n=== {label}: h={horizon} sesiones, embargo={embargo} ===")
        all_data = precompute_data_for_horizon(horizon, embargo, device, tuple(args.tickers))

        for variant in VARIANTS:
            params = best_hp[variant]["best_params"]
            for d in all_data:
                _, metrics = train_and_eval_final(variant, params, d["train"], d["val"], device)
                row = {
                    "horizon_label": label,
                    "horizon": horizon,
                    "ticker": d["ticker"],
                    "fold": d["fold"],
                    "variant": variant,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"  {d['ticker']} {d['fold']} {variant}: acc={metrics['accuracy']:.3f} "
                    f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f}"
                )

    tickers_tag = "_".join(t.lower() for t in args.tickers)
    new_df = pd.DataFrame(all_rows)
    new_path = RESULTS_DIR / f"horizon_diagnostic_metrics_{tickers_tag}.parquet"
    new_df.to_parquet(new_path, index=False)
    print(f"\n{len(new_df)} filas -> {new_path}")

    # se añade h=1, calculado con train_final_models.py, para comparar los tres
    # horizontes en una tabla. Solo si existe el fichero de estos mismos
    # activos: como lleva los activos en el nombre, basta con comprobar que
    # existe
    h1_path = RESULTS_DIR / f"final_models_metrics_{tickers_tag}.parquet"
    if not h1_path.exists():
        print(f"\n{h1_path.name} no existe: no se combina, resultado guardado solo en {new_path.name}.")
        return
    h1_df = pd.read_parquet(h1_path)
    h1_df = h1_df.assign(horizon_label="dia_siguiente_h1", horizon=1)

    combined = pd.concat([h1_df, new_df], ignore_index=True)
    combined_path = RESULTS_DIR / f"horizon_diagnostic_combined_{tickers_tag}.parquet"
    combined.to_parquet(combined_path, index=False)

    summary = (
        combined.groupby(["horizon", "horizon_label", "variant"])[["accuracy", "f1", "rmse_rv", "mae_rv"]]
        .median()
        .round(4)
        .sort_index()
    )
    print("\n=== Resumen (mediana across ticker x fold) ===")
    print(summary.to_string())
    summary_path = RESULTS_DIR / f"horizon_diagnostic_summary_{tickers_tag}.csv"
    summary.to_csv(summary_path)
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
