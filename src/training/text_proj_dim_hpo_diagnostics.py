"""Repite la búsqueda de hiperparámetros con `text_proj_dim=32`.

architecture_hparam_diagnostics.py probaba `text_proj_dim=32` con los
hiperparámetros encontrados para 64, lo que puede perjudicar a la dimensión
nueva. Aquí se hace una búsqueda propia (Optuna, 20 trials) con
`text_proj_dim=32` para las tres variantes que usan esa proyección
(late_fusion, early_fusion y text_only) y los tres horizontes (5, 20 y 60).
price_only no tiene rama de texto, y cross_attention resuelve su propia
proyección dentro de la atención.

Para cada variante y horizonte guarda los hiperparámetros encontrados
(results/best_hyperparameters_textproj32_h{H}.json, sin tocar los de 64) y las
métricas por activo y fold al reentrenar con ellos, para compararlas con las
obtenidas con hiperparámetros reutilizados.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.ingestion.download_prices import TICKERS
from src.training.hpo import L, precompute_all_data, run_study
from src.training.hyperparameters_io import hyperparameters_path_for, save_best_hyperparameters
from src.training.train_final_models import train_and_eval_final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

VARIANTS_WITH_TEXT_PROJ = ("late_fusion", "early_fusion", "text_only")
HORIZONS = (5, 20, 60)
TEXT_PROJ_DIM = 32
N_TRIALS = 20


def run_one(
    tickers: tuple[str, ...], horizon: int, device: str
) -> tuple[dict, pd.DataFrame]:
    """Búsqueda de hiperparámetros y métricas finales de las tres variantes
    en un horizonte. Devuelve (best_params_all, metrics_df)."""
    print(f"\n{'='*70}\nHorizonte h={horizon}\n{'='*70}")
    all_data = precompute_all_data(device, tickers=tickers, horizon=horizon, lookback=L)
    print(f"Datos precomputados: {len(all_data)} combinaciones ticker x fold")

    best_params_all: dict = {}
    rows = []

    for variant in VARIANTS_WITH_TEXT_PROJ:
        print(f"\n--- HPO ({variant}, h={horizon}, text_proj_dim={TEXT_PROJ_DIM}) ---")
        t0 = time.time()
        study = run_study(variant, all_data, device, N_TRIALS, text_proj_dim=TEXT_PROJ_DIM)
        t1 = time.time()

        trials_path = RESULTS_DIR / f"hpo_trials_{variant}_textproj32_h{horizon}.csv"
        study.trials_dataframe().to_csv(trials_path, index=False)

        best_params_all[variant] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
            "n_complete": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            "time_seconds": round(t1 - t0, 1),
        }
        print(
            f"{variant}: mejor pérdida val={study.best_value:.4f} params={study.best_params} "
            f"({best_params_all[variant]['n_complete']} completos, "
            f"{best_params_all[variant]['n_pruned']} podados, {t1 - t0:.0f}s)"
        )

        for d in all_data:
            _, metrics = train_and_eval_final(
                variant, study.best_params, d["train"], d["val"], device, text_proj_dim=TEXT_PROJ_DIM
            )
            rows.append(
                {
                    "ticker": d["ticker"],
                    "fold": d["fold"],
                    "variant": variant,
                    "horizon": horizon,
                    "text_proj_dim": TEXT_PROJ_DIM,
                    "hiperparametros": "hpopropio",
                    **metrics,
                }
            )

    return best_params_all, pd.DataFrame(rows)


def main() -> None:
    """Ejecuta la búsqueda y el cálculo de métricas en los tres horizontes y
    guarda los resultados en results/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS))
    args = parser.parse_args()
    tickers = tuple(t.upper() for t in args.tickers)
    tickers_tag = "_".join(t.lower() for t in tickers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"text_proj_dim={TEXT_PROJ_DIM} HPO propio en {device}, {N_TRIALS} trials/variante, "
          f"variantes={VARIANTS_WITH_TEXT_PROJ}, horizontes={HORIZONS}, tickers={tickers}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for horizon in HORIZONS:
        best_params_all, metrics_df = run_one(tickers, horizon, device)
        all_rows.append(metrics_df)

        out_hp_path = hyperparameters_path_for(horizon, RESULTS_DIR, tag="_textproj32")
        meta = {
            "tickers": list(tickers),
            "horizon": horizon,
            "lookback": L,
            "garch": "symmetric",
            "layernorm": False,
            "text_proj_dim": TEXT_PROJ_DIM,
            "n_trials": N_TRIALS,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nota": "HPO propio (no reutilizado) para verificar el hallazgo de architecture_hparam_diagnostics.py; solo cubre las 3 variantes con rama de texto que usan text_proj_dim.",
        }
        save_best_hyperparameters(out_hp_path, best_params_all, meta)
        print(f"-> {out_hp_path}")

    full_df = pd.concat(all_rows, ignore_index=True)
    full_path = RESULTS_DIR / f"text_proj_dim32_hpopropio_{tickers_tag}_L{L}.csv"
    full_df.to_csv(full_path, index=False)
    print(f"\n-> {full_path}")

    summary = full_df.groupby(["variant", "horizon"])[["accuracy", "f1", "rmse_rv", "mae_rv"]].median().round(4)
    summary_path = RESULTS_DIR / f"text_proj_dim32_hpopropio_summary_{tickers_tag}_L{L}.csv"
    summary.to_csv(summary_path)
    print(f"-> {summary_path}")
    print("\n=== Resumen (mediana por variante x horizonte) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
