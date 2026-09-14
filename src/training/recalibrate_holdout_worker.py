"""Proceso que entrena y evalúa una variante sobre el holdout, lanzado por
recalibrate_holdout_launcher.py.

Para los 4 activos y los 3 horizontes, entrena la variante sobre 2015-2021 con
sus hiperparámetros (igual que holdout_evaluation.py), la evalúa una vez sobre
2022-2023 y calcula también las métricas con el umbral calibrado en desarrollo
(compute_dev_thresholds.py), heredando la tasa de clasificación positiva.

Guarda el checkpoint de cada combinación en
models/{variante}_{activo}_holdout_h{H}_L20.pt. A partir de esos checkpoints,
consolidate_holdout.py genera las predicciones de las que salen todas las
tablas de resultados.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

# Fuerza algoritmos deterministas en la GPU: sin esto, algunas operaciones,
# sobre todo las de cross_attention, no reproducen exactamente el mismo
# resultado entre dos entrenamientos con la misma semilla. warn_only=True avisa
# en lugar de fallar si alguna operación no tiene versión determinista.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

from src.ingestion.download_prices import TICKERS
from src.training.hyperparameters_io import hyperparameters_path_for
from src.training.holdout_evaluation import main as holdout_main

HORIZONS = (5, 20, 60)

# cada horizonte usa los hiperparámetros de su propia búsqueda, porque el
# óptimo cambia con el horizonte; se indican de forma explícita
HP_PATHS = {h: hyperparameters_path_for(h) for h in HORIZONS}


def main() -> None:
    """Evalúa la variante indicada con `--variant` en los tres horizontes y
    guarda el resultado en la ruta de `--out`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    thr_df = pd.read_csv("results/dev_calibrated_thresholds.csv")

    all_rows = []
    for horizon in HORIZONS:
        thr_h = thr_df[thr_df["horizon"] == horizon]
        # (tasa heredada de desarrollo, umbral crudo para el calentamiento)
        dev_rates = {
            (r["variant"], r["ticker"]): (r["tasa_positiva_calibrada"], r["umbral_calibrado"])
            for _, r in thr_h.iterrows()
            if r["variant"] == args.variant
        }
        df = holdout_main(
            tickers=tuple(TICKERS),
            horizon=horizon,
            hp_path=HP_PATHS[horizon],
            variants=(args.variant,),
            dev_thresholds=dev_rates,
            save_checkpoints=True,
        )
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True)
    full.to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(full)} filas)", flush=True)


if __name__ == "__main__":
    main()
