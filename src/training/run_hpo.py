"""Lanza la búsqueda de hiperparámetros de las cinco variantes, una detrás de
otra, sin pasar por main.py.

Hace lo mismo que `main.py --hpo`, pero además guarda el detalle de todos los
trials de cada variante (results/hpo_trials_{variante}_{activos}.csv). Usa 20
trials por variante; el `MedianPruner` corta los que van claramente peor que la
mediana, así que el coste real es menor que 20 entrenamientos completos.

Los mejores hiperparámetros se guardan en
results/best_hyperparameters_h{H}.json, o en la ruta de `--hyperparams`, con un
bloque `_meta` que registra los activos, el horizonte y la ventana. Si el
fichero ya existe, pide confirmación antes de reemplazarlo.

Uso:
    python -m src.training.run_hpo SPY QQQ KO GS --horizon 5
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import TICKERS
from src.training.hpo import HORIZON, L, precompute_all_data, run_study
from src.training.hyperparameters_io import hyperparameters_path_for, save_best_hyperparameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

N_TRIALS = 20


def main() -> None:
    """Busca los hiperparámetros de las cinco variantes para los activos y
    el horizonte indicados y guarda los mejores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a incluir en la búsqueda (por defecto: TICKERS)")
    parser.add_argument("--horizon", type=int, default=HORIZON, help=f"Horizonte de la búsqueda, en sesiones (por defecto: {HORIZON})")
    parser.add_argument("--hyperparams", type=str, default=None, help="Ruta de salida (por defecto: results/best_hyperparameters_h{HORIZONTE}.json)")
    args = parser.parse_args()
    tickers = tuple(args.tickers)
    out_path = Path(args.hyperparams) if args.hyperparams else hyperparameters_path_for(args.horizon, RESULTS_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"HPO en {device}, {N_TRIALS} trials/variante, tickers={tickers}")

    all_data = precompute_all_data(device, tickers=tickers, horizon=args.horizon)
    print(f"Datos precomputados: {len(all_data)} combinaciones ticker x fold")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    best_params_all = {}

    for variant in VARIANTS:
        print(f"\n=== HPO: {variant} ===")
        t0 = time.time()
        study = run_study(variant, all_data, device, N_TRIALS)
        t1 = time.time()

        trials_df = study.trials_dataframe()
        trials_path = RESULTS_DIR / f"hpo_trials_{variant}_{tickers_tag}.csv"
        trials_df.to_csv(trials_path, index=False)

        best_params_all[variant] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
            "n_complete": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            "time_seconds": round(t1 - t0, 1),
        }
        print(
            f"{variant}: mejor pérdida val={study.best_value:.4f} "
            f"params={study.best_params} "
            f"({best_params_all[variant]['n_complete']} completos, "
            f"{best_params_all[variant]['n_pruned']} podados, "
            f"{t1 - t0:.0f}s)"
        )

    meta = {
        "tickers": list(tickers),
        "horizon": args.horizon,
        "lookback": L,
        "garch": "symmetric",
        "layernorm": False,
        "n_trials": N_TRIALS,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_best_hyperparameters(out_path, best_params_all, meta)


if __name__ == "__main__":
    main()
