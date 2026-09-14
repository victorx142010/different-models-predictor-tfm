"""Diagnóstico de dos dimensiones internas del modelo que no entran en la
búsqueda de hiperparámetros: `text_proj_dim` y `dense_hidden`, fijadas en 64.

Aquí los datos no cambian, solo el tamaño de una capa interna. Por eso el
dataset se construye una vez y lo que se repite es el entrenamiento, con un
valor distinto en cada pasada y los mismos hiperparámetros.

1. `dense_hidden` en {32, 64, 128}, con price_only: la capa densa final es
   común a las cinco variantes, así que basta con probarla en la más barata.
2. `text_proj_dim` en {32, 64, 128}, con late_fusion: es la variante más barata
   que usa esa proyección (price_only no tiene texto y cross_attention hace su
   propia proyección dentro de la atención).

Se evalúa solo en los folds de desarrollo, nunca en el holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.ingestion.download_prices import TICKERS
from src.training.hpo import precompute_all_data
from src.training.hyperparameters_io import hyperparameters_path_for, load_best_hyperparameters
from src.training.train_final_models import train_and_eval_final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

ABLATION_HORIZON = 5
ABLATION_LOOKBACK = 20
DENSE_HIDDEN_VARIANT = "price_only"
TEXT_PROJ_VARIANT = "late_fusion"
CANDIDATE_SIZES = (32, 64, 128)


def run_architecture_ablation(
    tickers: tuple[str, ...],
    hp_path: Path,
    variant: str,
    param_name: str,
    param_values: tuple[int, ...] = CANDIDATE_SIZES,
    horizon: int = ABLATION_HORIZON,
    lookback: int = ABLATION_LOOKBACK,
) -> pd.DataFrame:
    """Entrena `variant` en los folds de desarrollo de cada activo, una vez
    por cada valor de `param_name` ("text_proj_dim" o "dense_hidden"), con
    los mismos hiperparámetros y el mismo dataset: la única diferencia entre
    pasadas es el tamaño de esa capa."""
    if param_name not in ("text_proj_dim", "dense_hidden"):
        raise ValueError(f"param_name debe ser 'text_proj_dim' o 'dense_hidden', recibido '{param_name}'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ablation de arquitectura en {device}: {variant}, {param_name} en {param_values}, "
          f"h={horizon}, L={lookback}, tickers={tickers}")

    best_hp = load_best_hyperparameters(hp_path)
    params = best_hp[variant]["best_params"]
    print(f"Hiperparámetros de '{variant}' reutilizados de {hp_path.name}: {params}")

    all_data = precompute_all_data(device, tickers=tickers, horizon=horizon, lookback=lookback)

    rows = []
    for value in param_values:
        override = {param_name: value}
        for d in all_data:
            _, metrics = train_and_eval_final(variant, params, d["train"], d["val"], device, **override)
            row = {"ticker": d["ticker"], "fold": d["fold"], "variant": variant, param_name: value, **metrics}
            rows.append(row)
            print(
                f"  [{param_name}={value}] {d['ticker']} {d['fold']}: acc={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f} val_loss={metrics['best_val_loss']:.4f}"
            )

    return pd.DataFrame(rows)


def main() -> None:
    """Ejecuta los dos diagnósticos y guarda los resultados en results/."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a diagnosticar (por defecto: TICKERS)")
    parser.add_argument("--horizon", type=int, default=ABLATION_HORIZON)
    parser.add_argument("--lookback", type=int, default=ABLATION_LOOKBACK)
    parser.add_argument("--hyperparams", type=str, default=None, help="Ruta al JSON de hiperparámetros (por defecto: results/best_hyperparameters_h{H}.json, el del horizonte indicado)")
    args = parser.parse_args()

    tickers = tuple(t.upper() for t in args.tickers)
    tickers_tag = "_".join(t.lower() for t in tickers)
    hp_path = Path(args.hyperparams) if args.hyperparams else hyperparameters_path_for(args.horizon, RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 1. dense_hidden en {CANDIDATE_SIZES} sobre {DENSE_HIDDEN_VARIANT} ===")
    dense_df = run_architecture_ablation(
        tickers, hp_path, DENSE_HIDDEN_VARIANT, "dense_hidden",
        horizon=args.horizon, lookback=args.lookback,
    )
    dense_path = RESULTS_DIR / f"dense_hidden_ablation_{tickers_tag}_h{args.horizon}_L{args.lookback}.csv"
    dense_df.to_csv(dense_path, index=False)
    dense_summary = dense_df.groupby("dense_hidden")[["accuracy", "f1", "rmse_rv", "mae_rv", "best_val_loss"]].median().round(4)
    print("\n=== Resumen dense_hidden (mediana across ticker x fold) ===")
    print(dense_summary.to_string())
    print(f"-> {dense_path}")

    print(f"\n=== 2. text_proj_dim en {CANDIDATE_SIZES} sobre {TEXT_PROJ_VARIANT} ===")
    text_df = run_architecture_ablation(
        tickers, hp_path, TEXT_PROJ_VARIANT, "text_proj_dim",
        horizon=args.horizon, lookback=args.lookback,
    )
    text_path = RESULTS_DIR / f"text_proj_dim_ablation_{tickers_tag}_h{args.horizon}_L{args.lookback}.csv"
    text_df.to_csv(text_path, index=False)
    text_summary = text_df.groupby("text_proj_dim")[["accuracy", "f1", "rmse_rv", "mae_rv", "best_val_loss"]].median().round(4)
    print("\n=== Resumen text_proj_dim (mediana across ticker x fold) ===")
    print(text_summary.to_string())
    print(f"-> {text_path}")


if __name__ == "__main__":
    main()
