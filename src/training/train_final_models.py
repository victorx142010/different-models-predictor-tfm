"""Entrenamiento de un modelo con sus hiperparámetros ya elegidos
(`train_and_eval_final`), que usan main.py, la evaluación sobre el holdout y
los diagnósticos.

La búsqueda de hiperparámetros (hpo.py) entrena con poco margen para poder
probar muchas configuraciones: 60 épocas y paciencia 8. Aquí se da más margen
para que cada modelo converja: hasta 100 épocas y paciencia 10.

Como script, reentrena las cinco variantes en los folds de desarrollo al
horizonte por defecto de hpo.py (h=1) y guarda un checkpoint por combinación y
una tabla con la mediana de las métricas de cada variante.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.evaluation.financial_metrics import classification_agreement_metrics

from src.fusion_model.base_model import VARIANTS, MultiModalModel, UncertaintyWeightedLoss, multitask_loss
from src.ingestion.download_prices import TICKERS
from src.training.hpo import HORIZON, LAMBDA_RATIOS, TENSOR_KEYS_ALL, build_model, precompute_all_data
from src.training.hyperparameters_io import load_best_hyperparameters, require_hyperparameters_for
from src.training.train_control_variants import GRAD_CLIP_NORM, WEIGHT_DECAY, _rmse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64
SEED = 42


def train_and_eval_final(
    variant: str,
    params: dict,
    train_t: dict,
    val_t: dict,
    device: str,
    use_embedding_layernorm: bool = False,
    quant_input_dim: int | None = None,
    text_proj_dim: int | None = None,
    dense_hidden: int | None = None,
) -> tuple[MultiModalModel, dict]:
    """Entrena una variante con sus hiperparámetros en un activo y fold. Usa
    el mismo bucle que la búsqueda de hiperparámetros (`train_and_eval_fold`
    en hpo.py), con parada temprana y pesos de pérdida fijos o aprendidos,
    pero con más épocas y devolviendo también el modelo, para poder guardar
    el checkpoint.

    `quant_input_dim` se deduce de los datos si no se indica, y
    `text_proj_dim` y `dense_hidden` valen 64 por defecto. Así la misma
    función sirve para los diagnósticos de variables
    (feature_engineering_diagnostics.py) y de dimensiones
    (architecture_hparam_diagnostics.py)."""
    torch.manual_seed(SEED)
    if quant_input_dim is None:
        quant_input_dim = train_t["quant_seq"].shape[-1]
    model = build_model(variant, params, use_embedding_layernorm, quant_input_dim, text_proj_dim, dense_hidden).to(device)

    weighting = params["loss_weighting"]
    if weighting == "learned_uncertainty":
        loss_module: UncertaintyWeightedLoss | None = UncertaintyWeightedLoss().to(device)
        opt_params = list(model.parameters()) + list(loss_module.parameters())
        lambda_dir = lambda_vol = None
    else:
        loss_module = None
        lambda_dir, lambda_vol = LAMBDA_RATIOS[weighting]
        opt_params = list(model.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=params["learning_rate"], weight_decay=WEIGHT_DECAY)

    def compute_loss(out: dict, y_dir: torch.Tensor, y_vol: torch.Tensor) -> torch.Tensor:
        if loss_module is not None:
            loss, _ = loss_module(out["direction_logit"], out["vol_pred"], y_dir, y_vol)
        else:
            loss, _ = multitask_loss(
                out["direction_logit"], out["vol_pred"], y_dir, y_vol, lambda_dir, lambda_vol
            )
        return loss

    n = train_t["quant_seq"].shape[0]
    idx = np.arange(n)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(idx)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start : start + BATCH_SIZE]
            batch = {k: train_t[k][batch_idx] for k in TENSOR_KEYS_ALL}
            y_dir = train_t["y_direction"][batch_idx]
            y_vol = train_t["y_vol"][batch_idx]

            optimizer.zero_grad()
            out = model(batch)
            loss = compute_loss(out, y_dir, y_vol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
            out = model(val_batch)
            val_loss = compute_loss(out, val_t["y_direction"], val_t["y_vol"])
        val_loss_val = val_loss.item()

        if val_loss_val < best_val_loss - 1e-5:
            best_val_loss = val_loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
        out = model(val_batch)
        # se conserva la probabilidad, no solo la clase: el AUC la necesita, y
        # es la métrica que revela el colapso a clase constante (predecir
        # siempre lo mismo da una accuracy engañosamente alta, pero un Kappa
        # exactamente 0)
        dir_proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
        dir_pred = (dir_proba > 0.5).astype(float)
        vol_pred = out["vol_pred"].cpu().numpy()

    y_dir_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_dir_true, dir_pred),
        "f1": f1_score(y_dir_true, dir_pred, zero_division=0),
        "rmse_rv": _rmse(y_vol_true, vol_pred),
        "mae_rv": mean_absolute_error(y_vol_true, vol_pred),
        **classification_agreement_metrics(dir_proba, dir_pred, y_dir_true),
        "n_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
    }
    return model, metrics


def main() -> None:
    """Reentrena las cinco variantes en los folds de desarrollo, al
    horizonte por defecto de hpo.py, para los activos indicados (por
    defecto, `TICKERS`). Los hiperparámetros se leen del fichero de ese
    horizonte o de `--hyperparams`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a reentrenar (por defecto: TICKERS)")
    parser.add_argument("--hyperparams", type=str, default=None, help="Ruta al JSON de hiperparámetros (por defecto: results/best_hyperparameters_h1.json, el del horizonte de hpo.py)")
    args = parser.parse_args()
    tickers = tuple(args.tickers)
    tickers_tag = "_".join(t.lower() for t in args.tickers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reentrenamiento final en {device}, tickers={tickers}")

    # este script entrena al horizonte por defecto de hpo.py, así que necesita
    # los hiperparámetros de ese horizonte y no los de otro
    if args.hyperparams:
        hp_path = Path(args.hyperparams)
        best_hp = load_best_hyperparameters(hp_path)
    else:
        hp_path, best_hp = require_hyperparameters_for(HORIZON, RESULTS_DIR, tickers=tickers)
    print(f"Hiperparámetros: {hp_path.name}")
    all_data = precompute_all_data(device, tickers=tickers)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for variant in VARIANTS:
        params = best_hp[variant]["best_params"]
        print(f"\n=== {variant}: {params} ===")
        for d in all_data:
            model, metrics = train_and_eval_final(variant, params, d["train"], d["val"], device)

            ckpt_path = MODELS_DIR / f"{variant}_{d['ticker']}_{d['fold']}.pt"
            torch.save({"state_dict": model.state_dict(), "variant": variant, "params": params}, ckpt_path)

            row = {"ticker": d["ticker"], "fold": d["fold"], "variant": variant, **metrics}
            all_rows.append(row)
            print(
                f"  {d['ticker']} {d['fold']}: acc={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f} (epochs={metrics['n_epochs']})"
            )

    metrics_df = pd.DataFrame(all_rows)
    metrics_path = RESULTS_DIR / f"final_models_metrics_{tickers_tag}.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"\n{len(metrics_df)} filas -> {metrics_path}")

    # tabla final: una fila por variante, con sus hiperparámetros ganadores
    # y la mediana de sus métricas de validación a través de los 4 folds
    summary_rows = []
    for variant in VARIANTS:
        sub = metrics_df[metrics_df["variant"] == variant]
        row = {
            "modelo": variant,
            **best_hp[variant]["best_params"],
            "accuracy_mediana": round(sub["accuracy"].median(), 4),
            "f1_mediana": round(sub["f1"].median(), 4),
            "rmse_rv_mediana": round(sub["rmse_rv"].median(), 4),
            "mae_rv_mediana": round(sub["mae_rv"].median(), 4),
        }
        summary_rows.append(row)
    final_table = pd.DataFrame(summary_rows)
    final_table_path = RESULTS_DIR / f"tabla_hiperparametros_finales_{tickers_tag}.csv"
    final_table.to_csv(final_table_path, index=False)
    print(f"\nTabla final -> {final_table_path}")
    print(final_table.to_string(index=False))


if __name__ == "__main__":
    main()
