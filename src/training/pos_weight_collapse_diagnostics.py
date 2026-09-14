"""Prueba acotada: ¿ponderar la clase positiva en la BCE (`pos_weight`, en
`multitask_loss`) reduce el colapso a clase constante?

Se usan las 16 combinaciones activo × fold de desarrollo de early_fusion a h=5,
la variante que más colapsa, y se comparan dos condiciones:
- (a) los hiperparámetros ya existentes, sin pos_weight;
- (b) una búsqueda de Optuna con dos objetivos (maximizar el AUC de dirección y
  minimizar el RMSE de volatilidad) que incluye pos_weight entre los
  hiperparámetros.

La búsqueda usa el AUC y no Kappa porque Kappa depende del umbral de 0,5 y
premiaría configuraciones que casualmente encajan con él. Solo se usan folds de
desarrollo, nunca el holdout.

Salida:
    results/pos_weight_diagnostic_{baseline,treatment,trials}_early_fusion_h5.csv

La continuación está en pos_weight_threshold_diagnostics.py.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

from src.fusion_model.base_model import UncertaintyWeightedLoss, multitask_loss
from src.training.hpo import LAMBDA_RATIOS, TENSOR_KEYS_ALL, L, build_model, precompute_all_data
from src.training.hyperparameters_io import hyperparameters_path_for, load_best_hyperparameters

VARIANT = "early_fusion"
HORIZON = 5
TICKERS = ("SPY", "QQQ", "KO", "GS")
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SEED = 42
N_TRIALS = 20


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_eval(params: dict, train_t: dict, val_t: dict, device: str, pos_weight: float | None = None) -> dict:
    """Como `train_and_eval_final`, pero pasa `pos_weight` a la pérdida y
    devuelve también las probabilidades de dirección, necesarias para el AUC
    y Kappa."""
    torch.manual_seed(SEED)
    quant_input_dim = train_t["quant_seq"].shape[-1]
    model = build_model(VARIANT, params, quant_input_dim=quant_input_dim).to(device)

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

    def compute_loss(out, y_dir, y_vol):
        if loss_module is not None:
            loss, _ = loss_module(out["direction_logit"], out["vol_pred"], y_dir, y_vol, pos_weight)
        else:
            loss, _ = multitask_loss(
                out["direction_logit"], out["vol_pred"], y_dir, y_vol, lambda_dir, lambda_vol, pos_weight
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
        proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
        vol_pred = out["vol_pred"].cpu().numpy()

    y_dir_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()
    dir_pred = (proba > 0.5).astype(float)

    auc = roc_auc_score(y_dir_true, proba) if len(np.unique(y_dir_true)) > 1 else float("nan")
    kappa = cohen_kappa_score(y_dir_true, dir_pred) if len(np.unique(dir_pred)) > 1 else 0.0

    return {
        "accuracy": accuracy_score(y_dir_true, dir_pred),
        "f1": f1_score(y_dir_true, dir_pred, zero_division=0),
        "rmse_rv": _rmse(y_vol_true, vol_pred),
        "mae_rv": float(np.mean(np.abs(y_vol_true - vol_pred))),
        "auc": auc,
        "kappa": kappa,
        "collapsed": len(np.unique(dir_pred)) == 1,
        "n_epochs": epoch + 1,
    }


def evaluate_config(params: dict, all_data: list[dict], device: str, pos_weight: float | None = None) -> pd.DataFrame:
    """Entrena y evalúa una configuración en todas las combinaciones activo
    × fold y devuelve una fila de métricas por combinación."""
    rows = []
    for d in all_data:
        m = train_eval(params, d["train"], d["val"], device, pos_weight=pos_weight)
        rows.append({"ticker": d["ticker"], "fold": d["fold"], "pos_weight": pos_weight, **m})
        print(
            f"  [{d['ticker']} {d['fold']}] acc={m['accuracy']:.3f} auc={m['auc']:.3f} "
            f"kappa={m['kappa']:.3f} colapsado={m['collapsed']} rmse={m['rmse_rv']:.4f}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Evalúa las dos condiciones y guarda sus métricas y los trials de la
    búsqueda en results/."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Diagnóstico pos_weight/objetivo AUC+RMSE en {device}: {VARIANT}, h={HORIZON}, tickers={TICKERS}")

    all_data = precompute_all_data(device, tickers=TICKERS, horizon=HORIZON, lookback=L)
    print(f"Datos precomputados: {len(all_data)} combinaciones ticker x fold")

    print("\n=== (a) Baseline: hiperparámetros de producción, sin pos_weight ===")
    best_hp = load_best_hyperparameters(hyperparameters_path_for(HORIZON))
    base_params = best_hp[VARIANT]["best_params"]
    print(f"Hiperparámetros reutilizados: {base_params}")
    baseline_df = evaluate_config(base_params, all_data, device, pos_weight=None)
    baseline_df.to_csv("results/pos_weight_diagnostic_baseline_early_fusion_h5.csv", index=False)

    print("\n=== (b) Búsqueda multi-objetivo (AUC dirección / RMSE volatilidad) con pos_weight ===")

    def objective(trial: optuna.Trial):
        params = {
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
            "dropout": trial.suggest_float("dropout", 0.2, 0.4),
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-4, 1e-4]),
            "loss_weighting": trial.suggest_categorical("loss_weighting", ["1:1", "2:1", "1:2", "learned_uncertainty"]),
            "pos_weight": trial.suggest_float("pos_weight", 0.3, 1.5),
        }
        aucs, rmses = [], []
        for d in all_data:
            m = train_eval(params, d["train"], d["val"], device, pos_weight=params["pos_weight"])
            aucs.append(m["auc"])
            rmses.append(m["rmse_rv"])
        return float(np.nanmean(aucs)), float(np.mean(rmses))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nFrente de Pareto: {len(study.best_trials)} ensayos")
    for t in study.best_trials:
        print(f"  auc={t.values[0]:.4f} rmse={t.values[1]:.4f} params={t.params}")

    best_trial = max(study.best_trials, key=lambda t: t.values[0])
    print(f"\nEnsayo elegido (máximo AUC del frente de Pareto): {best_trial.params}")

    best_params = {k: v for k, v in best_trial.params.items() if k != "pos_weight"}
    best_pos_weight = best_trial.params["pos_weight"]
    treatment_df = evaluate_config(best_params, all_data, device, pos_weight=best_pos_weight)
    treatment_df.to_csv("results/pos_weight_diagnostic_treatment_early_fusion_h5.csv", index=False)

    study.trials_dataframe().to_csv("results/pos_weight_diagnostic_trials_early_fusion_h5.csv", index=False)

    print("\n=== Resumen: tasa de colapso en desarrollo (16 combinaciones) ===")
    print(f"(a) Baseline (sin pos_weight):        {baseline_df['collapsed'].sum()}/16 colapsadas, "
          f"AUC medio={baseline_df['auc'].mean():.4f}, RMSE mediano={baseline_df['rmse_rv'].median():.4f}")
    print(f"(b) Con pos_weight={best_pos_weight:.3f}:  {treatment_df['collapsed'].sum()}/16 colapsadas, "
          f"AUC medio={treatment_df['auc'].mean():.4f}, RMSE mediano={treatment_df['rmse_rv'].median():.4f}")


if __name__ == "__main__":
    main()
