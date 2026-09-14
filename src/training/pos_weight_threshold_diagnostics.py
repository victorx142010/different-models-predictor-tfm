"""Continuación de pos_weight_collapse_diagnostics.py, que mostró que
pos_weight reduce el colapso y sube el AUC, pero empeora la accuracy en varios
folds, incluso por debajo del azar. La causa es que la accuracy se mide con el
umbral fijo de 0,5, y pos_weight desplaza las probabilidades.

Este script añade la calibración del umbral con el J de Youden (el punto de la
curva ROC que maximiza sensibilidad + especificidad - 1) y compara cuatro
combinaciones sobre las mismas 16 combinaciones activo × fold de desarrollo
(early_fusion, h=5):
- (a) sin pos_weight y umbral 0,5;
- (b) sin pos_weight y umbral calibrado: ¿basta con calibrar, sin tocar el
  entrenamiento?;
- (c) con pos_weight y umbral 0,5;
- (d) con pos_weight y umbral calibrado.

Resultado: calibrar el umbral elimina el colapso por sí solo, sin reentrenar y
sin empeorar la volatilidad. Por eso el pipeline final calibra el umbral en
desarrollo (compute_dev_thresholds.py). Solo usa folds de desarrollo, nunca el
holdout.

Salida: results/pos_weight_threshold_diagnostic_early_fusion_h5.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, roc_curve

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

# el trial de mayor AUC del frente de Pareto de la búsqueda anterior
TREATMENT_PARAMS = {
    "hidden_size": 64, "num_layers": 1, "dropout": 0.23119890406724053,
    "learning_rate": 0.0005, "loss_weighting": "1:2",
}
TREATMENT_POS_WEIGHT = 0.5548069328139313


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral que maximiza el J de Youden (sensibilidad + especificidad - 1)
    sobre la curva ROC del propio fold de validación."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def train_and_get_proba(params: dict, train_t: dict, val_t: dict, device: str, pos_weight: float | None) -> dict:
    """Entrena una configuración en un fold y devuelve las probabilidades de
    dirección, las etiquetas y el error de volatilidad."""
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
            loss, _ = multitask_loss(out["direction_logit"], out["vol_pred"], y_dir, y_vol, lambda_dir, lambda_vol, pos_weight)
        return loss

    n = train_t["quant_seq"].shape[0]
    idx = np.arange(n)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

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

    y_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()
    return {"proba": proba, "y_true": y_true, "rmse_rv": _rmse(y_vol_true, vol_pred), "mae_rv": float(np.mean(np.abs(y_vol_true - vol_pred)))}


def metrics_at_threshold(y_true, proba, thr: float) -> dict:
    """Accuracy, F1, Kappa y si hay colapso a clase constante, con un umbral
    dado."""
    pred = (proba > thr).astype(float)
    kappa = cohen_kappa_score(y_true, pred) if len(np.unique(pred)) > 1 else 0.0
    return {
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "kappa": kappa,
        "collapsed": len(np.unique(pred)) == 1,
    }


def main() -> None:
    """Compara las cuatro combinaciones y guarda el resultado en results/."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calibración de umbral en {device}: {VARIANT}, h={HORIZON}, tickers={TICKERS}")
    all_data = precompute_all_data(device, tickers=TICKERS, horizon=HORIZON, lookback=L)

    best_hp = load_best_hyperparameters(hyperparameters_path_for(HORIZON))
    base_params = best_hp[VARIANT]["best_params"]

    rows = []
    for d in all_data:
        base = train_and_get_proba(base_params, d["train"], d["val"], device, pos_weight=None)
        treat = train_and_get_proba(TREATMENT_PARAMS, d["train"], d["val"], device, pos_weight=TREATMENT_POS_WEIGHT)

        for label, res in [("sin_pos_weight", base), ("con_pos_weight", treat)]:
            thr_calib = youden_threshold(res["y_true"], res["proba"])
            m_05 = metrics_at_threshold(res["y_true"], res["proba"], 0.5)
            m_cal = metrics_at_threshold(res["y_true"], res["proba"], thr_calib)
            auc = roc_auc_score(res["y_true"], res["proba"]) if len(np.unique(res["y_true"])) > 1 else float("nan")
            row = {
                "ticker": d["ticker"], "fold": d["fold"], "condicion": label,
                "rmse_rv": res["rmse_rv"], "mae_rv": res["mae_rv"], "auc": auc,
                "umbral_calibrado": thr_calib,
                "acc_05": m_05["accuracy"], "kappa_05": m_05["kappa"], "colapsado_05": m_05["collapsed"],
                "acc_calibrado": m_cal["accuracy"], "kappa_calibrado": m_cal["kappa"], "colapsado_calibrado": m_cal["collapsed"],
            }
            rows.append(row)
            print(f"  [{d['ticker']} {d['fold']} {label}] umbral={thr_calib:.3f} "
                  f"acc(0.5)={m_05['accuracy']:.3f}->acc(cal)={m_cal['accuracy']:.3f} "
                  f"kappa(0.5)={m_05['kappa']:.3f}->kappa(cal)={m_cal['kappa']:.3f} "
                  f"colapso(0.5)={m_05['collapsed']}->colapso(cal)={m_cal['collapsed']}")

    df = pd.DataFrame(rows)
    df.to_csv("results/pos_weight_threshold_diagnostic_early_fusion_h5.csv", index=False)

    print("\n=== Resumen (16 combinaciones, desarrollo) ===")
    for label in ["sin_pos_weight", "con_pos_weight"]:
        sub = df[df["condicion"] == label]
        print(f"\n{label}:")
        print(f"  umbral 0.5:       {sub['colapsado_05'].sum()}/16 colapsadas, "
              f"accuracy media={sub['acc_05'].mean():.3f}, kappa medio={sub['kappa_05'].mean():.3f}")
        print(f"  umbral calibrado: {sub['colapsado_calibrado'].sum()}/16 colapsadas, "
              f"accuracy media={sub['acc_calibrado'].mean():.3f}, kappa medio={sub['kappa_calibrado'].mean():.3f}")
        print(f"  RMSE mediano={sub['rmse_rv'].median():.4f}, AUC medio={sub['auc'].mean():.3f}")


if __name__ == "__main__":
    main()
