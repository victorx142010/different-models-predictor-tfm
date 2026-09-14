"""Comprueba, para una variante y un horizonte, si calibrar el umbral de
decisión en cada fold elimina el colapso a clase constante.

Para cada activo y fold carga el checkpoint de desarrollo ya entrenado
(models/{variante}_{activo}_fold{N}_h{H}_L20.pt), predice su validación y
calcula el umbral con el J de Youden sobre ese mismo fold. Después compara
accuracy, Kappa y colapso con el umbral de 0,5 y con el calibrado. No reentrena
nada y nunca usa el holdout.

Es la versión por fold de la calibración; el pipeline final calibra un único
umbral con los cuatro folds juntos (compute_dev_thresholds.py). Se lanza como
proceso independiente desde calibration_check_launcher.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, roc_curve

from src.training.hpo import TENSOR_KEYS_ALL, L, build_model, precompute_all_data

TICKERS = ("SPY", "QQQ", "KO", "GS")
MODELS_DIR = Path("models")


def _rmse(y_true, y_pred) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral que maximiza el J de Youden (sensibilidad + especificidad - 1)
    sobre la curva ROC; 0,5 si solo hay una clase."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])


def load_and_get_proba(variant: str, ticker: str, fold: str, horizon: int, val_t: dict, device: str) -> dict:
    """Carga el checkpoint de una combinación variante × activo × fold ×
    horizonte y predice su fold de validación, sin entrenar. Devuelve las
    probabilidades, las etiquetas y el error de volatilidad."""
    ckpt_path = MODELS_DIR / f"{variant}_{ticker}_{fold}_h{horizon}_L20.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    quant_input_dim = val_t["quant_seq"].shape[-1]
    model = build_model(variant, ckpt["params"], quant_input_dim=quant_input_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])

    model.eval()
    with torch.no_grad():
        val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
        out = model(val_batch)
        proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
        vol_pred = out["vol_pred"].cpu().numpy()

    y_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()
    return {"proba": proba, "y_true": y_true, "rmse_rv": _rmse(y_vol_true, vol_pred), "mae_rv": float(np.mean(np.abs(y_vol_true - vol_pred)))}


def main() -> None:
    """Evalúa la variante y el horizonte indicados en todos los activos y
    folds y guarda el resultado en `--out`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_data = precompute_all_data(device, tickers=TICKERS, horizon=args.horizon, lookback=L)

    rows = []
    for d in all_data:
        res = load_and_get_proba(args.variant, d["ticker"], d["fold"], args.horizon, d["val"], device)
        thr = youden_threshold(res["y_true"], res["proba"])
        pred_05 = (res["proba"] > 0.5).astype(float)
        pred_cal = (res["proba"] > thr).astype(float)
        auc = roc_auc_score(res["y_true"], res["proba"]) if len(np.unique(res["y_true"])) > 1 else float("nan")
        row = {
            "variant": args.variant, "horizon": args.horizon, "ticker": d["ticker"], "fold": d["fold"],
            "auc": auc, "rmse_rv": res["rmse_rv"], "mae_rv": res["mae_rv"], "umbral_calibrado": thr,
            "acc_05": accuracy_score(res["y_true"], pred_05),
            "kappa_05": cohen_kappa_score(res["y_true"], pred_05) if len(np.unique(pred_05)) > 1 else 0.0,
            "colapsado_05": len(np.unique(pred_05)) == 1,
            "acc_cal": accuracy_score(res["y_true"], pred_cal),
            "kappa_cal": cohen_kappa_score(res["y_true"], pred_cal) if len(np.unique(pred_cal)) > 1 else 0.0,
            "colapsado_cal": len(np.unique(pred_cal)) == 1,
        }
        rows.append(row)
        print(f"[{args.variant} h={args.horizon} {d['ticker']} {d['fold']}] "
              f"colapso 0.5={row['colapsado_05']} -> calibrado={row['colapsado_cal']} "
              f"kappa {row['kappa_05']:.3f}->{row['kappa_cal']:.3f}", flush=True)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
