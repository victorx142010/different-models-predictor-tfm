"""Calibra el umbral de decisión de cada combinación variante × activo ×
horizonte con datos de desarrollo.

Para cada combinación se juntan las probabilidades de validación de los cuatro
folds y se calcula un único umbral con el estadístico J de Youden. Un umbral
sobre las cuatro muestras juntas es más estable que promediar cuatro umbrales
por fold. Se reutilizan los checkpoints de desarrollo ya entrenados, sin
reentrenar, y nunca se usa el holdout.

Además del umbral se guarda la tasa de clasificación positiva que produce. Es
lo que hereda la evaluación sobre el holdout (holdout_evaluation.py y
recalibrate_holdout_worker.py), que recalcula el corte en cada sesión para
reproducir esa tasa.

Salida: results/dev_calibrated_thresholds.csv, con 60 filas (5 variantes × 4
activos × 3 horizontes).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from src.fusion_model.base_model import VARIANTS
from src.training.hpo import TENSOR_KEYS_ALL, L, build_model, precompute_all_data

TICKERS = ("SPY", "QQQ", "KO", "GS")
HORIZONS = (5, 20, 60)
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral que maximiza el estadístico J de Youden (sensibilidad +
    especificidad - 1) sobre la curva ROC. Si solo hay una clase, devuelve
    0,5."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])


def default_out_path(tickers, horizons, results_dir: Path = RESULTS_DIR) -> Path:
    """Ruta de salida. Con la configuración de referencia (los cuatro
    activos y los tres horizontes) devuelve el fichero de referencia; con
    cualquier otra añade un sufijo, para que una prueba puntual no
    sobrescriba los umbrales usados en las tablas."""
    if tuple(tickers) == TICKERS and tuple(horizons) == HORIZONS:
        return results_dir / "dev_calibrated_thresholds.csv"
    tk = "_".join(t.lower() for t in tickers)
    hs = "_".join(f"h{h}" for h in horizons)
    return results_dir / f"dev_calibrated_thresholds_{tk}_{hs}.csv"


def pooled_dev_probabilities(
    tickers=TICKERS,
    horizons=HORIZONS,
    variants=VARIANTS,
    lookback: int = L,
    device: str | None = None,
):
    """Recorre las combinaciones horizonte × variante × activo y, para cada
    una, junta las probabilidades de dirección de validación de los cuatro
    folds. Devuelve tuplas (horizonte, variante, activo, etiquetas,
    probabilidades).

    Carga los checkpoints de desarrollo ya entrenados, que genera main.py: no
    reentrena nada y nunca usa el holdout. La usan este script y
    dev_collapse_check.py."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    for horizon in horizons:
        all_data = precompute_all_data(device, tickers=tuple(tickers), horizon=horizon, lookback=lookback)
        by_ticker: dict[str, list[dict]] = {t: [] for t in tickers}
        for d in all_data:
            by_ticker[d["ticker"]].append(d)

        for variant in variants:
            for ticker in tickers:
                probas, y_trues = [], []
                for d in by_ticker[ticker]:
                    ckpt_path = MODELS_DIR / f"{variant}_{ticker}_{d['fold']}_h{horizon}_L{lookback}.pt"
                    if not ckpt_path.exists():
                        raise FileNotFoundError(
                            f"Falta {ckpt_path}.\n"
                            f"Los umbrales se calculan sobre los checkpoints de desarrollo ya entrenados.\n"
                            f"Entrénalos antes con:  python main.py {ticker} --horizon {horizon} --lookback {lookback}"
                        )
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    quant_input_dim = d["val"]["quant_seq"].shape[-1]
                    model = build_model(variant, ckpt["params"], quant_input_dim=quant_input_dim).to(device)
                    model.load_state_dict(ckpt["state_dict"])
                    model.eval()
                    with torch.no_grad():
                        val_batch = {k: d["val"][k] for k in TENSOR_KEYS_ALL}
                        proba = torch.sigmoid(model(val_batch)["direction_logit"]).cpu().numpy()
                    probas.append(proba)
                    y_trues.append(d["val"]["y_direction"].cpu().numpy())

                yield horizon, variant, ticker, np.concatenate(y_trues), np.concatenate(probas)


def compute(
    tickers=TICKERS,
    horizons=HORIZONS,
    variants=VARIANTS,
    lookback: int = L,
    device: str | None = None,
) -> pd.DataFrame:
    """Calcula el umbral (J de Youden) y la tasa positiva de cada
    combinación variante × activo × horizonte, juntando las probabilidades
    de validación de los cuatro folds. Necesita los checkpoints de
    desarrollo, que genera main.py."""
    rows = []
    for horizon, variant, ticker, pooled_y, pooled_proba in pooled_dev_probabilities(
        tickers, horizons, variants, lookback, device
    ):
        thr = youden_threshold(pooled_y, pooled_proba)
        tasa_positiva = float((pooled_proba > thr).mean())
        rows.append({
            "variant": variant, "ticker": ticker, "horizon": horizon,
            "umbral_calibrado": thr, "tasa_positiva_calibrada": tasa_positiva, "n_pooled": len(pooled_y),
        })
        print(f"[{variant} {ticker} h={horizon}] umbral={thr:.4f}  tasa positiva (n={len(pooled_y)}) = {tasa_positiva:.3f}")
    return pd.DataFrame(rows)


def load_dev_thresholds(path: Path, horizon: int) -> dict:
    """Lee un CSV de umbrales y devuelve el diccionario que espera
    `holdout_evaluation.main` para el horizonte pedido: {(variante, activo):
    (tasa, umbral)}."""
    df = pd.read_csv(path)
    sub = df[df["horizon"] == horizon]
    if sub.empty:
        disponibles = sorted(df["horizon"].unique().tolist())
        raise ValueError(
            f"{path} no tiene umbrales para h={horizon}. Horizontes disponibles: {disponibles}.\n"
            f"Calcúlalos con:  python -m src.training.compute_dev_thresholds --horizons {horizon}"
        )
    return {
        (r["variant"], r["ticker"]): (r["tasa_positiva_calibrada"], r["umbral_calibrado"])
        for _, r in sub.iterrows()
    }


def main() -> None:
    """Calcula los umbrales de las combinaciones indicadas y los guarda en
    results/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS),
                        help=f"Activos a calibrar (por defecto: {list(TICKERS)})")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS),
                        help=f"Horizontes a calibrar (por defecto: {list(HORIZONS)})")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                        help="Subconjunto de variantes (por defecto, las 5)")
    parser.add_argument("--lookback", type=int, default=L, help=f"Ventana L (por defecto: {L})")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de salida (por defecto se deduce de los parámetros)")
    args = parser.parse_args()

    tickers = tuple(t.upper() for t in args.tickers)
    horizons = tuple(args.horizons)
    df = compute(tickers, horizons, tuple(args.variants), args.lookback)

    out_path = Path(args.out) if args.out else default_out_path(tickers, horizons)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n-> {out_path} ({len(df)} filas)")


if __name__ == "__main__":
    main()
