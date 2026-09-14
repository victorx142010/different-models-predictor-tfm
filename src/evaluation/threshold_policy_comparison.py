"""Compara cuatro formas de fijar el umbral de decisión en el holdout y cuenta
en cuántas combinaciones el modelo acaba prediciendo siempre la misma clase.

Políticas, de la más simple a la que se usa en la evaluación final:
- umbral_05: el corte por defecto de 0,5.
- umbral_desarrollo: el umbral calibrado en desarrollo
  (compute_dev_thresholds.py), aplicado tal cual.
- cuantil_todo_holdout: se hereda de desarrollo la tasa de predicciones al alza
  y el corte se calcula con las probabilidades de todo el holdout. No es
  causal: el umbral del primer día depende de sesiones posteriores.
- cuantil_expansivo: también hereda la tasa, pero el corte de cada sesión solo
  usa las probabilidades observadas hasta ella
  (expanding_calibrated_threshold). Es la regla de la evaluación final.

El umbral de desarrollo no se puede aplicar tal cual porque el modelo final,
entrenado con todo 2015-2021, da probabilidades en otra escala que los modelos
de cada fold. En muchas combinaciones ese corte queda fuera del rango de sus
probabilidades y el modelo predice siempre la misma clase. La tasa, en cambio,
sí se puede trasladar de una escala a otra.

No entrena ni predice nada: parte de las predicciones del holdout que guarda
consolidate_holdout.py y de los umbrales de compute_dev_thresholds.py.

Salida: results/comparacion_politicas_umbral.csv, con una fila por política,
activo, horizonte y variante: accuracy, F1, Kappa, tasa de predicciones al
alza, p-valor de Pesaran-Timmermann y si colapsa. Por pantalla muestra cuántas
combinaciones colapsan con cada política.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from src.evaluation.financial_metrics import expanding_calibrated_threshold, pesaran_timmermann_test

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def policy_thresholds(proba: np.ndarray, positive_rate: float, dev_threshold: float) -> dict:
    """Umbral de cada política para las probabilidades de una combinación,
    en orden cronológico. La política expansiva devuelve un umbral por
    sesión; las demás, uno fijo."""
    return {
        "umbral_05": 0.5,
        "umbral_desarrollo": dev_threshold,
        "cuantil_todo_holdout": np.quantile(proba, 1.0 - positive_rate),
        "cuantil_expansivo": expanding_calibrated_threshold(proba, positive_rate, dev_threshold),
    }


def compare_policies(pred_df: pd.DataFrame, thr_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas de dirección de cada combinación activo × horizonte ×
    variante con cada política de umbral."""
    rates = {
        (r["variant"], r["ticker"], r["horizon"]): (r["tasa_positiva_calibrada"], r["umbral_calibrado"])
        for _, r in thr_df.iterrows()
    }
    rows = []
    for (ticker, horizon, variant), g in pred_df.groupby(["ticker", "horizon", "variant"]):
        # el umbral expansivo depende del orden temporal
        g = g.sort_values("date")
        # float32, el tipo con el que el modelo produce las probabilidades: con
        # float64 el cuantil cambia lo justo para que alguna sesión en la
        # frontera cambie de clase
        proba = g["proba_dir"].to_numpy(dtype=np.float32)
        y_dir = g["y_dir"].to_numpy()
        positive_rate, dev_threshold = rates[(variant, ticker, horizon)]

        for policy, threshold in policy_thresholds(proba, positive_rate, dev_threshold).items():
            pred = (proba > threshold).astype(float)
            rows.append({
                "politica": policy,
                "ticker": ticker,
                "horizon": horizon,
                "variant": variant,
                "accuracy": accuracy_score(y_dir, pred),
                "f1": f1_score(y_dir, pred, zero_division=0),
                "kappa": cohen_kappa_score(y_dir, pred),
                "tasa_pos": float(pred.mean()),
                "pt_p": pesaran_timmermann_test(pred, y_dir)["p_value"],
                "colapsado": len(np.unique(pred)) == 1,
            })
    return pd.DataFrame(rows)


def summary_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Combinaciones que colapsan y Kappa mediano con cada política, en el
    orden en que se definen las políticas."""
    return df.groupby("politica", sort=False).agg(
        colapsadas=("colapsado", "sum"),
        total=("colapsado", "count"),
        kappa_mediano=("kappa", "median"),
    )


def main() -> None:
    """Lee las predicciones del holdout y los umbrales de desarrollo, compara
    las políticas y guarda el resultado."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de salida (por defecto: results/comparacion_politicas_umbral.csv)")
    args = parser.parse_args()

    pred_df = pd.read_csv(RESULTS_DIR / "consolidado_predicciones_variantes.csv")
    thr_df = pd.read_csv(RESULTS_DIR / "dev_calibrated_thresholds.csv")
    df = compare_policies(pred_df, thr_df)

    out_path = Path(args.out) if args.out else RESULTS_DIR / "comparacion_politicas_umbral.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Combinaciones que predicen siempre la misma clase en el holdout, por política de umbral:")
    print(summary_by_policy(df).to_string(float_format="{:.3f}".format))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
