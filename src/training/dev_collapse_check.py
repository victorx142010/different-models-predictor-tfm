"""Comprueba en desarrollo cuántas combinaciones colapsan a una sola clase con
el umbral de 0,5.

Una combinación colapsa cuando el modelo predice la misma clase en todas las
sesiones de validación, pase lo que pase en la entrada. Su accuracy puede
parecer aceptable, porque acierta todas las sesiones de la clase mayoritaria,
pero su Kappa es 0. El AUC no depende del umbral e indica si, aun así, las
probabilidades ordenan bien las sesiones.

Para cada combinación variante × activo × horizonte se juntan las
probabilidades de validación de los cuatro folds, las mismas con las que
compute_dev_thresholds.py calibra el umbral. Solo usa los checkpoints de
desarrollo ya entrenados (los genera main.py): no reentrena nada y nunca usa el
holdout.

Salida: results/dev_collapse_check.csv, con una fila por combinación: accuracy
y Kappa con umbral 0,5, AUC, si colapsa y número de sesiones. Por pantalla
muestra cuántas combinaciones colapsan en cada variante.

Uso:
    python -m src.training.dev_collapse_check
    python -m src.training.dev_collapse_check --tickers SPY --horizons 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

from src.fusion_model.base_model import VARIANTS
from src.training.compute_dev_thresholds import HORIZONS, RESULTS_DIR, TICKERS, pooled_dev_probabilities
from src.training.hpo import L


def collapse_table(tickers=TICKERS, horizons=HORIZONS, variants=VARIANTS, lookback: int = L) -> pd.DataFrame:
    """Accuracy y Kappa con umbral 0,5, AUC y si hay colapso para cada
    combinación variante × activo × horizonte, con las probabilidades de
    validación de los cuatro folds juntas."""
    rows = []
    for horizon, variant, ticker, y_true, proba in pooled_dev_probabilities(tickers, horizons, variants, lookback):
        pred = (proba > 0.5).astype(float)
        row = {
            "variant": variant,
            "ticker": ticker,
            "horizon": horizon,
            "accuracy_05": accuracy_score(y_true, pred),
            "kappa_05": cohen_kappa_score(y_true, pred),
            "auc": roc_auc_score(y_true, proba),
            "colapsado": len(np.unique(pred)) == 1,
            "n": len(y_true),
        }
        rows.append(row)
        print(f"[{variant} {ticker} h={horizon}] colapsa={row['colapsado']}  "
              f"kappa={row['kappa_05']:.3f}  auc={row['auc']:.3f}  (n={row['n']})")
    return pd.DataFrame(rows)


def summary_by_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Combinaciones colapsadas, total y tasa de colapso de cada variante,
    ordenadas de mayor a menor tasa."""
    res = df.groupby("variant")["colapsado"].agg(colapsadas="sum", total="count")
    res["tasa"] = res["colapsadas"] / res["total"]
    return res.sort_values("tasa", ascending=False)


def default_out_path(tickers, horizons) -> Path:
    """Ruta de salida. Con los cuatro activos y los tres horizontes devuelve
    el fichero de referencia; con cualquier otra combinación añade un sufijo,
    para que una prueba puntual no lo sobrescriba."""
    if tuple(tickers) == TICKERS and tuple(horizons) == HORIZONS:
        return RESULTS_DIR / "dev_collapse_check.csv"
    tk = "_".join(t.lower() for t in tickers)
    hs = "_".join(f"h{h}" for h in horizons)
    return RESULTS_DIR / f"dev_collapse_check_{tk}_{hs}.csv"


def main() -> None:
    """Calcula la tabla de colapso, la guarda y muestra el resumen por
    variante."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS),
                        help=f"Activos que se comprueban (por defecto: {list(TICKERS)})")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS),
                        help=f"Horizontes que se comprueban (por defecto: {list(HORIZONS)})")
    parser.add_argument("--out", type=str, default=None,
                        help="Ruta de salida (por defecto se deduce de los activos y horizontes)")
    args = parser.parse_args()

    tickers = tuple(t.upper() for t in args.tickers)
    horizons = tuple(args.horizons)
    df = collapse_table(tickers, horizons)

    out_path = Path(args.out) if args.out else default_out_path(tickers, horizons)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nColapso a una sola clase con umbral 0,5 (validación de los cuatro folds):")
    print(summary_by_variant(df).to_string(formatters={"tasa": "{:.1%}".format}))
    print(f"\nTotal: {int(df['colapsado'].sum())} de {len(df)} combinaciones colapsan")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
