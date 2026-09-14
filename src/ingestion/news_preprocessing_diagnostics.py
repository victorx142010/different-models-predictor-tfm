"""Diagnóstico de dos constantes del preprocesamiento de noticias. No entrena
ningún modelo.

1. `m_max_truncation_stats`: comprueba cuántas sesiones superan el tope de
   `M_MAX=32` noticias del conjunto variable (variable_set_view.py) y cuántas
   noticias se pierden por ese recorte. Resultado: muy pocas, así que el tope
   apenas descarta información.
2. `dedup_threshold_sensitivity`: repite la deduplicación de clean_news.py con
   umbrales de 0,7, 0,8 y 0,9 y compara cuántos titulares se eliminan. Aquí el
   umbral sí influye, y se mantiene 0,8 como punto intermedio: con 0,9 se
   escapan duplicados reales, y con 0,7 se fusionan titulares distintos que
   solo comparten estructura.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.clean_news import MINHASH_THRESHOLD, clean_text, dedup_near_identical
from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"
RESULTS_DIR = PROJECT_ROOT / "results"

DEDUP_THRESHOLDS_TO_TEST = (0.7, 0.8, 0.9)


def m_max_truncation_stats(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Para cada activo: sesiones con noticias, percentiles 95 y 99 y máximo
    de noticias por sesión, sesiones que superan M_MAX=32 y total de
    noticias descartadas por el recorte."""
    rows = []
    for ticker in tickers:
        data = np.load(FEATURES_NLP_DIR / f"{ticker}_news_embeddings_variable_set.npz")
        n_real, n_trunc = data["n_real"], data["n_truncated"]
        n_sessions = len(n_real)
        n_days_over = int((n_trunc > 0).sum())
        rows.append(
            {
                "ticker": ticker,
                "n_sessions_con_noticia": n_sessions,
                "p95_noticias_dia": float(np.percentile(n_real, 95)),
                "p99_noticias_dia": float(np.percentile(n_real, 99)),
                "max_noticias_dia": int(n_real.max()),
                "dias_que_superan_m_max": n_days_over,
                "pct_dias_que_superan_m_max": round(n_days_over / n_sessions, 4),
                "noticias_descartadas_total": int(n_trunc.sum()),
            }
        )
    return pd.DataFrame(rows)


def dedup_threshold_sensitivity(
    tickers: tuple[str, ...], thresholds: tuple[float, ...] = DEDUP_THRESHOLDS_TO_TEST
) -> pd.DataFrame:
    """Para cada activo y umbral: titulares antes y después de deduplicar y
    porcentaje eliminado, a partir de las noticias sin procesar de data/raw."""
    rows = []
    for ticker in tickers:
        raw = pd.read_parquet(DATA_RAW_DIR / f"{ticker}_news_raw.parquet")
        df = raw.rename(columns={"date_pub": "date_publicacion"}).copy()
        df["texto_limpio"] = df["titular"].astype(str).map(clean_text)
        n_before = len(df)
        for threshold in thresholds:
            _, n_removed = dedup_near_identical(df, threshold=threshold)
            rows.append(
                {
                    "ticker": ticker,
                    "threshold": threshold,
                    "n_antes": n_before,
                    "n_despues": n_before - n_removed,
                    "n_eliminados": n_removed,
                    "pct_eliminado": round(n_removed / n_before, 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Ejecuta los dos diagnósticos y guarda sus resultados en results/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a diagnosticar (por defecto: TICKERS)")
    args = parser.parse_args()
    tickers = tuple(args.tickers)
    tickers_tag = "_".join(t.lower() for t in tickers)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1. M_MAX=32: fracción de sesiones truncadas en el conjunto variable ===")
    m_max_df = m_max_truncation_stats(tickers)
    print(m_max_df.to_string(index=False))
    m_max_path = RESULTS_DIR / f"m_max_truncation_stats_{tickers_tag}.csv"
    m_max_df.to_csv(m_max_path, index=False)
    print(f"-> {m_max_path}")

    print(f"\n=== 2. Sensibilidad del umbral de deduplicación MinHash (actual={MINHASH_THRESHOLD}) ===")
    dedup_df = dedup_threshold_sensitivity(tickers)
    print(dedup_df.to_string(index=False))
    dedup_path = RESULTS_DIR / f"dedup_threshold_sensitivity_{tickers_tag}.csv"
    dedup_df.to_csv(dedup_path, index=False)
    print(f"-> {dedup_path}")


if __name__ == "__main__":
    main()
