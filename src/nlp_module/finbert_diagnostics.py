"""Dos comprobaciones sobre el sentimiento de FinBERT antes de usarlo en el
resto del proyecto.

1. Modelo de contraste: se codifica el mismo corpus con un segundo FinBERT
   entrenado de forma independiente (yiyanghkust/finbert-tone) y se compara con
   el modelo principal (ProsusAI/finbert): porcentaje de titulares con la misma
   clase y correlación del sentimiento neto (p_pos - p_neg). Si dos modelos
   independientes coinciden, es más probable que la señal sea real y no un
   artefacto de uno de ellos.
2. Diagnóstico de señal: correlación entre el sentimiento neto diario y la
   dirección del precio a 1, 5, 20 y 60 sesiones, separando las sesiones de
   sentimiento positivo y negativo. Permite ver si la relación se mantiene,
   decae o crece con el horizonte, como aproximación al tiempo de reacción del
   mercado.

Solo mide correlaciones; no entrena ningún modelo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.nlp_module.finbert_embeddings import FinbertEncoder
from src.training.sequence_dataset import targets_with_horizon

DIAGNOSTIC_HORIZONS = (1, 5, 20, 60)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"
RESULTS_DIR = PROJECT_ROOT / "results"

CONTRAST_MODEL = "yiyanghkust/finbert-tone"
_CLASS_NAMES = np.array(["positive", "negative", "neutral"])


def _predicted_class(probs: np.ndarray) -> np.ndarray:
    """Clase más probable (positiva, negativa o neutra) de cada titular."""
    return _CLASS_NAMES[probs.argmax(axis=1)]


def compare_with_contrast_model(ticker: str, contrast_encoder: FinbertEncoder) -> dict:
    """Codifica las noticias de un activo con el modelo de contraste y las
    compara con las del modelo principal: tasa de acuerdo en la clase y
    correlación del sentimiento neto."""
    news = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet")
    primary = pd.read_parquet(FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet")

    contrast_probs, _ = contrast_encoder.encode(news["texto_limpio"].tolist(), return_embeddings=False)

    primary_probs = primary[["p_pos", "p_neg", "p_neu"]].to_numpy()
    primary_class = _predicted_class(primary_probs)
    contrast_class = _predicted_class(contrast_probs)

    agreement = float((primary_class == contrast_class).mean())
    net_primary = primary_probs[:, 0] - primary_probs[:, 1]
    net_contrast = contrast_probs[:, 0] - contrast_probs[:, 1]
    corr = float(np.corrcoef(net_primary, net_contrast)[0, 1])

    return {
        "n_articles": len(news),
        "class_agreement_rate": round(agreement, 4),
        "net_sentiment_correlation": round(corr, 4),
        "primary_class_dist": pd.Series(primary_class).value_counts(normalize=True).round(3).to_dict(),
        "contrast_class_dist": pd.Series(contrast_class).value_counts(normalize=True).round(3).to_dict(),
    }


def _signal_diagnostic_single_horizon(agg: pd.DataFrame, ticker: str, horizon: int) -> dict:
    """Diagnóstico de señal para un horizonte: correlación entre el
    sentimiento neto diario y la dirección a `horizon` sesiones, en total y
    separando sesiones de sentimiento positivo y negativo."""
    targets = targets_with_horizon(ticker, horizon)
    d_col = f"d_h{horizon}"

    merged = agg.merge(targets[["date", d_col]], on="date", how="inner").dropna(subset=[d_col, "net_sentiment"])
    # solo días con noticia real aportan señal de sentimiento distinta de cero
    merged_with_news = merged[merged["n_news"] > 0]

    if len(merged_with_news) < 30:
        return {"n_obs": len(merged_with_news), "note": "muestra insuficiente para correlación fiable"}

    positivos = merged_with_news[merged_with_news["net_sentiment"] > 0]
    negativos = merged_with_news[merged_with_news["net_sentiment"] < 0]

    corr = float(np.corrcoef(merged_with_news["net_sentiment"], merged_with_news[d_col])[0, 1])
    pct_up_positivo = float(positivos[d_col].mean()) if len(positivos) else None
    pct_up_negativo = float(negativos[d_col].mean()) if len(negativos) else None

    return {
        "n_obs": int(len(merged_with_news)),
        "corr_net_sentiment_vs_direccion": round(corr, 4),
        "n_obs_sentimiento_positivo": int(len(positivos)),
        "pct_up_tras_sentimiento_positivo": round(pct_up_positivo, 4) if pct_up_positivo is not None else None,
        "n_obs_sentimiento_negativo": int(len(negativos)),
        "pct_up_tras_sentimiento_negativo": round(pct_up_negativo, 4) if pct_up_negativo is not None else None,
        # Diferencia entre ambas polaridades: cuanto mayor, más separa el
        # sentimiento la dirección a este horizonte. Si decae al aumentar el
        # horizonte, sugiere que el mercado incorpora la noticia pronto; si se
        # mantiene o crece, que la incorpora más despacio.
        "asimetria_positivo_menos_negativo": (
            round(pct_up_positivo - pct_up_negativo, 4)
            if pct_up_positivo is not None and pct_up_negativo is not None
            else None
        ),
    }


def signal_diagnostic_by_horizon(ticker: str, horizons: tuple[int, ...] = DIAGNOSTIC_HORIZONS) -> dict:
    """Repite el diagnóstico de señal para cada horizonte (por defecto 1, 5,
    20 y 60 sesiones: aproximadamente un día, una semana, un mes y un
    trimestre) para comparar cómo evoluciona la relación con el plazo."""
    agg = pd.read_parquet(FEATURES_NLP_DIR / f"{ticker}_news_daily_agg_simple.parquet")
    return {f"h{h}": _signal_diagnostic_single_horizon(agg, ticker, h) for h in horizons}


def main() -> None:
    """Ejecuta las dos comprobaciones para los activos indicados (por
    defecto, `TICKERS`) y guarda un informe conjunto."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a diagnosticar (por defecto: TICKERS)")
    args = parser.parse_args()

    contrast_encoder = FinbertEncoder(CONTRAST_MODEL)
    print(f"Modelo de contraste cargado en {contrast_encoder.device}: {CONTRAST_MODEL}")

    report = {}
    for ticker in args.tickers:
        print(f"Comparando modelos para {ticker}...")
        comparison = compare_with_contrast_model(ticker, contrast_encoder)
        print(f"  acuerdo de clase: {comparison['class_agreement_rate']:.1%}, "
              f"corr. sentimiento neto: {comparison['net_sentiment_correlation']:.3f}")

        diag = signal_diagnostic_by_horizon(ticker)
        for h_label, d in diag.items():
            print(f"  diagnóstico de señal ({h_label}): {d}")

        report[ticker] = {"contrast_model_comparison": comparison, "signal_diagnostic_by_horizon": diag}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_path = RESULTS_DIR / f"finbert_diagnostics_{tickers_tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme guardado en {out_path}")


if __name__ == "__main__":
    main()
