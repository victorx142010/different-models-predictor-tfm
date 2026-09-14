"""Pasa cada titular limpio por FinBERT y guarda su sentimiento y su embedding.

Por cada noticia se obtiene:
- Las tres probabilidades de sentimiento del clasificador (positivo, negativo y
  neutro), que suman 1.
- El embedding del token [CLS] de la última capa (768 dimensiones), que
  representa el titular completo. Lo usan todas las variantes con texto, y se
  guarda sin normalizar.

Los titulares se truncan a 128 tokens sin perder información, porque la
limpieza comprobó que ninguno supera esa longitud.

Se calcula una sola vez por activo y se guarda en disco, para no recalcular
miles de embeddings cada vez que se entrena un modelo.

Salida: features/nlp/{ticker}_news_embeddings.parquet, con las columnas
date_publicacion, ticker, p_pos, p_neg, p_neu y embedding (vector de 768).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"

FINBERT_MODEL = "ProsusAI/finbert"
BATCH_SIZE = 64
MAX_LENGTH = 128


class FinbertEncoder:
    """Carga una sola vez el tokenizador y el modelo FinBERT de Hugging Face
    y procesa titulares por lotes con `encode`. El nombre del modelo es un
    parámetro porque finbert_diagnostics.py lo usa también con un segundo
    modelo de contraste."""

    def __init__(self, model_name: str = FINBERT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ValueError:
            # yiyanghkust/finbert-tone no publica un tokenizador «fast», solo
            # vocab.txt, y AutoTokenizer falla al intentar convertirlo.
            # BertTokenizer lo carga directamente.
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, output_hidden_states=True
            )
        except ValueError:
            # algunos repositorios de FinBERT (como yiyanghkust/finbert-tone)
            # tienen un config.json sin `model_type`, lo que impide la
            # detección automática del tipo de modelo
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, output_hidden_states=True
            )
        self.model = self.model.to(self.device).eval()
        # orden de las clases según el config del modelo ({0: positive, 1:
        # negative, 2: neutral} en ProsusAI/finbert); se lee en lugar de
        # suponerlo, por si se cambia de modelo
        id2label = {i: lbl.lower() for i, lbl in self.model.config.id2label.items()}
        self.label_idx = {v: k for k, v in id2label.items()}
        for lbl in ("positive", "negative", "neutral"):
            if lbl not in self.label_idx:
                raise ValueError(f"Modelo {model_name} no tiene la etiqueta esperada '{lbl}'")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = BATCH_SIZE, return_embeddings: bool = True):
        """Procesa los titulares por lotes. Devuelve (probs, embeddings):
        probs es un array (n, 3) con las probabilidades positiva, negativa y
        neutra en ese orden, y embeddings un array (n, 768) con el vector
        [CLS] de cada titular, o None si `return_embeddings=False` y solo
        interesa el sentimiento."""
        all_probs = np.empty((len(texts), 3), dtype="float32")
        all_embeddings = np.empty((len(texts), 768), dtype="float32") if return_embeddings else None

        use_amp = self.device == "cuda"
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = self.model(**enc)

            probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
            all_probs[start : start + len(batch)] = probs[
                :, [self.label_idx["positive"], self.label_idx["negative"], self.label_idx["neutral"]]
            ]

            if return_embeddings:
                cls = out.hidden_states[-1][:, 0, :].float().cpu().numpy()
                all_embeddings[start : start + len(batch)] = cls

        return all_probs, all_embeddings


def encode_ticker_news(ticker: str, encoder: FinbertEncoder) -> pd.DataFrame:
    """Codifica las noticias limpias de un activo y devuelve un DataFrame
    con el sentimiento y el embedding de cada una."""
    news = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet")
    probs, embeddings = encoder.encode(news["texto_limpio"].tolist())

    out = pd.DataFrame(
        {
            "date_publicacion": news["date_publicacion"],
            "ticker": news["ticker"],
            "session_date": news["session_date"],
            "p_pos": probs[:, 0],
            "p_neg": probs[:, 1],
            "p_neu": probs[:, 2],
        }
    )
    out["embedding"] = list(embeddings)
    return out


def main() -> None:
    """Codifica con FinBERT las noticias de los activos indicados (por
    defecto, `TICKERS`) y guarda el resultado en features/nlp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a codificar (por defecto: TICKERS)")
    args = parser.parse_args()

    FEATURES_NLP_DIR.mkdir(parents=True, exist_ok=True)
    encoder = FinbertEncoder()
    print(f"FinBERT cargado en {encoder.device}")

    for ticker in args.tickers:
        print(f"Codificando noticias de {ticker}...")
        df = encode_ticker_news(ticker, encoder)
        out_path = FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet"
        df.to_parquet(out_path, index=False)
        print(
            f"  {len(df)} noticias | p_pos medio={df['p_pos'].mean():.3f} "
            f"p_neg medio={df['p_neg'].mean():.3f} p_neu medio={df['p_neu'].mean():.3f}"
        )
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
