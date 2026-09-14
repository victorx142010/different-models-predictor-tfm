"""Conjunto variable de noticias por sesión, sin agregar, para la variante
cross_attention, que atiende a cada noticia por separado.

En lugar de un vector medio por día, se guardan los embeddings de hasta
M_MAX=32 noticias de cada sesión. Como cada sesión tiene un número distinto de
noticias, se representa con una matriz de tamaño fijo (M_MAX, 768): las
posiciones sobrantes se rellenan con ceros y una máscara indica cuáles son
noticias reales. La atención usa esa máscara para ignorar el relleno.

Solo se guardan sesiones con al menos una noticia. Para los días sin noticias
el modelo ya tiene su propio mecanismo, el vector nulo aprendido
(`NullEmbedding` en base_model.py), así que guardar matrices vacías sería
redundante.

Si una sesión tiene más de M_MAX noticias, se conservan las de sentimiento más
marcado (menor p_neu), con el mismo criterio que la agregación diaria.

Salida: features/nlp/{ticker}_news_embeddings_variable_set.npz, con los arrays
    session_date     (n_sesiones,)
    embeddings       (n_sesiones, M_MAX, 768)
    attention_mask   (n_sesiones, M_MAX): True si es noticia real, False si es relleno
    n_real           (n_sesiones,): noticias reales usadas
    n_truncated      (n_sesiones,): noticias descartadas por superar M_MAX
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"

M_MAX = 32
EMBEDDING_DIM = 768


def _select_top_m(g: pd.DataFrame, m_max: int) -> pd.DataFrame:
    """Si hay más de `m_max` noticias, conserva las de sentimiento más
    marcado (menor p_neu)."""
    if len(g) <= m_max:
        return g
    return g.nsmallest(m_max, "p_neu")


def build_variable_set_view(ticker: str, m_max: int = M_MAX) -> dict[str, np.ndarray]:
    """Construye, para cada sesión con noticias de un activo, la matriz
    (m_max, 768) con relleno y su máscara. Devuelve un diccionario de arrays
    listo para guardar con `np.savez_compressed`."""
    emb_df = pd.read_parquet(FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet")
    emb_df = emb_df.dropna(subset=["session_date"])

    session_dates, emb_rows, mask_rows, n_real_list, n_truncated_list = [], [], [], [], []

    for session_date, g in emb_df.groupby("session_date"):
        n_total = len(g)
        g_sel = _select_top_m(g, m_max)
        n_real = len(g_sel)

        arr = np.zeros((m_max, EMBEDDING_DIM), dtype="float32")
        arr[:n_real] = np.stack(g_sel["embedding"].to_numpy())
        mask = np.zeros(m_max, dtype="bool")
        mask[:n_real] = True

        session_dates.append(session_date)
        emb_rows.append(arr)
        mask_rows.append(mask)
        n_real_list.append(n_real)
        n_truncated_list.append(max(0, n_total - m_max))

    return {
        "session_date": np.array(session_dates, dtype="datetime64[ns]"),
        "embeddings": np.stack(emb_rows),
        "attention_mask": np.stack(mask_rows),
        "n_real": np.array(n_real_list, dtype="int32"),
        "n_truncated": np.array(n_truncated_list, dtype="int32"),
    }


def main() -> None:
    """Construye y guarda el conjunto variable de noticias de los activos
    indicados (por defecto, `TICKERS`)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a procesar (por defecto: TICKERS)")
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"Construyendo vista de conjunto variable para {ticker}...")
        data = build_variable_set_view(ticker)
        out_path = FEATURES_NLP_DIR / f"{ticker}_news_embeddings_variable_set.npz"
        np.savez_compressed(out_path, **data)

        n_sessions = len(data["session_date"])
        n_days_truncated = int((data["n_truncated"] > 0).sum())
        print(
            f"  {n_sessions} sesiones con noticia, {n_days_truncated} superaron "
            f"M_max={M_MAX} (se recortaron priorizando sentimiento marcado)"
        )
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
