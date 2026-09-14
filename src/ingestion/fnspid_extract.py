"""Extrae de FNSPID solo las noticias de un activo, sin descargar el CSV
completo (23,2 GB).

El fichero `Stock_news/nasdaq_exteral_data.csv` está ordenado alfabéticamente
por la columna `Stock_symbol` en su mayor parte (aproximadamente el primer 75%,
donde están los cuatro activos del proyecto). Así, todas las filas de un
mismo ticker ocupan un tramo contiguo de bytes, y basta con localizar ese
tramo y pedirlo con una petición HTTP Range: entre 45 y 65 MB por activo.

Pasos:

1. Búsqueda binaria (`_find_block_bounds`): se piden trozos pequeños del
   fichero, se mira qué tickers aparecen y se decide en qué mitad seguir, como
   al buscar una palabra en un diccionario.
2. Se descarga el tramo encontrado con un margen de seguridad (`PAD`) a cada
   lado.
3. Se busca el inicio real de una fila con el patrón de las dos primeras
   columnas (índice y fecha UTC). No basta con el primer salto de línea, porque
   el texto de las noticias puede contener saltos de línea dentro del propio
   campo.
4. Se leen las filas con csv.reader, que entiende comillas y saltos de línea
   dentro de un campo, y se descarta la última fila si quedó cortada.
5. Se filtra por el ticker y se comprueba que el bloque está completo: justo
   antes y justo después debe aparecer una fila de otro ticker. Si no aparece,
   se lanza un error para ampliar el margen.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import pandas as pd

from src.ingestion.verify_data_sources import (
    FNSPID_BASE,
    FNSPID_NEWS_FILES,
    TICKERS,
    _extract_tickers_from_chunk,
    _http_get_range,
    _http_head_size,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"

NEWS_FILE = FNSPID_NEWS_FILES[0]  # el CSV que sí está ordenado por ticker (ver verify_data_sources.py)
NEWS_URL = f"{FNSPID_BASE}/{NEWS_FILE}"

ROW_START_RE = re.compile(r"\d+\.\d+,\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC,")

COLUMNS = [
    "idx", "date_pub", "titular", "ticker", "url", "fuente", "autor",
    "articulo", "lsa_summary", "luhn_summary", "textrank_summary", "lexrank_summary",
]

PAD = 400_000  # margen mayor que el error de la búsqueda binaria, para no cortar el bloque


def _tickers_in_window(url: str, size: int, offset: int, window: int) -> list[str]:
    """Tickers distintos que aparecen en un trozo de `window` bytes a partir
    de `offset`."""
    start = max(0, offset)
    end = min(size - 1, offset + window)
    if start >= end:
        return []
    chunk = _http_get_range(url, start, end)
    return sorted(set(_extract_tickers_from_chunk(chunk)))


def _find_block_bounds(
    url: str, size: int, ticker: str, probe_window: int = 200_000, max_iters: int = 45
) -> tuple[int, int]:
    """Localiza por bisección los dos extremos del tramo de bytes de un
    ticker, con una búsqueda para el inicio y otra para el final. El
    resultado puede desviarse hasta `probe_window` bytes, que se compensan
    descargando un margen extra (`PAD`)."""

    def bisect(want_left: bool) -> int:
        lo, hi = 0, size
        for _ in range(max_iters):
            if hi - lo <= probe_window:
                break
            mid = (lo + hi) // 2
            tk = _tickers_in_window(url, size, mid, probe_window)
            if not tk:
                tk = _tickers_in_window(url, size, mid, probe_window * 4)
                if not tk:
                    lo = min(lo + probe_window, hi)
                    continue
            if want_left:
                if tk[-1] < ticker:
                    lo = mid
                else:
                    hi = mid
            else:
                if tk[-1] <= ticker:
                    lo = mid
                else:
                    hi = mid
        return lo if want_left else hi

    return bisect(True), bisect(False)


def _parse_block(raw: bytes, ticker: str) -> pd.DataFrame:
    """Convierte los bytes descargados en un DataFrame. Primero busca un
    inicio de fila fiable con `ROW_START_RE`, porque la descarga puede
    empezar a mitad de una fila, y lee desde ahí con csv.reader. La última
    fila puede quedar cortada; como le faltan columnas, el filtro `len(row)
    == len(COLUMNS)` la descarta."""
    text = raw.decode("utf-8", errors="replace")
    m = ROW_START_RE.search(text)
    if m is None:
        raise ValueError(f"No se encontró un inicio de fila reconocible para {ticker}")
    body = text[m.start():]

    reader = csv.reader(io.StringIO(body))
    rows = [row for row in reader if len(row) == len(COLUMNS)]
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def extract_ticker_news(ticker: str) -> pd.DataFrame:
    """Devuelve todas las noticias de un ticker en FNSPID: localiza su
    tramo, lo descarga, lo lee y comprueba que el bloque está completo."""
    size = _http_head_size(NEWS_URL)
    left, right = _find_block_bounds(NEWS_URL, size, ticker)
    left_padded = max(0, left - PAD)
    right_padded = min(size - 1, right + PAD)

    raw = _http_get_range(NEWS_URL, left_padded, right_padded)
    df = _parse_block(raw, ticker)

    all_tickers_seen = df["ticker"].tolist()
    match_mask = df["ticker"] == ticker
    idx_match = df.index[match_mask]

    if len(idx_match) == 0:
        raise ValueError(f"No se encontraron filas de {ticker} en el bloque descargado")

    # Si se capturó el bloque completo, justo antes y justo después de las
    # filas del ticker debe haber filas de otro ticker (sus vecinos
    # alfabéticos). Si no aparecen, el bloque se cortó y hay que ampliar PAD.
    first_i, last_i = idx_match.min(), idx_match.max()
    has_prior_neighbor = first_i > 0 and all_tickers_seen[first_i - 1] != ticker
    has_next_neighbor = last_i < len(all_tickers_seen) - 1 and all_tickers_seen[last_i + 1] != ticker

    if not has_prior_neighbor or not has_next_neighbor:
        raise ValueError(
            f"Frontera no verificada para {ticker}: no se ve un ticker vecino "
            f"distinto justo antes/después del bloque (prior={has_prior_neighbor}, "
            f"next={has_next_neighbor}). Aumentar PAD."
        )

    result = df.loc[match_mask].reset_index(drop=True)
    return result


def main() -> None:
    """Descarga las noticias de los activos indicados (por defecto,
    `TICKERS`) y las guarda sin procesar en data/raw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a extraer (por defecto: TICKERS)")
    args = parser.parse_args()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        print(f"Extrayendo noticias de {ticker}...")
        df = extract_ticker_news(ticker)
        out_path = DATA_RAW_DIR / f"{ticker}_news_raw.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {len(df)} filas -> {out_path}")
        print(f"  fechas: {df['date_pub'].min()} a {df['date_pub'].max()}")


if __name__ == "__main__":
    main()
