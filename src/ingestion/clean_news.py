"""Limpia el texto de las noticias, elimina titulares casi duplicados y calcula
métricas de calidad antes de pasar el corpus a FinBERT.

Pasos:

1. Limpieza de texto (`clean_text`): decodifica entidades HTML (`&amp;`...),
   quita restos de etiquetas, normaliza el Unicode y colapsa espacios
   repetidos, para dejar el texto en un formato homogéneo.
2. Deduplicación de titulares casi idénticos (`dedup_near_identical`, con
   MinHash y LSH de la librería datasketch). Un mismo titular puede publicarse
   varias veces en fechas distintas, por ejemplo cuando varios medios lo
   redistribuyen. Si se conservaran todas las copias, el modelo trataría como
   información nueva algo que el mercado ya incorporó la primera vez. Por eso,
   de cada grupo de duplicados solo se conserva la publicación más antigua.
3. Conteo de tokens con el tokenizador de FinBERT (`n_tokens`). No se usa para
   generar los embeddings: sirve para comprobar que ningún titular supera los
   128 tokens a los que se trunca después.
4. Control de calidad (`ticker_matcher_qc`): qué fracción de titulares menciona
   el símbolo o el nombre de la empresa. Es solo informativo, porque FNSPID ya
   asigna cada noticia a su ticker.

Salida: data/interim/{ticker}_news_clean.parquet, con las columnas
date_publicacion, ticker, titular, fuente, texto_limpio y n_tokens, más un
informe en results/limpieza_noticias_{activos}.json.
"""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

TICKER_ALIASES = {
    "SPY": ["spy", "s&p 500", "s&p500", "spdr"],
    "QQQ": ["qqq", "nasdaq-100", "nasdaq 100", "invesco qqq"],
    # El símbolo solo («KO», «GS») es demasiado corto: como subcadena
    # aparecería dentro de otras palabras («smoke», «book», «gasoline»...). Por
    # eso se usan alias de texto más distintivos.
    "KO": ["coca-cola", "coca cola"],
    "GS": ["goldman sachs", "goldman"],
}

MINHASH_NUM_PERM = 64
MINHASH_THRESHOLD = 0.8
SHINGLE_SIZE = 3  # palabras por shingle


def clean_text(text: str) -> str:
    """Limpieza básica de un titular: decodifica entidades HTML, quita
    etiquetas sueltas, normaliza el Unicode (NFKC) y colapsa los espacios
    repetidos."""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Divide el texto en «shingles» de k palabras consecutivas. Por
    ejemplo, con k=3, «el mercado sube hoy» da {«el mercado sube», «mercado
    sube hoy»}. MinHash compara estos conjuntos para medir el parecido entre
    titulares."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(text: str) -> MinHash:
    """Calcula la huella MinHash de un texto a partir de sus shingles.
    Textos parecidos producen huellas parecidas, lo que permite buscar casi
    duplicados sin comparar cada titular con todos los demás."""
    m = MinHash(num_perm=MINHASH_NUM_PERM)
    for shingle in _shingles(text):
        m.update(shingle.encode("utf8"))
    return m


def dedup_near_identical(
    df: pd.DataFrame,
    text_col: str = "texto_limpio",
    date_col: str = "date_publicacion",
    threshold: float = MINHASH_THRESHOLD,
) -> tuple[pd.DataFrame, int]:
    """Agrupa los titulares casi idénticos y conserva, de cada grupo, solo
    el publicado primero. Devuelve el DataFrame deduplicado y el número de
    filas eliminadas.

    Comparar cada titular con todos los demás tendría un coste O(n²). En su
    lugar se usa LSH (Locality-Sensitive Hashing): a partir de la huella
    MinHash, `lsh.query` devuelve solo los candidatos con alta probabilidad
    de superar el umbral. Los pares candidatos se agrupan con una estructura
    union-find, de modo que si A se parece a B y B se parece a C, los tres
    quedan en el mismo grupo aunque A y C no se hayan comparado
    directamente."""
    n = len(df)
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    minhashes = []
    for i, text in enumerate(df[text_col].tolist()):
        mh = _minhash(text)
        minhashes.append(mh)
        lsh.insert(str(i), mh)

    # union-find con compresión de caminos para agrupar los pares parecidos que
    # devuelve LSH
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j_str in lsh.query(minhashes[i]):
            j = int(j_str)
            if j != i:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    # de cada grupo de duplicados solo se conserva el que se publicó antes
    dates = df[date_col].tolist()
    keep_idx = sorted(min(members, key=lambda idx: dates[idx]) for members in groups.values())
    n_dropped = n - len(keep_idx)
    return df.iloc[keep_idx].reset_index(drop=True), n_dropped


def ticker_matcher_qc(df: pd.DataFrame, ticker: str, text_col: str = "texto_limpio") -> float:
    """Porcentaje de titulares que mencionan el ticker o alguno de sus alias
    (por ejemplo, el nombre de la empresa). Es informativo y no descarta
    filas. Para activos sin alias en `TICKER_ALIASES` se busca solo el
    símbolo, para que la función funcione con cualquier activo nuevo."""
    aliases = TICKER_ALIASES.get(ticker, [ticker.lower()])
    pattern = re.compile("|".join(re.escape(a) for a in aliases), re.IGNORECASE)
    return float(df[text_col].str.contains(pattern, regex=True).mean())


def clean_ticker_news(ticker: str, tokenizer) -> tuple[pd.DataFrame, dict]:
    """Limpieza completa de las noticias de un activo: carga el texto sin
    procesar, lo limpia, elimina casi duplicados, cuenta tokens y calcula el
    informe de calidad. Recibe el tokenizador ya cargado para no cargarlo en
    cada llamada."""
    raw = pd.read_parquet(DATA_RAW_DIR / f"{ticker}_news_raw.parquet")

    df = pd.DataFrame(
        {
            "date_publicacion": pd.to_datetime(raw["date_pub"], format="%Y-%m-%d %H:%M:%S UTC", utc=True),
            "ticker": raw["ticker"],
            "titular": raw["titular"],
            "fuente": raw["fuente"].replace("", pd.NA),
            "texto_limpio": raw["titular"].map(clean_text),
        }
    )

    n_before_dedup = len(df)
    df, n_dropped_dup = dedup_near_identical(df)

    qc_match_rate = ticker_matcher_qc(df, ticker)

    tokens = tokenizer(df["texto_limpio"].tolist(), add_special_tokens=True)["input_ids"]
    df["n_tokens"] = [len(t) for t in tokens]

    df = df.sort_values("date_publicacion").reset_index(drop=True)
    df = df[["date_publicacion", "ticker", "titular", "fuente", "texto_limpio", "n_tokens"]]

    report = {
        "n_raw": n_before_dedup,
        "n_after_dedup": len(df),
        "n_dropped_as_near_duplicate": n_dropped_dup,
        "pct_dropped": round(n_dropped_dup / n_before_dedup, 4),
        "ticker_matcher_qc_match_rate": round(qc_match_rate, 4),
        "pct_fuente_missing": round(df["fuente"].isna().mean(), 4),
        "n_tokens_p50": int(df["n_tokens"].median()),
        "n_tokens_p95": int(df["n_tokens"].quantile(0.95)),
        "pct_exceeding_128_tokens": round((df["n_tokens"] > 128).mean(), 4),
        "date_min": str(df["date_publicacion"].min()),
        "date_max": str(df["date_publicacion"].max()),
    }
    return df, report


def main() -> None:
    """Limpia las noticias de los activos indicados (por defecto, `TICKERS`)
    y guarda un informe con sus métricas de calidad: tasa de duplicados,
    tokens por titular, etc."""
    import argparse
    import json

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a limpiar (por defecto: TICKERS)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    for ticker in args.tickers:
        print(f"Limpiando noticias de {ticker}...")
        df, report = clean_ticker_news(ticker, tokenizer)
        all_reports[ticker] = report
        out_path = DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {report}")
        print(f"  -> {out_path}")

    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_report = RESULTS_DIR / f"limpieza_noticias_{tickers_tag}.json"
    out_report.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme guardado en {out_report}")


if __name__ == "__main__":
    main()
