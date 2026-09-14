"""Comprobaciones previas de las fuentes de datos, antes de construir nada
sobre ellas.

Ejecuta cinco comprobaciones rápidas:

1. Que yfinance devuelve precios para los activos del proyecto.
2. Que el repositorio de FNSPID en Hugging Face es accesible.
3. Que los CSV de noticias tienen las columnas esperadas.
4. Qué contiene el ZIP de precios de FNSPID, sin descargarlo entero.
5. Si el CSV de noticias (23,2 GB) está ordenado por `Stock_symbol`. Si lo
   está, se puede localizar el tramo de un activo con búsqueda binaria y
   peticiones HTTP Range, sin descargar el fichero completo.

Uso:
    python -m src.ingestion.verify_data_sources
    python -m src.ingestion.verify_data_sources AAPL MSFT
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

TICKERS = ["SPY", "QQQ", "KO", "GS"]

FNSPID_REPO = "Zihan1004/FNSPID"
FNSPID_BASE = f"https://huggingface.co/datasets/{FNSPID_REPO}/resolve/main"
FNSPID_NEWS_FILES = [
    "Stock_news/nasdaq_exteral_data.csv",
    "Stock_news/All_external.csv",
]
FNSPID_ZIP_FILE = "Stock_price/full_history.zip"

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# Busca el patrón «TICKER,https://», que aparece una vez por fila (el símbolo
# va justo antes de la URL). Se usa en lugar de separar por saltos de línea
# porque un trozo descargado del medio del fichero casi nunca empieza en el
# borde de una fila, y este patrón se reconoce igual.
_TICKER_URL_RE = re.compile(r",([A-Z][A-Z0-9.\-]{0,9}),https?://")


def _http_get_range(url: str, start: int, end: int, timeout: int = 30) -> bytes:
    """Pide solo el rango de bytes [start, end] de una URL remota (cabecera
    HTTP Range), sin descargar el fichero completo."""
    req = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_head_size(url: str, timeout: int = 30) -> int:
    """Pregunta por HEAD el tamaño en bytes de un fichero remoto sin
    descargar ni un byte del contenido."""
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=timeout) as resp:
        return int(resp.headers.get("Content-Length", "0"))


@dataclass
class TickerSearchResult:
    """Resultado de buscar un ticker por bisección: si se encontró, cuántas
    iteraciones hicieron falta, qué tickers vecinos se vieron y si el
    resultado es fiable."""

    ticker: str
    found: bool
    iterations: int
    nearest_tickers_seen: list[str] = field(default_factory=list)
    reliable: bool = True
    note: str = ""


def check_yfinance(tickers: list[str] = TICKERS) -> dict:
    """Descarga las dos últimas semanas de precios de cada activo para
    comprobar que yfinance responde."""
    import yfinance as yf

    end = datetime.today()
    start = end - timedelta(days=14)
    report = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
            )
            report[ticker] = {
                "ok": not df.empty,
                "rows": int(len(df)),
                "first_date": str(df.index.min()) if not df.empty else None,
                "last_date": str(df.index.max()) if not df.empty else None,
                "has_adj_close": "Adj Close" in df.columns.get_level_values(0)
                if not df.empty
                else False,
            }
        except Exception as exc:  # noqa: BLE001 - queremos capturar y reportar cualquier fallo de red/API
            report[ticker] = {"ok": False, "error": str(exc)}
    return report


def check_fnspid_repo_accessible() -> dict:
    """Comprueba que el repositorio de FNSPID responde y devuelve la lista
    de sus ficheros con su tamaño, sin descargar ninguno."""
    api_url = f"https://huggingface.co/api/datasets/{FNSPID_REPO}"
    with urlopen(api_url, timeout=30) as resp:
        meta = json.loads(resp.read())

    files = [s["rfilename"] for s in meta.get("siblings", [])]
    sizes = {}
    for fname in files:
        if fname == "README.md":
            continue
        try:
            sizes[fname] = _http_head_size(f"{FNSPID_BASE}/{fname}")
        except Exception as exc:  # noqa: BLE001
            sizes[fname] = f"error: {exc}"

    return {
        "repo_reachable": True,
        "private": meta.get("private"),
        "files": files,
        "file_sizes_bytes": sizes,
    }


def check_fnspid_news_schema(file: str = FNSPID_NEWS_FILES[0], n_bytes: int = 3000) -> list[str]:
    """Descarga solo los primeros bytes de un CSV de FNSPID y devuelve los
    nombres de sus columnas."""
    raw = _http_get_range(f"{FNSPID_BASE}/{file}", 0, n_bytes)
    header_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    return [col.strip("\r") for col in header_line.split(",")]


def list_fnspid_price_zip_contents() -> list[str]:
    """Lista el contenido del ZIP de precios de FNSPID sin descargarlo
    entero.

    Al final de todo ZIP hay un bloque (End Of Central Directory) que indica
    dónde empieza y cuánto ocupa su índice de contenidos. Se piden los
    últimos 64 KB para leer ese bloque y, después, solo el rango exacto del
    índice."""
    import struct

    url = f"{FNSPID_BASE}/{FNSPID_ZIP_FILE}"
    size = _http_head_size(url)

    tail = _http_get_range(url, max(0, size - 65_536), size - 1)
    idx = tail.rfind(b"PK\x05\x06")
    if idx == -1:
        raise ValueError("No se encontró el EOCD del ZIP en los últimos 64KB")
    sig, disknum, diskstart, ndisk, ntotal, cdsize, cdoffset, commentlen = struct.unpack(
        "<IHHHHIIH", tail[idx : idx + 22]
    )

    cd_and_eocd = _http_get_range(url, cdoffset, size - 1)
    buf = BytesIO(b"\x00" * cdoffset + cd_and_eocd)
    with zipfile.ZipFile(buf) as zf:
        return zf.namelist()


def _extract_tickers_from_chunk(chunk: bytes) -> list[str]:
    """Devuelve los símbolos de ticker que aparecen en un trozo de bytes
    descargado."""
    text = chunk.decode("utf-8", errors="ignore")
    return _TICKER_URL_RE.findall(text)


def search_ticker_in_fnspid_csv(
    file: str,
    ticker: str,
    chunk_size: int = 400_000,
    max_iters: int = 22,
    max_empty_regions: int = 4,
) -> TickerSearchResult:
    """Busca un ticker en un CSV de FNSPID por bisección, suponiendo que el
    fichero está ordenado por `Stock_symbol`.

    En cada iteración se descarga un trozo alrededor del punto medio del
    rango, se extraen los tickers que aparecen y se decide en qué mitad
    seguir, como al buscar una palabra en un diccionario. Es una
    comprobación por muestreo: si el ticker aparece, seguro que está; si no
    aparece, es un indicio fuerte, pero no una prueba.

    El orden se cumple en nasdaq_exteral_data.csv, pero no en
    All_external.csv, que mezcla tramos de noticias sin ticker con tramos
    ordenados. Si varias regiones seguidas no contienen ningún ticker, el
    resultado se marca como no fiable (`reliable=False`) en lugar de
    devolver «no encontrado»."""
    url = f"{FNSPID_BASE}/{file}"
    size = _http_head_size(url)
    lo, hi = 0, size
    seen: list[str] = []
    empty_regions = 0

    for i in range(max_iters):
        mid = (lo + hi) // 2
        start = max(0, mid - chunk_size // 2)
        end = min(size - 1, start + chunk_size)
        chunk = _http_get_range(url, start, end)
        tickers = sorted(set(_extract_tickers_from_chunk(chunk)))

        if not tickers:
            empty_regions += 1
            if empty_regions > max_empty_regions:
                return TickerSearchResult(
                    ticker,
                    False,
                    i + 1,
                    seen[-10:],
                    reliable=False,
                    note=(
                        f"{empty_regions} regiones consecutivas sin tickers "
                        "reconocibles: el fichero probablemente no está "
                        "globalmente ordenado por Stock_symbol en este tramo."
                    ),
                )
            # si el trozo no contiene tickers (por ejemplo, un bloque de
            # noticias sin ticker asignado), se avanza el límite inferior para
            # salir de esa zona
            lo = min(lo + chunk_size, hi)
            continue

        seen.extend(tickers)
        if ticker in tickers:
            return TickerSearchResult(ticker, True, i + 1, tickers)

        # se compara con los tickers del trozo para decidir en qué mitad seguir
        if tickers[-1] < ticker:
            lo = mid
        elif tickers[0] > ticker:
            hi = mid
        else:
            # el ticker cae dentro del rango alfabético de este trozo pero no
            # aparece: casi seguro que no está en el fichero
            return TickerSearchResult(ticker, False, i + 1, tickers)

        if hi - lo < chunk_size:
            break

    return TickerSearchResult(
        ticker,
        False,
        max_iters,
        seen[-10:],
        reliable=bool(seen),
        note="" if seen else "no se recogió ninguna muestra de ticker válida",
    )


def main() -> None:
    """Ejecuta las cinco comprobaciones y guarda un informe en JSON. Sin
    argumentos usa `TICKERS`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a comprobar (por defecto: TICKERS)")
    args = parser.parse_args()
    tickers = args.tickers

    report: dict = {"generated_at": datetime.now().isoformat()}

    print("== 1. yfinance ==")
    report["yfinance"] = check_yfinance(tickers)
    for ticker, res in report["yfinance"].items():
        print(f"  {ticker}: {res}")

    print("\n== 2. Repositorio FNSPID (Hugging Face) ==")
    report["fnspid_repo"] = check_fnspid_repo_accessible()
    for fname, size in report["fnspid_repo"]["file_sizes_bytes"].items():
        size_gb = size / 1e9 if isinstance(size, int) else size
        print(f"  {fname}: {size_gb if isinstance(size, str) else f'{size_gb:.2f} GB'}")

    print("\n== 3. Esquema de los CSV de noticias ==")
    report["fnspid_news_schema"] = {}
    for f in FNSPID_NEWS_FILES:
        cols = check_fnspid_news_schema(f)
        report["fnspid_news_schema"][f] = cols
        print(f"  {f}: {cols}")

    print("\n== 4. Contenido de Stock_price/full_history.zip ==")
    zip_contents = list_fnspid_price_zip_contents()
    csv_tickers = {
        n.split("/")[-1][:-4].upper()
        for n in zip_contents
        if n.startswith("full_history/") and n.endswith(".csv")
    }
    report["fnspid_price_zip_n_files"] = len(zip_contents)
    report["fnspid_price_zip_n_tickers"] = len(csv_tickers)
    print(f"  {len(zip_contents)} entradas ({len(csv_tickers)} tickers con CSV de precios)")
    for t in tickers:
        print(f"  ¿contiene {t}.csv?: {t in csv_tickers}")

    print(f"\n== 5. Búsqueda binaria de {'/'.join(tickers)} en FNSPID ==")
    report["ticker_coverage"] = {}
    for f in FNSPID_NEWS_FILES:
        report["ticker_coverage"][f] = {}
        for ticker in tickers:
            res = search_ticker_in_fnspid_csv(f, ticker)
            report["ticker_coverage"][f][ticker] = {
                "found": res.found,
                "iterations": res.iterations,
                "nearby_tickers": res.nearest_tickers_seen,
                "reliable": res.reliable,
                "note": res.note,
            }
            status = "ENCONTRADO" if res.found else "no encontrado"
            reliability = "" if res.reliable else "  [MUESTRA NO CONCLUYENTE]"
            print(
                f"  [{f}] {ticker}: {status}{reliability} "
                f"({res.iterations} iters, vecinos={res.nearest_tickers_seen})"
            )
            if res.note:
                print(f"      nota: {res.note}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in tickers)
    out_path = RESULTS_DIR / f"verificacion_fuentes_{tickers_tag}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme guardado en {out_path}")


if __name__ == "__main__":
    main()
