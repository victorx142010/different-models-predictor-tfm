"""Punto de entrada del proyecto: ejecuta el pipeline completo para uno o
varios activos y cualquier combinación de horizonte y ventana.

Encadena, en orden, los módulos de src/: descarga y limpieza de precios y
noticias, variables GARCH y embeddings FinBERT, construcción de los datasets
walk-forward, entrenamiento y evaluación de las variantes y, opcionalmente,
búsqueda de hiperparámetros, líneas base y holdout. Si un activo aún no tiene
sus datos en caché, los descarga y calcula sobre la marcha.

Etapas opcionales:
- `--hpo`: busca hiperparámetros con Optuna para estos activos, horizonte y
  ventana y los guarda en results/best_hyperparameters_h{H}.json (si el fichero
  ya existe, pide confirmación antes de reemplazarlo). Sin esta opción se
  reutilizan los ya guardados.
- `--baselines`: evalúa también las tres líneas base en los folds de
  desarrollo.
- `--finbert-diagnostics`: comprueba la calidad de la señal de noticias (modelo
  de contraste y correlación entre sentimiento y dirección). No afecta al
  modelo.
- `--holdout`: evalúa una única vez sobre 2022-2023. Con `--dev-thresholds`
  aplica además el umbral calibrado en desarrollo.
- `--solo-holdout`: va directamente al holdout, sin reentrenar los folds de
  desarrollo.

Los datos de cada activo se guardan en caché y se reutilizan si ya existen. Los
resultados llevan en el nombre los activos, el horizonte y la ventana, para que
ejecuciones con distinta configuración no se sobrescriban:

    results/custom_run_{activos}_h{H}_L{L}.csv
    results/custom_run_{activos}_h{H}_L{L}_baselines.csv   (con --baselines)
    results/custom_run_{activos}_h{H}_L{L}_comparison.csv  (con --baselines)
    models/{variante}_{ACTIVO}_{fold}_h{H}_L{L}.pt

Uso:
    python main.py SPY QQQ KO GS --horizon 5 --hpo --baselines --holdout
    python main.py QQQ --horizon 20 --variants price_only cross_attention

Horizonte y ventana son independientes: `--horizon` indica cuántas sesiones
hacia delante se predice y `--lookback`, cuántas sesiones pasadas ve la LSTM.
El embargo se ajusta a max(5, horizonte) para que no haya solapamiento entre
entrenamiento y validación.

Limitaciones conocidas:
- Si yfinance no tiene el activo, se muestra su error tal cual.
- Si el activo no tiene historia suficiente antes de 2015, el GARCH no se puede
  ajustar y se informa con un error explícito.
- Si el activo no aparece en FNSPID, se continúa solo con la rama cuantitativa:
  las variantes con texto usan el vector nulo aprendido.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent

from src.baselines.arima_garch import evaluate_fold as evaluate_arima_garch_fold
from src.baselines.naive_baselines import evaluate_fold as evaluate_naive_baselines_fold
from src.fusion_model.base_model import VARIANTS
from src.ingestion.clean_news import clean_ticker_news
from src.ingestion.clean_prices import clean_ticker
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, download_ticker
from src.ingestion.fnspid_extract import extract_ticker_news
from src.ingestion.verify_data_sources import FNSPID_NEWS_FILES, search_ticker_in_fnspid_csv
from src.alignment.align_news import align_ticker_news
from src.alignment.news_daily_index import build_daily_news_index
from src.nlp_module.daily_aggregation import build_simple_daily_aggregation
from src.nlp_module.finbert_diagnostics import CONTRAST_MODEL, compare_with_contrast_model, signal_diagnostic_by_horizon
from src.nlp_module.finbert_embeddings import FinbertEncoder, encode_ticker_news
from src.nlp_module.variable_set_view import EMBEDDING_DIM, M_MAX, build_variable_set_view
from src.quant_module.garch_features import compute_garch_features, compute_warmup_segment
from src.targets.targets import HORIZONS as CANONICAL_HORIZONS
from src.targets.targets import compute_targets
from src.training import holdout_evaluation
from src.training.hpo import precompute_all_data, run_study
from src.training.hyperparameters_io import (
    hyperparameters_path_for,
    load_best_hyperparameters,
    require_hyperparameters_for,
    save_best_hyperparameters,
    warn_if_provenance_mismatch,
)
from src.training.train_final_models import train_and_eval_final
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS, WalkForwardSplitter

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FEATURES_QUANT_DIR = PROJECT_ROOT / "features" / "quant"
FEATURES_NLP_DIR = PROJECT_ROOT / "features" / "nlp"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_prices(ticker: str) -> Path:
    """Devuelve la ruta de los precios limpios de un activo; si aún no
    existen, los descarga y limpia antes."""
    raw_path = DATA_RAW_DIR / f"{ticker}_prices_raw_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    clean_path = DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    if clean_path.exists():
        print(f"[precios] {ticker}: ya en caché -> {clean_path.name}")
        return clean_path

    print(f"[precios] {ticker}: no encontrados en caché, descargando...")
    if not raw_path.exists():
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df = download_ticker(ticker)
        df.to_parquet(raw_path, index=False)

    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    clean_df, report = clean_ticker(ticker)
    clean_df.to_parquet(clean_path, index=False)
    gr, st = report["gap_report"], report["stationarity"]
    print(
        f"         {len(clean_df)} sesiones, huecos largos sin interpolar={gr['n_long_gap_sessions_left_as_nan']}, "
        f"ADF p={st['adf_pvalue']:.4g}, KPSS p={st['kpss_pvalue']:.4g}"
    )
    return clean_path


def ensure_canonical_targets(ticker: str, prices_path: Path) -> None:
    """Calcula y guarda las etiquetas de los horizontes 1, 3 y 5 si aún no
    existen. Los demás horizontes se calculan en memoria con
    `targets_with_horizon`."""
    targets_path = DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    if targets_path.exists():
        return
    print(f"[targets] {ticker}: calculando targets canónicos (h={CANONICAL_HORIZONS})...")
    prices = pd.read_parquet(prices_path)
    targets_df = compute_targets(prices, horizons=CANONICAL_HORIZONS)
    targets_df.to_parquet(targets_path, index=False)


def targets_with_horizon(ticker: str, horizon: int) -> pd.DataFrame:
    """Etiquetas de un activo para el horizonte pedido. Si es uno de los
    guardados (1, 3 o 5), las lee del fichero; si no, las calcula en memoria
    sin modificar el fichero, para no mezclar los horizontes de referencia
    con los de una exploración puntual."""
    targets_path = DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    canonical = pd.read_parquet(targets_path)
    if f"d_h{horizon}" in canonical.columns:
        return canonical
    print(f"         horizonte h={horizon} no está en el fichero canónico ({CANONICAL_HORIZONS}); calculando en memoria...")
    prices_path = DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    prices = pd.read_parquet(prices_path)
    extra = compute_targets(prices, horizons=(horizon,))
    return canonical.merge(extra, on=["date", "ticker"], how="left")


def ensure_garch_features(ticker: str, prices_path: Path, garch_suffix: str = "") -> None:
    """Calcula y guarda las variables GARCH por fold y el tramo de
    calentamiento de un activo, si aún no existen. Si el activo no tiene
    historia suficiente antes de 2015, lanza un error claro en lugar de un
    fallo difícil de interpretar de la librería.

    Con `garch_suffix="gjr"` calcula GJR-GARCH (o=1) y lo guarda en ficheros
    aparte."""
    o = 1 if garch_suffix == "gjr" else 0
    splitter = WalkForwardSplitter()  # embargo por defecto (5 sesiones), el mismo con el que se generaron los ficheros GARCH en caché
    span_start = splitter.folds[0].train_start.strftime("%Y-%m-%d")
    span_end = splitter.folds[-1].val_end.strftime("%Y-%m-%d")
    infix = f"_{garch_suffix}" if garch_suffix else ""
    garch_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features{infix}_{span_start}_{span_end}.parquet"
    warmup_path = FEATURES_QUANT_DIR / f"{ticker}_garch_features{infix}_warmup.parquet"
    label = "GJR-GARCH(1,1,1)" if o else "GARCH(1,1)"
    if garch_path.exists() and warmup_path.exists():
        print(f"[GARCH] {ticker}: {label} ya en caché -> {garch_path.name}")
        return

    print(f"[GARCH] {ticker}: ajustando {label} por fold...")
    FEATURES_QUANT_DIR.mkdir(parents=True, exist_ok=True)
    prices_df = pd.read_parquet(prices_path)
    try:
        features_df = compute_garch_features(prices_df, splitter, o=o)
    except Exception as exc:  # noqa: BLE001 - historial insuficiente en algún fold es un error real a comunicar, no a enmascarar
        raise RuntimeError(
            f"No se pudo ajustar {label} por fold para {ticker}. Motivo probable: historial de precios "
            f"insuficiente en alguno de los folds 2015-2021 (p.ej. una salida a bolsa reciente). "
            f"Error original: {exc}"
        ) from exc
    features_df.to_parquet(garch_path, index=False)

    try:
        warmup_df = compute_warmup_segment(prices_df, splitter.folds[0].train_start, o=o)
        warmup_df.to_parquet(warmup_path, index=False)
    except ValueError as exc:
        raise RuntimeError(
            f"No se pudo construir el segmento de calentamiento GARCH para {ticker}: {exc}. "
            f"Este ticker probablemente no tiene suficiente historial antes de 2015-01-02 "
            f"(p.ej. cotiza desde después de esa fecha)."
        ) from exc


def _empty_news_clean_df() -> pd.DataFrame:
    """Tabla vacía con las mismas columnas que produce clean_news.py. Se usa
    para los activos sin noticias en FNSPID, de modo que el resto del
    pipeline los trata igual que a los demás."""
    return pd.DataFrame(
        {
            "date_publicacion": pd.Series([], dtype="datetime64[ns, UTC]"),
            "ticker": pd.Series([], dtype="object"),
            "titular": pd.Series([], dtype="object"),
            "fuente": pd.Series([], dtype="object"),
            "texto_limpio": pd.Series([], dtype="object"),
            "n_tokens": pd.Series([], dtype="int64"),
            "session_date": pd.Series([], dtype="datetime64[ns]"),
        }
    )


def _write_empty_news_artifacts(ticker: str) -> None:
    """Para un activo sin noticias, escribe los mismos ficheros que
    generaría el pipeline de noticias, pero vacíos o con has_news=False en
    todas las sesiones. Así las etapas siguientes no tienen que comprobar si
    el activo tiene noticias."""
    news_clean_path = DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet"
    daily_index_path = DATA_INTERIM_DIR / f"{ticker}_news_daily_index.parquet"
    agg_path = FEATURES_NLP_DIR / f"{ticker}_news_daily_agg_simple.parquet"
    varset_path = FEATURES_NLP_DIR / f"{ticker}_news_embeddings_variable_set.npz"

    _empty_news_clean_df().to_parquet(news_clean_path, index=False)

    daily_index = build_daily_news_index(ticker)  # con el parquet de noticias vacío, esto simplemente marca has_news=False en todas las sesiones
    daily_index.to_parquet(daily_index_path, index=False)

    agg = daily_index[["date", "ticker", "n_news", "has_news"]].copy()
    for col in ["p_pos", "p_neg", "p_neu", "net_sentiment", "embedding"]:
        agg[col] = None
    agg.to_parquet(agg_path, index=False)

    np.savez_compressed(
        varset_path,
        session_date=np.array([], dtype="datetime64[ns]"),
        embeddings=np.zeros((0, M_MAX, EMBEDDING_DIM), dtype="float32"),
        attention_mask=np.zeros((0, M_MAX), dtype="bool"),
        n_real=np.zeros((0,), dtype="int32"),
        n_truncated=np.zeros((0,), dtype="int32"),
    )


def ensure_news_pipeline(ticker: str) -> bool:
    """Ejecuta, si hace falta, el pipeline de noticias completo de un
    activo: lo busca en FNSPID, extrae y limpia sus titulares, los alinea
    con las sesiones y calcula los embeddings de FinBERT. Si el activo no
    aparece en FNSPID o la extracción falla, escribe los ficheros vacíos de
    `_write_empty_news_artifacts`. Devuelve True si el activo tiene noticias
    y False si no."""
    news_clean_path = DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet"
    daily_index_path = DATA_INTERIM_DIR / f"{ticker}_news_daily_index.parquet"
    agg_path = FEATURES_NLP_DIR / f"{ticker}_news_daily_agg_simple.parquet"
    varset_path = FEATURES_NLP_DIR / f"{ticker}_news_embeddings_variable_set.npz"

    if news_clean_path.exists() and daily_index_path.exists() and agg_path.exists() and varset_path.exists():
        has_news = pd.read_parquet(news_clean_path).shape[0] > 0
        print(f"[noticias] {ticker}: ya en caché -> {news_clean_path.name} (cobertura real: {has_news})")
        return has_news

    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_NLP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[noticias] {ticker}: verificando cobertura en FNSPID (búsqueda binaria)...")
    search = search_ticker_in_fnspid_csv(FNSPID_NEWS_FILES[0], ticker)

    raw_news = None
    if search.found:
        print(f"         {ticker} encontrado en FNSPID ({search.iterations} iteraciones) -> extrayendo bloque de bytes...")
        try:
            raw_news = extract_ticker_news(ticker)
        except ValueError as exc:
            print(f"         [WARN] la extracción del bloque falló pese a que la búsqueda lo localizó ({exc}); "
                  f"se continúa sin noticias para {ticker}")
            raw_news = None
    else:
        print(f"         {ticker} no se encontró en FNSPID (o el fichero no está ordenado en ese tramo, "
              f"nota: '{search.note}')")

    if raw_news is None or raw_news.empty:
        print("         -> se continúa SOLO con la rama cuantitativa (vector nulo aprendido en todas las sesiones)")
        _write_empty_news_artifacts(ticker)
        return False

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_news.to_parquet(DATA_RAW_DIR / f"{ticker}_news_raw.parquet", index=False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    clean_df, report = clean_ticker_news(ticker, tokenizer)
    clean_df.to_parquet(news_clean_path, index=False)
    print(f"         {report['n_after_dedup']} titulares tras dedup ({report['pct_dropped']:.1%} descartado como duplicado)")

    aligned = align_ticker_news(ticker)
    aligned.to_parquet(news_clean_path, index=False)

    daily_index = build_daily_news_index(ticker)
    daily_index.to_parquet(daily_index_path, index=False)

    print("[noticias] FinBERT: generando embeddings...")
    encoder = FinbertEncoder()
    print(f"         FinBERT cargado en {encoder.device}")
    emb_df = encode_ticker_news(ticker, encoder)
    (FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet").parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(FEATURES_NLP_DIR / f"{ticker}_news_embeddings.parquet", index=False)

    agg_df = build_simple_daily_aggregation(ticker)
    agg_df.to_parquet(agg_path, index=False)

    varset = build_variable_set_view(ticker)
    np.savez_compressed(varset_path, **varset)

    return True


def train_all_variants(
    tickers: list[str],
    horizon: int,
    lookback: int,
    variants: list[str],
    device: str,
    best_hp: dict,
    garch_suffix: str = "",
    use_embedding_layernorm: bool = False,
    text_proj_dim: int | None = None,
) -> pd.DataFrame:
    """Entrena las variantes indicadas en los cuatro folds de desarrollo de
    todos los activos a la vez, con los hiperparámetros de `best_hp`, y
    guarda un checkpoint por combinación. Devuelve una tabla con una fila
    por variante, activo y fold.

    Construye los datos con `precompute_all_data` (hpo.py), la misma función
    que usa la búsqueda de hiperparámetros.

    `garch_suffix`, `use_embedding_layernorm` y `text_proj_dim` activan
    configuraciones alternativas: GJR-GARCH, LayerNorm sobre los embeddings
    y otra dimensión de proyección de texto. Están desactivadas por defecto;
    cuando se usan, el nombre de los checkpoints lleva un sufijo para no
    sobrescribir los de la configuración por defecto. Por defecto se
    mantiene text_proj_dim=64, porque 32 no resultó mejor de forma uniforme
    en todas las variantes."""
    embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
    print(f"[entrenamiento] construyendo datasets (tickers={tickers}, L={lookback}, h={horizon}, embargo={embargo} sesiones)...")
    all_data = precompute_all_data(
        device, garch_suffix=garch_suffix, tickers=tuple(tickers), horizon=horizon, lookback=lookback
    )
    if not all_data:
        raise RuntimeError(
            f"No se pudo construir ningún fold con datos para {tickers} (L={lookback}, h={horizon}). "
            f"¿Historial insuficiente para esa ventana/horizonte?"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    text_proj_tag = f"_textproj{text_proj_dim}" if text_proj_dim is not None else ""
    print(f"         Entrenando {len(variants)} variante(s) x {len(all_data)} combinación(es) ticker×fold "
          f"(garch={'GJR' if garch_suffix else 'simétrico'}, layernorm={use_embedding_layernorm}, "
          f"text_proj_dim={text_proj_dim if text_proj_dim is not None else 64})...")
    rows = []
    for variant in variants:
        params = best_hp[variant]["best_params"]
        for d in all_data:
            model, metrics = train_and_eval_final(
                variant, params, d["train"], d["val"], device, use_embedding_layernorm, text_proj_dim=text_proj_dim
            )
            ckpt_path = MODELS_DIR / f"{variant}_{d['ticker']}_{d['fold']}_h{horizon}_L{lookback}{text_proj_tag}.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "variant": variant,
                    "params": params,
                    "ticker": d["ticker"],
                    "horizon": horizon,
                    "lookback": lookback,
                },
                ckpt_path,
            )
            row = {
                "ticker": d["ticker"],
                "fold": d["fold"],
                "variant": variant,
                "horizon": horizon,
                "lookback": lookback,
                **metrics,
            }
            rows.append(row)
            print(
                f"         {d['ticker']} {d['fold']} {variant}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
                f"rmse_rv={metrics['rmse_rv']:.4f} (epochs={metrics['n_epochs']})"
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Punto de entrada: lee los argumentos, prepara los activos si hace
    falta y ejecuta la búsqueda de hiperparámetros, el entrenamiento, las
    líneas base, el diagnóstico de FinBERT y el holdout según las opciones
    indicadas."""
    parser = argparse.ArgumentParser(
        description="Ejecuta el pipeline completo (datos, variables, hiperparámetros, entrenamiento y evaluación) "
        "para uno o varios activos, con el horizonte y la ventana indicados. La búsqueda de hiperparámetros, "
        "las líneas base, el diagnóstico de FinBERT y el holdout son opcionales."
    )
    parser.add_argument("tickers", nargs="+", type=str, help="Uno o varios símbolos bursátiles, por ejemplo: SPY QQQ KO GS")
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Horizonte de predicción, en sesiones: cuántas sesiones hacia delante se predicen la dirección "
        "y la volatilidad (por defecto: 1)",
    )
    parser.add_argument(
        "--lookback", type=int, default=20,
        help="Ventana L: cuántas sesiones pasadas recibe la LSTM (por defecto: 20)",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
        help="Variantes que se entrenan y, con --hpo, para las que se buscan hiperparámetros (por defecto: las 5)",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Fuerza el dispositivo (por defecto, cuda si está disponible)")
    parser.add_argument(
        "--garch", choices=["symmetric", "gjr"], default="symmetric",
        help="Modelo de volatilidad para cond_vol y std_resid: 'symmetric', GARCH(1,1) (por defecto), "
        "o 'gjr', GJR-GARCH, que da más peso a los retornos negativos (efecto apalancamiento).",
    )
    parser.add_argument(
        "--layernorm", action="store_true",
        help="Aplica LayerNorm a los embeddings de FinBERT antes de usarlos (desactivado por defecto).",
    )
    parser.add_argument(
        "--text-proj-dim", type=int, default=None,
        help="Tamaño de la proyección del texto (por defecto: 64). Solo afecta a late_fusion, early_fusion "
        "y text_only: price_only no usa texto y cross_attention hace su propia proyección dentro de la "
        "atención. Los resultados se guardan con un sufijo para no sobrescribir los obtenidos con 64.",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="Busca hiperparámetros con Optuna para estos activos, horizonte y ventana, en lugar de reutilizar "
        "los guardados. Si el fichero de destino ya existe, pide confirmación antes de reemplazarlo.",
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Trials de Optuna por variante con --hpo (por defecto: 20)")
    parser.add_argument(
        "--hyperparams", type=str, default=None,
        help="Fichero JSON de hiperparámetros (por defecto: results/best_hyperparameters_h{H}.json, el del "
        "horizonte indicado). Con --hpo es donde se guardan; sin --hpo, de donde se leen.",
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Evalúa también las tres líneas base (camino aleatorio, persistencia y ARIMA-GARCH) en los folds "
        "de desarrollo, con el mismo horizonte.",
    )
    parser.add_argument(
        "--finbert-diagnostics", action="store_true",
        help="Ejecuta también el diagnóstico de FinBERT (comparación con un segundo modelo de sentimiento y "
        "correlación entre sentimiento y dirección) en los activos con noticias.",
    )
    parser.add_argument(
        "--dev-thresholds", type=str, default=None,
        help="CSV de umbrales calibrados en desarrollo (lo genera compute_dev_thresholds.py). Si se indica, "
        "el holdout se evalúa con el umbral calibrado además de con 0,5. Sin él solo se usa 0,5, con el que "
        "la accuracy puede parecer buena aunque el modelo prediga siempre la misma clase.",
    )
    parser.add_argument(
        "--solo-holdout", action="store_true",
        help="Se salta el entrenamiento en los 4 folds de desarrollo y va directamente al holdout (implica "
        "--holdout). Sirve cuando los modelos ya están entrenados y calibrados, para no sobrescribir los "
        "checkpoints con los que se calcularon los umbrales. No es compatible con --baselines, que solo "
        "evalúa en desarrollo.",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="Evalúa también, una única vez, en el holdout 2022-2023, que no se usa en ningún momento del desarrollo.",
    )
    args = parser.parse_args()
    if args.solo_holdout:
        if args.baselines:
            parser.error("--solo-holdout no es compatible con --baselines: las líneas base de main.py "
                         "solo se evalúan en los folds de desarrollo, que --solo-holdout se salta. "
                         "Para evaluarlas en el holdout, usa src.evaluation.holdout_baseline_significance.")
        args.holdout = True

    tickers = [t.upper() for t in args.tickers]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    garch_suffix = "" if args.garch == "symmetric" else "gjr"
    hp_path = Path(args.hyperparams) if args.hyperparams else hyperparameters_path_for(args.horizon, RESULTS_DIR)
    tickers_tag = "_".join(t.lower() for t in tickers)
    improved_tag = (
        ("_gjr" if garch_suffix else "")
        + ("_layernorm" if args.layernorm else "")
        + (f"_textproj{args.text_proj_dim}" if args.text_proj_dim is not None else "")
    )

    print(f"=== main.py: {tickers} | horizonte={args.horizon} sesiones | lookback={args.lookback} sesiones | "
          f"device={device} | garch={args.garch} | layernorm={args.layernorm} ===\n")

    news_coverage: dict[str, bool] = {}
    for ticker in tickers:
        prices_path = ensure_prices(ticker)
        ensure_canonical_targets(ticker, prices_path)
        ensure_garch_features(ticker, prices_path, garch_suffix=garch_suffix)
        has_news = ensure_news_pipeline(ticker)
        news_coverage[ticker] = has_news
        if not has_news:
            print("         (sin cobertura real de noticias: text_only/late_fusion/early_fusion/cross_attention "
                  "se apoyarán exclusivamente en el vector nulo aprendido)")

    if args.finbert_diagnostics:
        print("\n[FinBERT diagnóstico]")
        tickers_with_news = [t for t in tickers if news_coverage[t]]
        skipped = [t for t in tickers if not news_coverage[t]]
        if not tickers_with_news:
            print("         ningún ticker tiene cobertura real de noticias: nada que diagnosticar")
        else:
            contrast_encoder = FinbertEncoder(CONTRAST_MODEL)
            print(f"         modelo de contraste cargado en {contrast_encoder.device}: {CONTRAST_MODEL}")
            for ticker in tickers_with_news:
                comparison = compare_with_contrast_model(ticker, contrast_encoder)
                diag = signal_diagnostic_by_horizon(ticker)
                print(f"         {ticker}: acuerdo de clase={comparison['class_agreement_rate']:.1%}, "
                      f"corr. sentimiento neto={comparison['net_sentiment_correlation']:.3f}")
                for h_label, d in diag.items():
                    print(f"         {ticker} ({h_label}): {d}")
        if skipped:
            print(f"         sin cobertura real de noticias, omitidos: {skipped}")

    if args.hpo:
        print(f"\n[HPO] buscando hiperparámetros para tickers={tickers}, h={args.horizon}, L={args.lookback} "
              f"({args.n_trials} trials/variante, garch={args.garch}, layernorm={args.layernorm})...")
        hpo_data = precompute_all_data(
            device, garch_suffix=garch_suffix, tickers=tuple(tickers), horizon=args.horizon, lookback=args.lookback
        )
        if not hpo_data:
            raise RuntimeError(
                f"No se pudo construir ningún fold con datos para {tickers} (L={args.lookback}, h={args.horizon})."
            )
        searched_params = {}
        for variant in args.variants:
            print(f"         --- {variant} ---")
            study = run_study(variant, hpo_data, device, args.n_trials, args.layernorm)
            searched_params[variant] = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
                "n_pruned": sum(1 for t in study.trials if t.state.name == "PRUNED"),
                "n_complete": sum(1 for t in study.trials if t.state.name == "COMPLETE"),
            }
            print(f"         {variant}: mejor pérdida val={study.best_value:.4f} params={study.best_params}")

        # si el fichero ya tenía hiperparámetros de otras variantes (porque
        # --variants no las incluye todas), se conservan: --hpo solo reemplaza
        # las variantes que se han buscado ahora
        best_hp = dict(load_best_hyperparameters(hp_path)) if hp_path.exists() else {}
        best_hp.pop("_meta", None)
        best_hp.update(searched_params)
        meta = {
            "tickers": tickers,
            "horizon": args.horizon,
            "lookback": args.lookback,
            "garch": args.garch,
            "layernorm": args.layernorm,
            "n_trials": args.n_trials,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_best_hyperparameters(hp_path, best_hp, meta)
    else:
        if args.hyperparams:
            # ruta explícita: se usa tal cual, y el aviso de procedencia de más
            # abajo indicará si no corresponde a esta configuración
            if not hp_path.exists():
                raise FileNotFoundError(f"No existe {hp_path} (indicado con --hyperparams).")
            best_hp = load_best_hyperparameters(hp_path)
        else:
            # sin ruta explícita, los hiperparámetros del propio horizonte; si
            # no se han buscado nunca, falla indicando cómo obtenerlos
            hp_path, best_hp = require_hyperparameters_for(args.horizon, RESULTS_DIR, tickers=tuple(tickers))
        print(f"\n[hiperparámetros] reutilizando {hp_path}")
        warn_if_provenance_mismatch(
            best_hp, hp_path, horizon=args.horizon, tickers=tuple(tickers),
            lookback=args.lookback, garch=args.garch,
        )
        missing = [v for v in args.variants if v not in best_hp]
        if missing:
            raise KeyError(
                f"{hp_path} no tiene hiperparámetros guardados para: {missing}. Usa --hpo o revisa --hyperparams."
            )

    if args.solo_holdout:
        # modelos ya entrenados y calibrados: no se reentrenan los folds de
        # desarrollo, porque se sobrescribirían los checkpoints usados para
        # calcular los umbrales
        print()
        print("[desarrollo] omitido por --solo-holdout: no se reentrenan los 4 folds")
    else:
        metrics_df = train_all_variants(
            tickers, args.horizon, args.lookback, args.variants, device, best_hp,
            garch_suffix=garch_suffix, use_embedding_layernorm=args.layernorm, text_proj_dim=args.text_proj_dim,
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"custom_run_{tickers_tag}_h{args.horizon}_L{args.lookback}{improved_tag}.csv"
        metrics_df.to_csv(out_path, index=False)

        print(f"\n=== Resultado ({', '.join(tickers)}, h={args.horizon}, L={args.lookback}) — mediana across ticker×fold ===")
        summary = metrics_df.groupby("variant")[
            ["accuracy", "f1", "roc_auc", "cohen_kappa", "rmse_rv", "mae_rv"]
        ].median().round(4)
        print(summary.to_string())

        # el colapso a clase constante no se ve en la accuracy, que sale alta
        # al predecir siempre la clase mayoritaria, pero sí en un Kappa
        # exactamente nulo; se avisa aquí para detectarlo antes de la
        # evaluación final
        colapsadas = metrics_df[metrics_df["cohen_kappa"].abs() < 1e-12]
        if not colapsadas.empty:
            n = len(colapsadas)
            print(f"\n!! AVISO: {n} de {len(metrics_df)} combinaciones colapsan a clase constante "
                  f"(Kappa = 0) con el umbral por defecto de 0,5.")
            print("     Su accuracy está inflada por el acierto fácil de la clase mayoritaria.")
            print("     Mira el AUC, que no depende del umbral. Para corregirlo, calibra con")
            print("     compute_dev_thresholds.py y pasa --dev-thresholds al evaluar el holdout.")
            for _, r in colapsadas.iterrows():
                print(f"       {r['ticker']} {r['fold']} {r['variant']}: "
                      f"accuracy={r['accuracy']:.4f} pero AUC={r['roc_auc']:.4f}")
        print(f"\n-> {out_path}")
        print(f"-> checkpoints en {MODELS_DIR}\\*_h{args.horizon}_L{args.lookback}{improved_tag}.pt")

    if args.baselines:
        print("\n[baselines]")
        # mismo embargo que las variantes, max(5, h), para que las líneas base
        # se validen sobre las mismas fechas que ellas
        splitter = WalkForwardSplitter(embargo_sessions=max(DEFAULT_EMBARGO_SESSIONS, args.horizon))
        baseline_rows = []
        for ticker in tickers:
            prices_df = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet")
            targets_df = targets_with_horizon(ticker, args.horizon)
            for fold in splitter.folds:
                baseline_rows.extend(
                    evaluate_naive_baselines_fold(ticker, fold, prices_df, targets_df, splitter, horizons=(args.horizon,))
                )
                print(f"         ARIMA-GARCH {ticker} {fold.name}...")
                baseline_rows.extend(
                    evaluate_arima_garch_fold(ticker, fold, prices_df, targets_df, splitter, horizons=(args.horizon,))
                )
        baselines_df = pd.DataFrame(baseline_rows)
        baselines_path = RESULTS_DIR / f"custom_run_{tickers_tag}_h{args.horizon}_L{args.lookback}{improved_tag}_baselines.csv"
        baselines_df.to_csv(baselines_path, index=False)
        print(f"         -> {baselines_path}")

        deep_long = metrics_df.melt(
            id_vars=["ticker", "fold", "variant"],
            value_vars=["accuracy", "f1", "rmse_rv", "mae_rv"],
            var_name="metric", value_name="value",
        ).rename(columns={"variant": "model"})
        combined = pd.concat(
            [deep_long, baselines_df[["ticker", "fold", "model", "metric", "value"]]], ignore_index=True
        )
        comparison = combined.groupby(["model", "metric"])["value"].median().unstack("metric").round(4)
        print("\n=== Comparación con baselines (mediana across ticker×fold) ===")
        print(comparison.to_string())
        comparison_path = RESULTS_DIR / f"custom_run_{tickers_tag}_h{args.horizon}_L{args.lookback}{improved_tag}_comparison.csv"
        comparison.to_csv(comparison_path)
        print(f"         -> {comparison_path}")

    if args.holdout:
        print("\n[holdout 2022-2023]")
        dev_thr = None
        if args.dev_thresholds:
            from src.training.compute_dev_thresholds import load_dev_thresholds
            dev_thr = load_dev_thresholds(Path(args.dev_thresholds), args.horizon)
            print(f"         umbral calibrado desde {args.dev_thresholds} ({len(dev_thr)} combinaciones)")
        else:
            print("         sin --dev-thresholds: se reporta con umbral 0,5 (sin calibrar)")
        holdout_evaluation.main(
            tickers=tuple(tickers),
            horizon=args.horizon,
            hp_path=hp_path,
            dev_thresholds=dev_thr,
            lookback=args.lookback,
            garch_suffix=garch_suffix,
            use_embedding_layernorm=args.layernorm,
            variants=tuple(args.variants),
            text_proj_dim=args.text_proj_dim,
        )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, KeyError) as e:
        # errores de configuración previsibles (falta el fichero de
        # hiperparámetros de este horizonte o no incluye alguna variante): el
        # mensaje ya explica qué hacer, y la traza completa solo lo ocultaría
        print("", file=sys.stderr)
        print(e.args[0] if e.args else e, file=sys.stderr)
        raise SystemExit(1)
