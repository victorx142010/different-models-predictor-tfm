"""Limpia los precios descargados y calcula el retorno logarítmico diario.

Pasos:
1. Alinear cada serie al calendario de sesiones de la NYSE, para que no falte
   ni sobre ningún día de mercado.
2. Rellenar huecos: un día suelto que falte se interpola linealmente entre el
   anterior y el siguiente. Si faltan dos o más sesiones seguidas, no se
   interpola y se dejan en NaN, porque una recta entre puntos tan separados ya
   no aproxima bien el precio.
3. Calcular el retorno logarítmico diario sobre el precio ajustado (adj_close),
   que es la variable que entra al modelo.
4. Comprobar con los contrastes ADF y KPSS que el retorno es estacionario,
   condición necesaria para modelarlo con GARCH.

Salida: data/interim/{ticker}_prices_clean_{inicio}_{fin}.parquet, con las
columnas date, ticker, open, high, low, close, adj_close, volume y log_return,
más un informe en results/limpieza_precios_{activos}.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from statsmodels.tsa.stattools import adfuller, kpss

from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

OHLCV_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()


def align_to_nyse_calendar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Alinea la serie al calendario de la NYSE e interpola solo los huecos
    de una sesión. Devuelve la serie alineada y un informe con los huecos
    encontrados e interpolados."""
    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index).normalize()

    sessions = _nyse_sessions(df.index.min().strftime("%Y-%m-%d"), df.index.max().strftime("%Y-%m-%d"))
    missing = sessions.difference(df.index)

    reindexed = df.reindex(sessions)

    # Hay que distinguir un hueco de una sola sesión, que se puede interpolar,
    # de una racha de dos o más sesiones seguidas, que no. Para ello se agrupan
    # las fechas que faltan y son consecutivas en el calendario.
    is_missing = reindexed[OHLCV_COLS[0]].isna()
    isolated_gap_dates: list[pd.Timestamp] = []
    long_gap_dates: list[pd.Timestamp] = []
    missing_idx = list(reindexed.index[is_missing])
    groups: list[list[pd.Timestamp]] = []
    for ts in missing_idx:
        if groups and sessions.get_loc(ts) == sessions.get_loc(groups[-1][-1]) + 1:
            groups[-1].append(ts)
        else:
            groups.append([ts])

    for g in groups:
        if len(g) == 1:
            isolated_gap_dates.append(g[0])
        else:
            long_gap_dates.extend(g)

    # con limit=1 nunca se rellenan dos huecos seguidos; volver a poner NaN en
    # los huecos largos es una salvaguarda adicional
    interpolated = reindexed.copy()
    interpolated[OHLCV_COLS] = interpolated[OHLCV_COLS].interpolate(
        method="linear", limit=1, limit_area="inside"
    )
    if long_gap_dates:
        interpolated.loc[long_gap_dates, OHLCV_COLS] = np.nan

    report = {
        "n_sessions_expected": int(len(sessions)),
        "n_sessions_present_raw": int(len(df)),
        "n_missing_total": int(len(missing)),
        "n_isolated_1session_gaps_interpolated": len(isolated_gap_dates),
        "isolated_gap_dates": [d.strftime("%Y-%m-%d") for d in isolated_gap_dates],
        "n_long_gap_sessions_left_as_nan": len(long_gap_dates),
        "long_gap_dates": [d.strftime("%Y-%m-%d") for d in long_gap_dates],
    }

    interpolated = interpolated.reset_index(names="date")
    return interpolated, report


def compute_log_return(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna log_return = ln(adj_close_t / adj_close_{t-1})."""
    df = df.copy()
    df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    return df


def stationarity_tests(log_return: pd.Series) -> dict:
    """Aplica los contrastes ADF y KPSS al retorno logarítmico. Tienen
    hipótesis nulas opuestas (ADF: hay raíz unitaria; KPSS: la serie es
    estacionaria), así que usarlos juntos es más robusto que fiarse de uno
    solo."""
    series = log_return.dropna()
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    return {
        "n_obs": int(len(series)),
        "adf_statistic": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "adf_rejects_unit_root_at_5pct": bool(adf_p < 0.05),
        "kpss_statistic": float(kpss_stat),
        "kpss_pvalue": float(kpss_p),
        "kpss_rejects_stationarity_at_5pct": bool(kpss_p < 0.05),
    }


def clean_ticker(ticker: str) -> tuple[pd.DataFrame, dict]:
    """Limpieza completa de un activo: carga los precios sin procesar, los
    alinea al calendario, calcula el retorno y aplica los contrastes.
    Devuelve el DataFrame limpio y los dos informes."""
    raw_path = DATA_RAW_DIR / f"{ticker}_prices_raw_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
    raw = pd.read_parquet(raw_path)

    aligned, gap_report = align_to_nyse_calendar(raw)
    aligned = compute_log_return(aligned)
    aligned["ticker"] = ticker
    aligned = aligned[["date", "ticker"] + OHLCV_COLS + ["log_return"]]

    stat_report = stationarity_tests(aligned["log_return"])

    return aligned, {"gap_report": gap_report, "stationarity": stat_report}


def main() -> None:
    """Limpia los precios de los activos indicados (por defecto, `TICKERS`)
    y guarda un informe con los huecos y los contrastes de estacionariedad."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a limpiar (por defecto: TICKERS)")
    args = parser.parse_args()

    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    full_report = {}

    for ticker in args.tickers:
        print(f"Limpiando {ticker}...")
        clean_df, report = clean_ticker(ticker)
        full_report[ticker] = report

        out_path = (
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        clean_df.to_parquet(out_path, index=False)

        gr = report["gap_report"]
        st = report["stationarity"]
        print(
            f"  sesiones esperadas={gr['n_sessions_expected']} "
            f"faltantes={gr['n_missing_total']} "
            f"(interpoladas={gr['n_isolated_1session_gaps_interpolated']}, "
            f"huecos largos sin interpolar={gr['n_long_gap_sessions_left_as_nan']})"
        )
        print(
            f"  ADF p={st['adf_pvalue']:.4g} (rechaza raíz unitaria: "
            f"{st['adf_rejects_unit_root_at_5pct']}) | "
            f"KPSS p={st['kpss_pvalue']:.4g} (rechaza estacionariedad: "
            f"{st['kpss_rejects_stationarity_at_5pct']})"
        )
        print(f"  -> {out_path}")

    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_report = RESULTS_DIR / f"limpieza_precios_{tickers_tag}.json"
    out_report.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme de limpieza guardado en {out_report}")


if __name__ == "__main__":
    main()
