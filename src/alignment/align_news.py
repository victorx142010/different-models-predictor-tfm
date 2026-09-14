"""Asigna cada noticia a la sesión de bolsa en la que el mercado pudo
reaccionar a ella. Las noticias se publican a cualquier hora, incluso en fin de
semana, mientras que el mercado solo cotiza en sesiones concretas.

La regla es que una noticia publicada después del cierre de la sesión anterior
y hasta el cierre de la sesión t (incluido) se asigna a la sesión t:

    N_t = { noticias publicadas en (cierre_{t-1}, cierre_t] }

Así, N_t reúne exactamente la información disponible hasta el cierre de t: ni
una noticia posterior, que sería mirar al futuro, ni una menos.

No hace falta tratar aparte fines de semana ni festivos. Con la lista ordenada
de cierres reales de la NYSE, a cada publicación se le asigna la primera sesión
cuyo cierre es igual o posterior (`np.searchsorted`). Una noticia del sábado no
encuentra ningún cierre ese día, así que se asigna automáticamente al cierre
del lunes, junto con el resto del fin de semana.

Todas las horas se comparan en la zona horaria de la NYSE (America/New_York),
para evitar ambigüedades de huso horario.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.ingestion.download_prices import TICKERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

NY_TZ = "America/New_York"


def get_session_closes(start: str, end: str) -> pd.Series:
    """Hora exacta de cierre, con zona horaria, de cada sesión de la NYSE
    entre `start` y `end`, indexada por la fecha de la sesión."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    closes = schedule["market_close"].copy()
    closes.index = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    return closes.sort_index()


def assign_to_session(pub_timestamps: pd.Series, session_closes: pd.Series) -> pd.Series:
    """Asigna a cada instante de publicación la fecha de la primera sesión
    cuyo cierre es igual o posterior. Es una función pura, separada de
    `align_ticker_news` para poder probarla con casos construidos a mano.

    `session_closes` debe empezar con margen antes de la primera
    publicación. Si el calendario empezara justo en la fecha de la primera
    noticia, una noticia publicada poco antes se asignaría por error a esa
    primera sesión. Por eso `align_ticker_news` carga el calendario con
    varios días de margen.

    Las publicaciones posteriores a la última sesión cargada devuelven NaT,
    sin sesión asignada."""
    closes_sorted = session_closes.sort_index()
    session_dates = closes_sorted.index.values
    closes_ny = pd.DatetimeIndex(closes_sorted).tz_convert(NY_TZ)

    pub_ny = pd.DatetimeIndex(pub_timestamps).tz_convert(NY_TZ)

    # Se comparan instantes absolutos (nanosegundos desde 1970). Cambiar la
    # zona horaria no cambia el instante, solo cómo se muestra, así que .asi8
    # es válido y evita perder la zona horaria, que se perdería con .values.
    idx = np.searchsorted(closes_ny.asi8, pub_ny.asi8, side="left")

    assigned = np.full(len(pub_timestamps), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    in_range = idx < len(session_dates)
    assigned[in_range] = session_dates[idx[in_range]]

    return pd.Series(assigned, index=pub_timestamps.index, name="session_date")


def align_ticker_news(ticker: str, calendar_pad_days: int = 10) -> pd.DataFrame:
    """Carga las noticias limpias de un activo y añade la columna
    session_date con la sesión asignada. El calendario se carga con
    `calendar_pad_days` días de margen a cada lado, para que ninguna noticia
    se quede sin sesión por caer en el borde."""
    news = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet")

    cal_start = (news["date_publicacion"].min() - pd.Timedelta(days=calendar_pad_days)).strftime("%Y-%m-%d")
    cal_end = (news["date_publicacion"].max() + pd.Timedelta(days=calendar_pad_days)).strftime("%Y-%m-%d")
    closes = get_session_closes(cal_start, cal_end)

    news = news.copy()
    news["session_date"] = assign_to_session(news["date_publicacion"], closes)

    n_out_of_range = news["session_date"].isna().sum()
    if n_out_of_range:
        print(f"  [WARN] {n_out_of_range} noticias de {ticker} sin sesión asignable (fuera de rango)")

    return news


def main() -> None:
    """Alinea las noticias de los activos indicados (por defecto, `TICKERS`)
    y añade la columna session_date al fichero de noticias limpias."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a alinear (por defecto: TICKERS)")
    args = parser.parse_args()

    for ticker in args.tickers:
        print(f"Alineando noticias de {ticker} a sesiones NYSE...")
        aligned = align_ticker_news(ticker)
        out_path = DATA_INTERIM_DIR / f"{ticker}_news_clean.parquet"
        aligned.to_parquet(out_path, index=False)
        n_sessions_with_news = aligned["session_date"].nunique()
        print(f"  {len(aligned)} noticias -> {n_sessions_with_news} sesiones distintas con noticia")
        print(f"  -> {out_path} (columna session_date añadida)")


if __name__ == "__main__":
    main()
