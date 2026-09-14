"""Pruebas de align_news.py: cada noticia se asigna a la primera sesión cuyo
cierre es igual o posterior a su publicación, también en fines de semana,
festivos y con el cambio de zona horaria.
"""

import pandas as pd
import pytest

from src.alignment.align_news import assign_to_session, get_session_closes


@pytest.fixture(scope="module")
def closes() -> pd.Series:
    # incluye un festivo (2023-01-16, MLK Day) y un fin de semana normal
    return get_session_closes("2023-01-01", "2023-01-31")


def test_publication_exactly_at_close_belongs_to_that_session(closes: pd.Series) -> None:
    session = pd.Timestamp("2023-01-04")
    close_t = closes.loc[session]
    result = assign_to_session(pd.Series([close_t]), closes)
    assert result.iloc[0] == pd.Timestamp(session)


def test_publication_one_microsecond_after_close_belongs_to_next_session(closes: pd.Series) -> None:
    session = pd.Timestamp("2023-01-04")
    next_session = pd.Timestamp("2023-01-05")
    close_t = closes.loc[session]
    just_after = close_t + pd.Timedelta(microseconds=1)
    result = assign_to_session(pd.Series([just_after]), closes)
    assert result.iloc[0] == next_session


def test_publication_one_microsecond_before_next_close_belongs_to_next_session(closes: pd.Series) -> None:
    session = pd.Timestamp("2023-01-04")
    close_t = closes.loc[session]
    just_before = close_t - pd.Timedelta(microseconds=1)
    result = assign_to_session(pd.Series([just_before]), closes)
    assert result.iloc[0] == session


def test_weekend_publication_rolls_forward_to_monday(closes: pd.Series) -> None:
    # 2023-01-07 (sábado) y 2023-01-08 (domingo) no son sesión;
    # el siguiente día de sesión es 2023-01-09 (lunes)
    saturday_noon = pd.Timestamp("2023-01-07 12:00:00", tz="America/New_York")
    result = assign_to_session(pd.Series([saturday_noon]), closes)
    assert result.iloc[0] == pd.Timestamp("2023-01-09")


def test_holiday_publication_rolls_forward_past_mlk_day(closes: pd.Series) -> None:
    # 2023-01-16 (lunes) es MLK Day, no hay sesión; el mercado abre de nuevo
    # el 2023-01-17
    holiday_afternoon = pd.Timestamp("2023-01-16 15:00:00", tz="America/New_York")
    result = assign_to_session(pd.Series([holiday_afternoon]), closes)
    assert result.iloc[0] == pd.Timestamp("2023-01-17")


def test_friday_after_close_rolls_to_next_monday(closes: pd.Series) -> None:
    friday = pd.Timestamp("2023-01-06")
    close_friday = closes.loc[friday]
    just_after_friday_close = close_friday + pd.Timedelta(minutes=1)
    result = assign_to_session(pd.Series([just_after_friday_close]), closes)
    assert result.iloc[0] == pd.Timestamp("2023-01-09")


def test_timezone_conversion_matters() -> None:
    """El cierre se compara en hora de Nueva York (en junio, EDT = UTC-4),
    no tratando la hora UTC como si ya fuera la hora local del mercado."""
    closes = get_session_closes("2023-06-01", "2023-06-10")
    close_0601_utc = closes.loc[pd.Timestamp("2023-06-01")].tz_convert("UTC")

    just_before_close = close_0601_utc - pd.Timedelta(minutes=1)
    result = assign_to_session(pd.Series([just_before_close]), closes)
    assert result.iloc[0] == pd.Timestamp("2023-06-01")

    just_after_close = close_0601_utc + pd.Timedelta(minutes=1)
    result2 = assign_to_session(pd.Series([just_after_close]), closes)
    assert result2.iloc[0] == pd.Timestamp("2023-06-02")


def test_out_of_range_publication_returns_nat(closes: pd.Series) -> None:
    far_future = pd.Timestamp("2099-01-01", tz="America/New_York")
    result = assign_to_session(pd.Series([far_future]), closes)
    assert pd.isna(result.iloc[0])


def test_no_leakage_invariant_on_random_timestamps() -> None:
    """Para cualquier hora de publicación, la sesión asignada cierra en ese
    momento o después, y la sesión anterior cerró estrictamente antes: la
    noticia no se asigna ni a una sesión pasada ni a una posterior a la que
    le corresponde."""
    closes = get_session_closes("2020-01-01", "2020-12-31")
    session_list = closes.index.to_list()

    rng = pd.Series(pd.date_range("2020-01-01", "2020-12-30", freq="7h", tz="UTC"))
    assigned = assign_to_session(rng, closes)

    for pub, sess in zip(rng, assigned):
        if pd.isna(sess):
            continue
        close_t = closes.loc[sess].tz_convert("UTC")
        assert pub <= close_t, f"fuga: {pub} asignado a sesión {sess} que cerró antes"

        pos = session_list.index(sess)
        if pos > 0:
            prev_close = closes.loc[session_list[pos - 1]].tz_convert("UTC")
            assert pub > prev_close, f"{pub} debería pertenecer a una sesión anterior a {sess}"


def test_real_news_data_never_leaks() -> None:
    """Comprobación sobre los datos reales ya alineados: ninguna noticia
    tiene asignada una sesión que cerró antes de su publicación, lo que
    sería una fuga de información."""
    from pathlib import Path

    from src.alignment.align_news import align_ticker_news
    from src.ingestion.download_prices import TICKERS

    interim = Path(__file__).resolve().parents[1] / "data" / "interim"
    for ticker in TICKERS:
        path = interim / f"{ticker}_news_clean.parquet"
        if not path.exists():
            pytest.skip(f"{path} no existe todavía (ejecutar align_news primero)")

        df = align_ticker_news(ticker)
        assert "session_date" in df.columns

        cal_start = (df["date_publicacion"].min() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        cal_end = (df["date_publicacion"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        closes = get_session_closes(cal_start, cal_end)

        valid = df.dropna(subset=["session_date"])
        close_of_session = valid["session_date"].map(lambda d: closes.loc[pd.Timestamp(d)])
        close_utc = pd.DatetimeIndex(close_of_session).tz_convert("UTC")
        pub_utc = pd.DatetimeIndex(valid["date_publicacion"]).tz_convert("UTC")
        assert (pub_utc <= close_utc).all(), f"{ticker}: hay noticias con fuga hacia el futuro"
