"""Validación walk-forward con embargo: el modelo se entrena siempre con el
pasado y se valida en un periodo posterior que no ha visto.

Se generan cuatro folds anuales. El entrenamiento empieza siempre el 2015-01-02
y se amplía año a año: el fold 1 entrena hasta 2017 y valida en 2018, el fold 2
entrena hasta 2018 y valida en 2019, y así hasta el fold 4, que valida en 2021.
Repetir la validación en varios años evita que el resultado dependa de que un
año concreto fuera fácil o difícil.

Entre entrenamiento y validación se deja un embargo de `embargo_sessions`
sesiones de la NYSE (festivos y fines de semana no cuentan). La etiqueta del
día t usa precios hasta t+h, así que sin ese hueco las últimas etiquetas de
entrenamiento se construirían con precios del periodo de validación. Por eso el
embargo debe cubrir al menos el horizonte: por defecto son 5 sesiones, y los
scripts que evalúan horizontes mayores lo amplían a max(5, h).

El holdout (2022-2023) se define aquí, pero queda fuera de los folds: solo se
usa una vez, en la evaluación final.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_market_calendars as mcal

SAMPLE_START = "2015-01-01"
HOLDOUT_START = "2022-01-01"
HOLDOUT_END = "2023-12-31"
DEFAULT_EMBARGO_SESSIONS = 5  # valor por defecto; con horizontes mayores se usa max(5, h)
FOLD_VAL_YEARS = [2018, 2019, 2020, 2021]


def nyse_sessions(start: str, end: str) -> pd.DatetimeIndex:
    """Sesiones de negociación de la NYSE entre dos fechas, sin zona
    horaria."""
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()


@dataclass(frozen=True)
class Fold:
    """Fechas de un fold: entrenamiento, embargo (sesiones que se descartan)
    y validación."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp


class WalkForwardSplitter:
    """Calcula los cuatro folds y recorta cualquier DataFrame con columna de
    fecha al tramo que corresponda.

    Ejemplo:
        splitter = WalkForwardSplitter()
        for fold in splitter.folds:
            train_df = splitter.slice_train(df, fold)
            val_df = splitter.slice_val(df, fold)

        # el holdout se recorta aparte y solo se usa en la evaluación final
        holdout_df = splitter.slice_holdout(df)
    """

    def __init__(
        self,
        sample_start: str = SAMPLE_START,
        holdout_start: str = HOLDOUT_START,
        holdout_end: str = HOLDOUT_END,
        embargo_sessions: int = DEFAULT_EMBARGO_SESSIONS,
        fold_val_years: list[int] | None = None,
    ) -> None:
        self.sample_start = pd.Timestamp(sample_start)
        self.holdout_start = pd.Timestamp(holdout_start)
        self.holdout_end = pd.Timestamp(holdout_end)
        self.embargo_sessions = embargo_sessions
        self.fold_val_years = fold_val_years or FOLD_VAL_YEARS

        last_val_year = max(self.fold_val_years)
        self._sessions = nyse_sessions(sample_start, f"{last_val_year}-12-31")
        self.folds: list[Fold] = self._build_folds()

    def _last_session_on_or_before(self, date: pd.Timestamp) -> pd.Timestamp:
        eligible = self._sessions[self._sessions <= date]
        if eligible.empty:
            raise ValueError(f"No hay sesiones NYSE en/antes de {date.date()}")
        return eligible[-1]

    def _first_session_on_or_after(self, date: pd.Timestamp) -> pd.Timestamp:
        eligible = self._sessions[self._sessions >= date]
        if eligible.empty:
            raise ValueError(f"No hay sesiones NYSE en/después de {date.date()}")
        return eligible[0]

    def _build_folds(self) -> list[Fold]:
        train_start = self._first_session_on_or_after(self.sample_start)
        folds = []
        for i, val_year in enumerate(self.fold_val_years, start=1):
            train_end_year = val_year - 1
            train_end = self._last_session_on_or_before(pd.Timestamp(f"{train_end_year}-12-31"))

            train_end_pos = self._sessions.get_loc(train_end)
            embargo_slice = self._sessions[
                train_end_pos + 1 : train_end_pos + 1 + self.embargo_sessions
            ]
            if len(embargo_slice) != self.embargo_sessions:
                raise ValueError(
                    f"No hay suficientes sesiones para el embargo del fold {i} "
                    f"(se necesitan {self.embargo_sessions}, calendario cargado hasta "
                    f"{self._sessions[-1].date()})"
                )
            embargo_start, embargo_end = embargo_slice[0], embargo_slice[-1]

            val_start_pos = train_end_pos + 1 + self.embargo_sessions
            val_start = self._sessions[val_start_pos]
            val_end = self._last_session_on_or_before(pd.Timestamp(f"{val_year}-12-31"))

            folds.append(
                Fold(
                    name=f"fold{i}",
                    train_start=train_start,
                    train_end=train_end,
                    embargo_start=embargo_start,
                    embargo_end=embargo_end,
                    val_start=val_start,
                    val_end=val_end,
                )
            )
        return folds

    @staticmethod
    def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, date_col: str) -> pd.DataFrame:
        mask = (df[date_col] >= start) & (df[date_col] <= end)
        return df.loc[mask].copy()

    def slice_train(self, df: pd.DataFrame, fold: Fold, date_col: str = "date") -> pd.DataFrame:
        return self._slice(df, fold.train_start, fold.train_end, date_col)

    def slice_val(self, df: pd.DataFrame, fold: Fold, date_col: str = "date") -> pd.DataFrame:
        return self._slice(df, fold.val_start, fold.val_end, date_col)

    def slice_holdout(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Recorta el holdout (2022-2023). Solo se usa en la evaluación
        final, nunca para elegir arquitectura ni hiperparámetros."""
        return self._slice(df, self.holdout_start, self.holdout_end, date_col)

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "fold": f.name,
                "train_start": f.train_start.date(),
                "train_end": f.train_end.date(),
                "embargo_start": f.embargo_start.date(),
                "embargo_end": f.embargo_end.date(),
                "val_start": f.val_start.date(),
                "val_end": f.val_end.date(),
                "n_train_sessions": self._sessions.get_loc(f.train_end)
                - self._sessions.get_loc(f.train_start)
                + 1,
                "n_val_sessions": self._sessions.get_loc(f.val_end)
                - self._sessions.get_loc(f.val_start)
                + 1,
            }
            for f in self.folds
        ]
        return pd.DataFrame(rows)


if __name__ == "__main__":
    # Ejecutar el fichero directamente imprime las fechas de cada fold para
    # revisarlas.
    splitter = WalkForwardSplitter()
    print(splitter.summary().to_string(index=False))
    print(f"\nHoldout (reservado para la evaluación final): {HOLDOUT_START} a {HOLDOUT_END}")
