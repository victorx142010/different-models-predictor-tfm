"""Pruebas del splitter walk-forward: cuatro folds con ventana expansiva,
embargo de 5 sesiones entre entrenamiento y validación y holdout 2022-2023
separado de todos los folds.
"""

import pandas as pd
import pytest

from src.validation.splitter import WalkForwardSplitter


@pytest.fixture(scope="module")
def splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter()


def test_four_folds_generated(splitter: WalkForwardSplitter) -> None:
    assert [f.name for f in splitter.folds] == ["fold1", "fold2", "fold3", "fold4"]


def test_train_start_fixed_expanding_window(splitter: WalkForwardSplitter) -> None:
    starts = {f.train_start for f in splitter.folds}
    assert len(starts) == 1  # ventana expandida: mismo inicio, fin creciente


def test_train_end_strictly_increases(splitter: WalkForwardSplitter) -> None:
    ends = [f.train_end for f in splitter.folds]
    assert ends == sorted(ends)
    assert len(set(ends)) == 4


def test_val_years_match_spec(splitter: WalkForwardSplitter) -> None:
    expected_years = [2018, 2019, 2020, 2021]
    for fold, year in zip(splitter.folds, expected_years):
        assert fold.val_start.year == year
        assert fold.val_end.year == year


def test_embargo_is_exactly_5_sessions(splitter: WalkForwardSplitter) -> None:
    for fold in splitter.folds:
        pos_start = splitter._sessions.get_loc(fold.embargo_start)
        pos_end = splitter._sessions.get_loc(fold.embargo_end)
        assert pos_end - pos_start + 1 == 5


def test_no_overlap_train_embargo_val(splitter: WalkForwardSplitter) -> None:
    for fold in splitter.folds:
        assert fold.train_end < fold.embargo_start
        assert fold.embargo_end < fold.val_start
        assert fold.val_start <= fold.val_end


def test_embargo_immediately_follows_train_end(splitter: WalkForwardSplitter) -> None:
    for fold in splitter.folds:
        pos_train_end = splitter._sessions.get_loc(fold.train_end)
        pos_embargo_start = splitter._sessions.get_loc(fold.embargo_start)
        assert pos_embargo_start == pos_train_end + 1


def test_val_immediately_follows_embargo(splitter: WalkForwardSplitter) -> None:
    for fold in splitter.folds:
        pos_embargo_end = splitter._sessions.get_loc(fold.embargo_end)
        pos_val_start = splitter._sessions.get_loc(fold.val_start)
        assert pos_val_start == pos_embargo_end + 1


def test_slice_train_val_disjoint_on_real_dataframe(splitter: WalkForwardSplitter) -> None:
    dates = pd.date_range("2015-01-01", "2021-12-31", freq="B")
    df = pd.DataFrame({"date": dates, "value": range(len(dates))})

    for fold in splitter.folds:
        train_df = splitter.slice_train(df, fold)
        val_df = splitter.slice_val(df, fold)
        assert set(train_df["date"]).isdisjoint(set(val_df["date"]))
        assert train_df["date"].max() == fold.train_end
        assert val_df["date"].min() == fold.val_start
        assert val_df["date"].max() == fold.val_end


def test_holdout_disjoint_from_all_folds(splitter: WalkForwardSplitter) -> None:
    holdout_start = splitter.holdout_start
    for fold in splitter.folds:
        assert fold.train_end < holdout_start
        assert fold.val_end < holdout_start


def test_slice_holdout_returns_only_2022_2023(splitter: WalkForwardSplitter) -> None:
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="B")
    df = pd.DataFrame({"date": dates, "value": range(len(dates))})
    holdout_df = splitter.slice_holdout(df)
    assert holdout_df["date"].min() >= pd.Timestamp("2022-01-01")
    assert holdout_df["date"].max() <= pd.Timestamp("2023-12-31")
