"""Pruebas de financial_metrics.py con casos construidos a mano, en los que el
resultado correcto se conoce de antemano. No evalúan el modelo: comprueban que
cada fórmula se comporta como se espera.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.financial_metrics import (
    classification_agreement_metrics,
    compute_all_financial_metrics,
    diebold_mariano_test,
    direction_to_strategy_returns,
    pesaran_timmermann_test,
    sharpe_ratio,
    sortino_ratio,
)


def test_direction_to_strategy_returns_flips_sign_on_down_prediction() -> None:
    direction_pred = np.array([1.0, 0.0, 1.0, 0.0])
    actual_returns = np.array([0.02, 0.02, -0.01, -0.01])
    result = direction_to_strategy_returns(direction_pred, actual_returns)
    # predicción de subida -> posición larga (mismo signo); predicción de
    # bajada -> posición corta (signo invertido)
    np.testing.assert_allclose(result, [0.02, -0.02, -0.01, 0.01])


def test_sharpe_ratio_matches_manual_calculation() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert np.isclose(sharpe_ratio(returns, periods_per_year=252), expected)


def test_sharpe_ratio_zero_when_constant_returns() -> None:
    # desviación típica nula: por construcción no hay riesgo que dividir
    assert sharpe_ratio(np.full(10, 0.01)) == 0.0


def test_sortino_ratio_ignores_upside_variability() -> None:
    # dos series con la misma media y muy distinta variabilidad al alza,
    # pero idéntica variabilidad a la baja: el Sortino debe salir igual
    returns_a = np.array([-0.01, -0.01, 0.02, 0.02])
    returns_b = np.array([-0.01, -0.01, 0.10, 0.10])  # mucha más varianza al alza
    sortino_a = sortino_ratio(returns_a)
    sortino_b = sortino_ratio(returns_b)
    # el numerador (media - target) sí cambia, pero el denominador
    # (semi-desviación a la baja) es idéntico en ambos casos
    downside_a = np.sqrt(np.mean(np.minimum(returns_a, 0.0) ** 2))
    downside_b = np.sqrt(np.mean(np.minimum(returns_b, 0.0) ** 2))
    assert np.isclose(downside_a, downside_b)
    assert sortino_b > sortino_a  # b tiene mejor media con el mismo riesgo a la baja


def test_pesaran_timmermann_detects_perfect_predictor() -> None:
    rng = np.random.default_rng(0)
    direction_actual = rng.integers(0, 2, size=500).astype(float)
    result = pesaran_timmermann_test(direction_actual, direction_actual)  # predicción == realidad siempre
    assert result["accuracy"] == 1.0
    assert result["p_value"] < 0.001  # rechazo claro de "no hay predictibilidad"


def test_pesaran_timmermann_does_not_reject_independent_signal() -> None:
    rng = np.random.default_rng(1)
    direction_actual = rng.integers(0, 2, size=500).astype(float)
    direction_pred = rng.integers(0, 2, size=500).astype(float)  # generada aparte, sin relación
    result = pesaran_timmermann_test(direction_pred, direction_actual)
    assert result["p_value"] > 0.05  # no se rechaza independencia al 5%


def test_diebold_mariano_undefined_for_exactly_identical_errors() -> None:
    # con e1 == e2, d_t = |e1|^2 - |e2|^2 vale siempre cero: no hay varianza
    # que estimar, así que el test devuelve NaN y no un 0 que parecería un
    # resultado válido
    rng = np.random.default_rng(2)
    errors = rng.normal(0, 1, size=200)
    result = diebold_mariano_test(errors, errors, h=1)
    assert np.isnan(result["statistic"])
    assert np.isnan(result["p_value"])


def test_diebold_mariano_no_significant_difference_for_similar_models() -> None:
    # dos modelos con errores de la misma distribución (independientes, no
    # idénticos): no debería rechazarse que tienen la misma precisión
    rng = np.random.default_rng(6)
    errors_a = rng.normal(0, 1, size=300)
    errors_b = rng.normal(0, 1, size=300)
    result = diebold_mariano_test(errors_a, errors_b, h=1)
    assert result["p_value"] > 0.05


def test_diebold_mariano_favors_model_with_smaller_errors() -> None:
    rng = np.random.default_rng(3)
    errors_good = rng.normal(0, 0.5, size=300)
    errors_bad = rng.normal(0, 3.0, size=300)
    result = diebold_mariano_test(errors_good, errors_bad, h=1)
    assert result["statistic"] < 0  # d = |e_good|^2 - |e_bad|^2 < 0 en promedio
    assert result["p_value"] < 0.05


def test_diebold_mariano_rejects_mismatched_lengths() -> None:
    import pytest

    with pytest.raises(ValueError):
        diebold_mariano_test(np.zeros(10), np.zeros(11))


def test_classification_agreement_matches_sklearn_directly() -> None:
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 2, size=200).astype(float)
    proba = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=200), 0, 1)
    binary = (proba > 0.5).astype(float)

    result = classification_agreement_metrics(proba, binary, y_true)
    assert np.isclose(result["roc_auc"], roc_auc_score(y_true, proba))
    assert np.isclose(result["cohen_kappa"], cohen_kappa_score(y_true, binary))


def test_compute_all_financial_metrics_returns_expected_keys() -> None:
    rng = np.random.default_rng(5)
    n = 100
    y_true = rng.integers(0, 2, size=n).astype(float)
    proba = np.clip(y_true * 0.55 + rng.normal(0, 0.3, size=n), 0, 1)
    actual_returns = rng.normal(0, 0.01, size=n)

    result = compute_all_financial_metrics(proba, y_true, actual_returns)
    expected_keys = {"sharpe_ratio", "sortino_ratio", "pt_statistic", "pt_p_value", "roc_auc", "cohen_kappa"}
    assert expected_keys.issubset(result.keys())
