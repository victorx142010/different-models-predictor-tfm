"""Métricas de significancia financiera y estadística, que completan a
accuracy, F1, RMSE y MAE: si la señal de dirección es aprovechable
económicamente y si el acierto del modelo se distingue del azar o de un modelo
rival.

El módulo es independiente del entrenamiento: ninguna de estas métricas
interviene en la función de pérdida ni en la búsqueda de hiperparámetros,
porque eso orientaría la selección hacia lo fácil de optimizar y no hacia lo
correcto. Se aplican después, sobre predicciones ya generadas, así que reciben
arrays de predicciones o errores y no un modelo.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

TRADING_DAYS_PER_YEAR = 252


def direction_to_strategy_returns(
    direction_pred: np.ndarray, actual_returns: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Convierte una señal de dirección en el retorno de una estrategia
    direccional simétrica: posición larga si se predice subida y corta si se
    predice bajada. Es lo que permite calcular Sharpe y Sortino sobre un
    clasificador. `threshold` (0,5 por defecto) es el corte sobre
    `direction_pred`; si ya llega binarizada, basta con que quede entre las
    dos clases."""
    position = np.where(np.asarray(direction_pred) > threshold, 1.0, -1.0)
    return position * np.asarray(actual_returns)


def sharpe_ratio(
    strategy_returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Ratio de Sharpe anualizado: rentabilidad media en exceso de la tasa
    libre de riesgo dividida por su volatilidad y escalada por
    sqrt(periods_per_year). Con retornos acumulados a h sesiones conviene
    usar 252/h para no sobreanualizar."""
    excess = np.asarray(strategy_returns, dtype=float) - risk_free_rate
    std = excess.std(ddof=1)
    # se usa una tolerancia en lugar de == 0: con retornos constantes, el
    # redondeo rara vez deja la varianza en cero exacto, y dividir por ese
    # residuo daría un Sharpe enorme y artificial
    if std < 1e-12 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(
    strategy_returns: np.ndarray, target_return: float = 0.0, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Ratio de Sortino: como el de Sharpe, pero sustituye la volatilidad
    total por la semidesviación a la baja. Solo penalizan los retornos por
    debajo de `target_return`, porque la variabilidad al alza no es un
    riesgo para quien sigue la señal."""
    returns = np.asarray(strategy_returns, dtype=float)
    downside = np.minimum(returns - target_return, 0.0)
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev < 1e-12 or np.isnan(downside_dev):
        return 0.0
    return float((returns.mean() - target_return) / downside_dev * np.sqrt(periods_per_year))


def pesaran_timmermann_test(direction_pred: np.ndarray, direction_actual: np.ndarray) -> dict:
    """Test de Pesaran y Timmermann: comprueba si el acierto direccional del
    modelo es mayor que el de una señal independiente del mercado con las
    mismas proporciones de subidas y bajadas.

    Compara el acierto observado P con P* = p_real·p_modelo + (1 -
    p_real)·(1 - p_modelo), el acierto que tendría alguien que dijera «sube»
    con la misma frecuencia que el modelo, pero al azar. Así no premia el
    sesgo alcista o bajista del periodo: un modelo que dice siempre «sube»
    tiene P = P*.

    Devuelve el estadístico (normal estándar si no hay capacidad
    predictiva), su p-valor unilateral y las cantidades intermedias."""
    y = np.asarray(direction_actual, dtype=float)
    x = np.asarray(direction_pred, dtype=float)
    n = len(y)
    if n == 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "accuracy": float("nan"), "expected_under_independence": float("nan")}

    p_hat = float(np.mean((x > 0.5).astype(float) == y))
    p_y = float(y.mean())
    p_x = float((x > 0.5).mean())
    p_star = p_y * p_x + (1 - p_y) * (1 - p_x)

    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = (
        ((2 * p_y - 1) ** 2) * p_x * (1 - p_x) / n
        + ((2 * p_x - 1) ** 2) * p_y * (1 - p_y) / n
        + 4 * p_x * p_y * (1 - p_x) * (1 - p_y) / n**2
    )
    var_diff = var_p_hat - var_p_star
    if var_diff <= 0:
        # caso degenerado (p_x o p_y en 0 o 1): no hay varianza que contrastar
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "accuracy": p_hat,
            "expected_under_independence": p_star,
        }

    statistic = (p_hat - p_star) / np.sqrt(var_diff)
    p_value = float(1 - stats.norm.cdf(statistic))
    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "accuracy": p_hat,
        "expected_under_independence": p_star,
    }


def _autocovariance(d: np.ndarray, lag: int) -> float:
    """Autocovarianza muestral de `d` en el retardo `lag`, normalizada por n
    (no por n - lag), como en el estimador de varianza a largo plazo de
    Diebold-Mariano."""
    n = len(d)
    d_centered = d - d.mean()
    if lag == 0:
        return float(np.sum(d_centered * d_centered) / n)
    return float(np.sum(d_centered[lag:] * d_centered[: n - lag]) / n)


def diebold_mariano_test(errors_model: np.ndarray, errors_baseline: np.ndarray, h: int = 1, power: int = 2) -> dict:
    """Test de Diebold y Mariano con la corrección para muestras pequeñas de
    Harvey, Leybourne y Newbold: comprueba si la diferencia de precisión
    entre dos modelos es significativa, a partir de la diferencia de
    pérdidas en cada punto, d_t = |e_1,t|^power - |e_2,t|^power.

    `h` es el horizonte de los pronósticos: sus errores están
    correlacionados hasta el retardo h-1, así que determina cuántas
    autocovarianzas entran en la varianza (por el mismo motivo por el que el
    walk-forward usa un embargo). `power` es 1 para el error absoluto y 2
    para el cuadrático (por defecto).

    No se calcula automáticamente para cada variante: compara dos series de
    errores ya guardadas y se aplica después, a los pares de modelos que se
    quieren comparar."""
    e1 = np.asarray(errors_model, dtype=float)
    e2 = np.asarray(errors_baseline, dtype=float)
    if len(e1) != len(e2):
        raise ValueError(f"errors_model y errors_baseline deben tener la misma longitud ({len(e1)} vs {len(e2)})")
    n = len(e1)

    d = np.abs(e1) ** power - np.abs(e2) ** power
    d_bar = float(d.mean())

    long_run_var = _autocovariance(d, 0)
    for lag in range(1, h):
        long_run_var += 2 * _autocovariance(d, lag)
    var_d_bar = long_run_var / n

    if var_d_bar <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "mean_loss_diff": d_bar}

    dm_stat = d_bar / np.sqrt(var_d_bar)
    # Corrección para muestras pequeñas de Harvey, Leybourne y Newbold: el
    # estadístico DM asintótico sobreestima la significancia cuando la muestra
    # es pequeña respecto al horizonte h. Este factor lo corrige, y a cambio se
    # compara con una t de Student de n-1 grados de libertad en lugar de una
    # normal.
    correction = np.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n)
    dm_stat_corrected = float(dm_stat * correction)

    p_value = float(2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), df=n - 1)))
    return {"statistic": dm_stat_corrected, "p_value": p_value, "mean_loss_diff": d_bar}


def classification_agreement_metrics(
    direction_pred_proba: np.ndarray, direction_pred_binary: np.ndarray, direction_actual: np.ndarray
) -> dict:
    """AUC de la curva ROC, que necesita la probabilidad continua, y Kappa
    de Cohen, que necesita la clase ya binarizada, entre la predicción del
    modelo y la dirección real."""
    y_true = np.asarray(direction_actual)
    try:
        auc = float(roc_auc_score(y_true, direction_pred_proba))
    except ValueError:
        # el AUC no está definido si y_true tiene una sola clase
        auc = float("nan")
    kappa = float(cohen_kappa_score(y_true, direction_pred_binary))
    return {"roc_auc": auc, "cohen_kappa": kappa}


def compute_all_financial_metrics(
    direction_pred_proba: np.ndarray,
    direction_actual: np.ndarray,
    actual_returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
    threshold: float = 0.5,
) -> dict:
    """Calcula de una vez Sharpe, Sortino, el test de Pesaran-Timmermann, el
    AUC y Kappa para un conjunto de predicciones, como columnas que se
    añaden a una tabla de métricas (ver holdout_evaluation.py). No incluye
    Diebold-Mariano, que compara dos modelos.

    `threshold` (0,5 por defecto) es el corte sobre `direction_pred_proba`.
    No afecta al AUC, que no depende de ningún umbral."""
    direction_pred_binary = (np.asarray(direction_pred_proba) > threshold).astype(float)
    strategy_returns = direction_to_strategy_returns(direction_pred_binary, actual_returns, threshold=0.5)
    pt = pesaran_timmermann_test(direction_pred_binary, direction_actual)
    agreement = classification_agreement_metrics(direction_pred_proba, direction_pred_binary, direction_actual)
    return {
        "sharpe_ratio": sharpe_ratio(strategy_returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(strategy_returns, risk_free_rate, periods_per_year),
        "pt_statistic": pt["statistic"],
        "pt_p_value": pt["p_value"],
        **agreement,
    }


THRESHOLD_WARMUP_SESSIONS = 20


def expanding_calibrated_threshold(
    proba: np.ndarray,
    positive_rate: float,
    warmup_threshold: float,
    min_sessions: int = THRESHOLD_WARMUP_SESSIONS,
) -> np.ndarray:
    """Umbral de decisión causal: para cada sesión t devuelve el corte que
    reproduce `positive_rate` sobre las probabilidades observadas hasta t,
    incluida. Nunca usa sesiones posteriores. `proba` debe venir en orden
    cronológico.

    Por qué no se traslada el umbral de desarrollo: el modelo final,
    entrenado con todo 2015-2021, da probabilidades en otra escala que los
    modelos de fold con los que se calibró. Un corte de 0,74 en desarrollo
    puede equivaler a 0,83 en el modelo final, y aplicado tal cual
    clasificaría casi todo como subida. Lo que sí se transfiere entre
    escalas es la tasa de clasificación positiva.

    Por qué no se calcula el cuantil con todo el holdout de una vez: el
    umbral del primer día dependería de probabilidades de hasta dos años
    después. No es una fuga de etiquetas, pero sí una regla de decisión que
    mira al futuro, algo imposible en producción.

    Durante las primeras `min_sessions` sesiones no hay muestra suficiente
    para un cuantil y se usa `warmup_threshold`, el umbral calibrado en
    desarrollo. Con 10, 20 o 60 sesiones de calentamiento las conclusiones
    no cambian. En la última sesión el umbral coincide con el cuantil de
    todo el holdout."""
    proba = np.asarray(proba, dtype=float)
    thresholds = np.empty(len(proba), dtype=float)
    for i in range(len(proba)):
        if i + 1 < min_sessions:
            thresholds[i] = warmup_threshold
        else:
            thresholds[i] = float(np.quantile(proba[: i + 1], 1.0 - positive_rate))
    return thresholds
