"""Indicadores técnicos RSI y ATR, usados solo en el diagnóstico de
feature_engineering_diagnostics.py para comprobar si añadirlos a la rama
cuantitativa mejora el modelo. No forman parte del pipeline principal.

Se eligen estos dos porque miden cosas distintas: el RSI mide el momentum y el
ATR la volatilidad relativa.

Se calculan sobre el precio sin ajustar (high, low, close), que es la
convención de estos indicadores: con el precio ajustado por dividendos, el ATR
mostraría saltos artificiales en los días ex-dividendo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Wilder: media exponencial (alpha = 1/window) de las ganancias
    y de las pérdidas, que es su definición clásica. Sustituir las pérdidas
    nulas por un valor mínimo evita dividir por cero en rachas solo
    alcistas, en las que el RSI tiende a 100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR de Wilder dividido por el precio de cierre, para medir la
    volatilidad en términos relativos. Sin esa normalización no se podrían
    comparar activos de precio muy distinto, como SPY (cientos de dólares) y
    KO (decenas)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return atr / close


def compute_technical_indicators(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Devuelve las columnas date, rsi_14 y atr_14_norm a partir de los
    precios limpios de un activo. Las primeras sesiones de la serie (de
    2010, muy anteriores a cualquier fold) quedan en NaN hasta tener
    historia suficiente; se descartan en `build_quant_feature_table`."""
    return pd.DataFrame(
        {
            "date": prices["date"],
            "rsi_14": compute_rsi(prices["close"], window),
            "atr_14_norm": compute_atr(prices["high"], prices["low"], prices["close"], window),
        }
    )
