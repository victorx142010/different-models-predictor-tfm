"""Pruebas del escalado de las variables cuantitativas: el escalador se ajusta
solo con entrenamiento y se aplica tal cual a validación.
"""

import numpy as np

from src.training.train_control_variants import scale_quant_seq


def test_scaler_fit_only_on_train_not_val() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, scale=1.0, size=(50, 5, 3)).astype("float32")
    # validación con una media y una escala muy distintas: si el escalador se
    # ajustara, aunque fuera en parte, con validación, el entrenamiento
    # escalado cambiaría; se comprueba que no cambia
    val_normal = rng.normal(loc=0.0, scale=1.0, size=(10, 5, 3)).astype("float32")
    val_shifted = rng.normal(loc=500.0, scale=50.0, size=(10, 5, 3)).astype("float32")

    train_scaled_a, _ = scale_quant_seq(train.copy(), val_normal)
    train_scaled_b, _ = scale_quant_seq(train.copy(), val_shifted)

    assert np.allclose(train_scaled_a, train_scaled_b)


def test_scaled_train_has_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(loc=10.0, scale=3.0, size=(200, 4, 3)).astype("float32")
    val = rng.normal(loc=10.0, scale=3.0, size=(20, 4, 3)).astype("float32")

    train_scaled, val_scaled = scale_quant_seq(train, val)

    flat = train_scaled.reshape(-1, 3)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-5)
    assert train_scaled.shape == train.shape
    assert val_scaled.shape == val.shape


def test_val_scaled_with_train_statistics() -> None:
    """Si validación tiene otra media que entrenamiento, al escalarla con
    las estadísticas de entrenamiento no debe quedar centrada en 0. Confirma
    que no se reajusta nada con validación."""
    rng = np.random.default_rng(2)
    train = rng.normal(loc=0.0, scale=1.0, size=(100, 3, 3)).astype("float32")
    val = rng.normal(loc=5.0, scale=1.0, size=(20, 3, 3)).astype("float32")

    _, val_scaled = scale_quant_seq(train, val)
    assert val_scaled.reshape(-1, 3).mean() > 3.0
