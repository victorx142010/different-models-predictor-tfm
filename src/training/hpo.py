"""Búsqueda de hiperparámetros con Optuna para las cinco variantes.

Para cada variante se lanza un estudio que prueba combinaciones de
hiperparámetros (`suggest_params`), entrena el modelo con cada una en los folds
walk-forward y se queda con la de menor pérdida de validación.

Cada combinación se evalúa con la pérdida media sobre todos los pares activo ×
fold a la vez, no sobre una partición concreta. Así se evita elegir
hiperparámetros que solo funcionan en un fold o en un activo, algo
especialmente importante porque la cobertura de noticias es muy desigual entre
activos y años.

Dos decisiones mantienen la búsqueda manejable:
- La ventana L se fija en 20 y no se busca: probar otros valores obligaría a
  reconstruir todas las secuencias precalculadas para cada uno. Su elección se
  comprobó aparte (feature_engineering_diagnostics.py).
- El GARCH es siempre (1,1) y no forma parte de la búsqueda; en su ajuste solo
  se elige la distribución por AIC (garch_features.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from src.fusion_model.base_model import MultiModalModel, UncertaintyWeightedLoss, multitask_loss
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.quant_module.lstm import QUANT_INPUT_DIM
from src.training.sequence_dataset import (
    build_dataset,
    build_quant_feature_table,
    load_daily_agg,
    load_variable_set,
    targets_with_horizon,
)
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import TENSOR_KEYS as TENSOR_KEYS_ALL
from src.training.train_cross_attention import to_tensors
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS, WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

L = 20
HORIZON = 1
BATCH_SIZE = 64
MAX_EPOCHS = 60  # menos que las 100 del entrenamiento final: cada trial debe ser barato
PATIENCE = 8
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
SEED = 42

LAMBDA_RATIOS = {"1:1": (1.0, 1.0), "2:1": (2.0, 1.0), "1:2": (1.0, 2.0)}
LOSS_WEIGHTING_OPTIONS = ("1:1", "2:1", "1:2", "learned_uncertainty")


def precompute_all_data(
    device: str,
    garch_suffix: str = "",
    tickers: tuple[str, ...] = tuple(TICKERS),
    horizon: int = HORIZON,
    lookback: int = L,
    include_volume: bool = False,
    include_technical: bool = False,
    include_garch_forecast: bool = False,
) -> list[dict]:
    """Construye una sola vez los pares entrenamiento/validación de cada
    activo y fold, incluido el conjunto variable de noticias (lo usa
    cross_attention; el resto lo ignora).

    Por defecto usa `TICKERS`, h=1 y L=20, pero `tickers`, `horizon` y
    `lookback` son parámetros, así que sirve para cualquier otro universo u
    horizonte.

    Las demás opciones se pasan a `build_quant_feature_table`:
    `garch_suffix="gjr"` usa las variables GJR-GARCH, y `include_volume`,
    `include_technical` e `include_garch_forecast` añaden variables
    candidatas para los diagnósticos. El pronóstico GARCH se calcula al
    mismo horizonte que se predice."""
    # el embargo debe cubrir al menos el horizonte; si no, el final del
    # entrenamiento y el inicio de la validación compartirían información en
    # horizontes mayores de 5 sesiones (el mismo criterio que main.py)
    embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
    splitter = WalkForwardSplitter(embargo_sessions=embargo)
    all_data = []
    for ticker in tickers:
        quant_df = build_quant_feature_table(
            ticker,
            garch_suffix=garch_suffix,
            include_volume=include_volume,
            include_technical=include_technical,
            include_garch_forecast=include_garch_forecast,
            garch_forecast_horizon=horizon if include_garch_forecast else None,
        )
        daily_agg = load_daily_agg(ticker)
        variable_set = load_variable_set(ticker)
        targets = targets_with_horizon(ticker, horizon)
        prices = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )

        for fold in splitter.folds:
            train_dates = splitter.slice_train(prices, fold)["date"]
            val_dates = splitter.slice_val(prices, fold)["date"]
            train_ds = build_dataset(
                train_dates, quant_df, daily_agg, targets, lookback, horizon, variable_set=variable_set
            )
            val_ds = build_dataset(
                val_dates, quant_df, daily_agg, targets, lookback, horizon, variable_set=variable_set
            )
            if train_ds is None or val_ds is None:
                continue

            train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
            train_ds = dict(train_ds, quant_seq=train_q)
            val_ds = dict(val_ds, quant_seq=val_q)

            all_data.append(
                {
                    "ticker": ticker,
                    "fold": fold.name,
                    "train": to_tensors(train_ds, device),
                    "val": to_tensors(val_ds, device),
                }
            )
    return all_data


def build_model(
    variant: str,
    params: dict,
    use_embedding_layernorm: bool = False,
    quant_input_dim: int = QUANT_INPUT_DIM,
    text_proj_dim: int | None = None,
    dense_hidden: int | None = None,
) -> MultiModalModel:
    """Construye un MultiModalModel a partir de un diccionario de
    hiperparámetros como el de `suggest_params` o el del JSON guardado.

    `use_embedding_layernorm` y `quant_input_dim` se pasan tal cual al
    modelo; este último debe coincidir con el número de variables de
    `quant_seq` (4 si se añade el volumen). `text_proj_dim` y `dense_hidden`
    no forman parte de la búsqueda: si valen None se usa el 64 por defecto,
    y solo se cambian en el diagnóstico de dimensiones
    (architecture_hparam_diagnostics.py)."""
    kwargs = dict(
        variant=variant,
        quant_input_dim=quant_input_dim,
        lstm_hidden_size=params["hidden_size"],
        lstm_num_layers=params["num_layers"],
        lstm_dropout=params["dropout"] if params["num_layers"] > 1 else 0.0,
        dense_dropout=params["dropout"],
        use_embedding_layernorm=use_embedding_layernorm,
    )
    if text_proj_dim is not None:
        kwargs["text_proj_dim"] = text_proj_dim
    if dense_hidden is not None:
        kwargs["dense_hidden"] = dense_hidden
    if variant == "cross_attention":
        kwargs["attn_d_k"] = params["d_k"]
        kwargs["attn_n_heads"] = params["n_heads"]
    return MultiModalModel(**kwargs)


def train_and_eval_fold(
    variant: str,
    params: dict,
    train_t: dict,
    val_t: dict,
    device: str,
    use_embedding_layernorm: bool = False,
    text_proj_dim: int | None = None,
) -> float:
    """Entrena un modelo con una combinación de hiperparámetros en un fold y
    devuelve solo su pérdida de validación: durante la búsqueda no hace
    falta guardar el modelo, solo comparar combinaciones.

    `quant_input_dim` se deduce de los datos, para que la búsqueda funcione
    también con más variables de entrada (por ejemplo, en `run_ablation` de
    feature_engineering_diagnostics.py)."""
    torch.manual_seed(SEED)
    quant_input_dim = train_t["quant_seq"].shape[-1]
    model = build_model(
        variant, params, use_embedding_layernorm, quant_input_dim, text_proj_dim=text_proj_dim
    ).to(device)

    weighting = params["loss_weighting"]
    if weighting == "learned_uncertainty":
        loss_module: UncertaintyWeightedLoss | None = UncertaintyWeightedLoss().to(device)
        opt_params = list(model.parameters()) + list(loss_module.parameters())
        lambda_dir = lambda_vol = None
    else:
        loss_module = None
        lambda_dir, lambda_vol = LAMBDA_RATIOS[weighting]
        opt_params = list(model.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=params["learning_rate"], weight_decay=WEIGHT_DECAY)

    def compute_loss(out: dict, y_dir: torch.Tensor, y_vol: torch.Tensor) -> torch.Tensor:
        if loss_module is not None:
            loss, _ = loss_module(out["direction_logit"], out["vol_pred"], y_dir, y_vol)
        else:
            loss, _ = multitask_loss(
                out["direction_logit"], out["vol_pred"], y_dir, y_vol, lambda_dir, lambda_vol
            )
        return loss

    n = train_t["quant_seq"].shape[0]
    idx = np.arange(n)
    best_val_loss = float("inf")
    best_scoring_loss = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(idx)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start : start + BATCH_SIZE]
            batch = {k: train_t[k][batch_idx] for k in TENSOR_KEYS_ALL}
            y_dir = train_t["y_direction"][batch_idx]
            y_vol = train_t["y_vol"][batch_idx]

            optimizer.zero_grad()
            out = model(batch)
            loss = compute_loss(out, y_dir, y_vol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
            out = model(val_batch)
            val_loss = compute_loss(out, val_t["y_direction"], val_t["y_vol"])
            # Se calcula aparte una pérdida de referencia con pesos fijos 1:1,
            # sea cual sea el esquema que prueba el trial. Hace falta porque
            # «learned_uncertainty» suma los términos s_dir y s_vol, que no
            # tienen un mínimo acotado como BCE o Huber: comparando por la
            # pérdida sin corregir, Optuna favorecería ese esquema solo porque
            # su valor es menor, no porque prediga mejor. Con la pérdida de
            # referencia, todos los trials se comparan con la misma medida.
            scoring_loss, _ = multitask_loss(
                out["direction_logit"], out["vol_pred"], val_t["y_direction"], val_t["y_vol"], 1.0, 1.0
            )
        val_loss_val = val_loss.item()

        if val_loss_val < best_val_loss - 1e-5:
            best_val_loss = val_loss_val
            best_scoring_loss = scoring_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return best_scoring_loss


def suggest_params(trial: optuna.Trial, variant: str) -> dict:
    """Espacio de búsqueda: qué hiperparámetros se prueban y con qué
    valores. cross_attention tiene dos más (d_k y n_heads), propios del
    mecanismo de atención."""
    params = {
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "dropout": trial.suggest_float("dropout", 0.2, 0.4),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-4, 1e-4]),
        "loss_weighting": trial.suggest_categorical("loss_weighting", LOSS_WEIGHTING_OPTIONS),
    }
    if variant == "cross_attention":
        params["d_k"] = trial.suggest_categorical("d_k", [32, 64, 128])
        params["n_heads"] = trial.suggest_categorical("n_heads", [2, 4])
    return params


def make_objective(
    variant: str,
    all_data: list[dict],
    device: str,
    use_embedding_layernorm: bool = False,
    text_proj_dim: int | None = None,
):
    """Función objetivo que Optuna minimiza para una variante. En cada trial
    entrena y evalúa la combinación en todos los pares activo × fold, uno
    tras otro, y comunica la media parcial tras cada uno (`trial.report`).
    Si el trial va claramente peor que los anteriores, Optuna lo corta antes
    de terminar (`trial.should_prune`) y no se pierde tiempo.

    `text_proj_dim` y `use_embedding_layernorm` se pasan a cada fold;
    `text_proj_dim` solo se fija para buscar hiperparámetros propios de otra
    dimensión de proyección."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, variant)
        fold_losses = []
        for i, d in enumerate(all_data):
            val_loss = train_and_eval_fold(
                variant, params, d["train"], d["val"], device, use_embedding_layernorm, text_proj_dim
            )
            fold_losses.append(val_loss)
            trial.report(float(np.mean(fold_losses)), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_losses))

    return objective


def run_study(
    variant: str,
    all_data: list[dict],
    device: str,
    n_trials: int,
    use_embedding_layernorm: bool = False,
    text_proj_dim: int | None = None,
) -> optuna.Study:
    """Lanza el estudio completo de una variante, con TPESampler (búsqueda
    bayesiana, más eficiente que probar combinaciones al azar) y
    MedianPruner (corta un trial si va peor que la mediana de los anteriores
    en el mismo punto)."""
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(
        make_objective(variant, all_data, device, use_embedding_layernorm, text_proj_dim),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study
