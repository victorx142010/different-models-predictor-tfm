"""Evaluación final, una sola vez, sobre el holdout 2022-2023.

Todas las decisiones de diseño (arquitectura, variantes e hiperparámetros) se
tomaron solo con los cuatro folds de 2015-2021. Este script no decide nada: con
los hiperparámetros ya elegidos, entrena cada modelo una vez sobre todo el
desarrollo (2015-01-02 a 2021-12-31) y lo evalúa una vez sobre 2022-2023. Si su
resultado se usara para cambiar algo, el holdout dejaría de ser una medida
honesta de generalización.

El GARCH de este tramo se ajusta igual que en los folds: se estima con todo
2015-2021 y se aplica, con los parámetros fijos, sobre 2022-2023.

Además de accuracy, F1, RMSE y MAE, calcula las métricas de
financial_metrics.py: Sharpe, Sortino, AUC, Kappa de Cohen y Pesaran-
Timmermann.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.evaluation.financial_metrics import (
    TRADING_DAYS_PER_YEAR,
    compute_all_financial_metrics,
    expanding_calibrated_threshold,
)
from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START
from src.quant_module.garch_features import _garch_segment
from src.training.hpo import TENSOR_KEYS_ALL
from src.training.hyperparameters_io import (
    hyperparameters_path_for,
    load_best_hyperparameters,
    require_hyperparameters_for,
    warn_if_provenance_mismatch,
)
from src.training.sequence_dataset import (
    build_dataset,
    build_quant_feature_table,
    load_daily_agg,
    load_variable_set,
    targets_with_horizon,
)
from src.training.train_control_variants import scale_quant_seq
from src.training.train_cross_attention import to_tensors
from src.training.train_final_models import train_and_eval_final
from src.validation.splitter import DEFAULT_EMBARGO_SESSIONS, Fold, WalkForwardSplitter, nyse_sessions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

L = 20


def build_holdout_fold(embargo_sessions: int) -> Fold:
    """Construye un `Fold` que representa el holdout: entrenamiento con todo
    2015-2021, el embargo correspondiente y validación sobre 2022-2023. Se
    reutiliza la clase `Fold` para que el resto del pipeline (GARCH,
    datasets) funcione sin cambios."""
    splitter = WalkForwardSplitter()
    full_sessions = nyse_sessions(
        splitter.sample_start.strftime("%Y-%m-%d"), splitter.holdout_end.strftime("%Y-%m-%d")
    )
    train_start = splitter.folds[0].train_start
    train_end = splitter.folds[-1].val_end  # 2021-12-31: todo el desarrollo walk-forward combinado

    train_end_pos = full_sessions.get_loc(train_end)
    embargo_slice = full_sessions[train_end_pos + 1 : train_end_pos + 1 + embargo_sessions]
    embargo_start, embargo_end = embargo_slice[0], embargo_slice[-1]

    val_start_pos = train_end_pos + 1 + embargo_sessions
    val_start = full_sessions[val_start_pos]
    val_end = full_sessions[full_sessions <= splitter.holdout_end][-1]

    return Fold(
        name="holdout_final",
        train_start=train_start,
        train_end=train_end,
        embargo_start=embargo_start,
        embargo_end=embargo_end,
        val_start=val_start,
        val_end=val_end,
    )


def extend_quant_features_to_holdout(ticker: str, fold: Fold, garch_suffix: str = "") -> pd.DataFrame:
    """Añade a las variables GARCH de 2015-2021 el tramo 2022-2023,
    calculado con el GARCH ajustado sobre todo el desarrollo y aplicado con
    los parámetros fijos. Con `garch_suffix="gjr"` usa GJR-GARCH."""
    base = build_quant_feature_table(ticker, garch_suffix=garch_suffix)  # 2015-2021, ya calculado y guardado en disco

    prices = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet")
    o = 1 if garch_suffix == "gjr" else 0
    holdout_segment = _garch_segment(
        prices,
        fold,
        window_start=fold.train_start,
        window_end=fold.val_end,
        output_start=fold.val_start,
        output_end=fold.val_end,
        o=o,
    )
    extra = holdout_segment[["date", "cond_vol", "std_resid"]].merge(
        prices[["date", "log_return"]], on="date", how="left"
    )[["date", "log_return", "cond_vol", "std_resid"]]

    combined = pd.concat([base, extra], ignore_index=True).sort_values("date").reset_index(drop=True)
    assert combined["date"].is_unique, "fechas duplicadas entre desarrollo y holdout: revisar límites"
    return combined


def main(
    tickers: tuple[str, ...] = ("SPY",),
    horizon: int = 1,
    hp_path: Path | None = None,
    lookback: int = L,
    garch_suffix: str = "",
    use_embedding_layernorm: bool = False,
    variants: tuple[str, ...] = VARIANTS,
    text_proj_dim: int | None = None,
    dev_thresholds: dict[tuple[str, str], float] | None = None,
    save_checkpoints: bool = False,
) -> pd.DataFrame:
    """Entrena las variantes indicadas sobre todo el desarrollo (2015-2021)
    y las evalúa una vez sobre el holdout (2022-2023).

    Parámetros principales:
    - `variants`: variantes que se evalúan (por defecto, las cinco); deben
      tener hiperparámetros en `hp_path`.
    - `hp_path`: fichero de hiperparámetros; por defecto, el del propio
      horizonte.
    - `garch_suffix`, `use_embedding_layernorm` y `text_proj_dim`:
      configuraciones alternativas del modelo, desactivadas por defecto.
    - `dev_thresholds`: umbrales calibrados en desarrollo, {(variante,
      activo): (tasa, umbral)} (ver compute_dev_thresholds.py). Se hereda la
      tasa de clasificación positiva, no el umbral: el corte se recalcula en
      cada sesión con las probabilidades observadas hasta ese día, porque el
      modelo final puede dar probabilidades en otra escala que los modelos
      de fold. Añade columnas con el sufijo `_calibrado` sin modificar las
      originales.
    - `save_checkpoints`: guarda cada modelo en
      models/{variante}_{activo}_holdout_h{H}_L{L}.pt, para poder recalcular
      métricas sin reentrenar.

    El resultado se guarda en
    results/holdout_final_evaluation_h{H}_{activos}[...].csv. El nombre
    lleva un sufijo por cada opción distinta de la configuración por
    defecto, de modo que ninguna ejecución sobrescribe otra."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embargo = max(DEFAULT_EMBARGO_SESSIONS, horizon)
    fold = build_holdout_fold(embargo)
    print(
        f"Holdout final (h={horizon}, L={lookback}): train {fold.train_start.date()} -> {fold.train_end.date()}, "
        f"embargo {fold.embargo_start.date()} -> {fold.embargo_end.date()} ({embargo} sesiones), "
        f"holdout (val) {fold.val_start.date()} -> {fold.val_end.date()}"
    )

    # sin ruta explícita se usan los hiperparámetros del propio horizonte,
    # nunca los de otro
    if hp_path is None:
        hp_path, best_hp = require_hyperparameters_for(horizon, RESULTS_DIR, tickers=tickers)
    else:
        best_hp = load_best_hyperparameters(hp_path)
    print(f"Hiperparámetros: {hp_path.name}")
    warn_if_provenance_mismatch(
        best_hp,
        hp_path,
        horizon=horizon,
        tickers=tuple(tickers),
        lookback=lookback,
        garch="gjr" if garch_suffix else "symmetric",
    )

    all_rows = []
    for ticker in tickers:
        quant_df = extend_quant_features_to_holdout(ticker, fold, garch_suffix=garch_suffix)
        daily_agg = load_daily_agg(ticker)
        variable_set = load_variable_set(ticker)
        targets = targets_with_horizon(ticker, horizon)
        prices = pd.read_parquet(DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet")

        train_dates = prices.loc[(prices["date"] >= fold.train_start) & (prices["date"] <= fold.train_end), "date"]
        val_dates = prices.loc[(prices["date"] >= fold.val_start) & (prices["date"] <= fold.val_end), "date"]

        train_ds = build_dataset(train_dates, quant_df, daily_agg, targets, lookback, horizon, variable_set=variable_set)
        val_ds = build_dataset(val_dates, quant_df, daily_agg, targets, lookback, horizon, variable_set=variable_set)

        train_q, val_q = scale_quant_seq(train_ds["quant_seq"], val_ds["quant_seq"])
        train_ds = dict(train_ds, quant_seq=train_q)
        val_ds = dict(val_ds, quant_seq=val_q)
        train_t = to_tensors(train_ds, device)
        val_t = to_tensors(val_ds, device)

        n_news_holdout = int(val_t["text_today_has_news"].sum().item())
        print(
            f"\n{ticker}: train={len(train_ds['date'])} muestras (2015-2021), "
            f"holdout={len(val_ds['date'])} muestras (2022-2023), "
            f"sesiones con noticia real en holdout={n_news_holdout}/{len(val_ds['date'])}"
        )

        # Retorno logarítmico real a `horizon` sesiones en las mismas fechas
        # que quedaron en val_ds. Convierte la señal de dirección en el retorno
        # de una estrategia direccional simétrica (Sharpe y Sortino, ver
        # financial_metrics.py). Como se toma en cada sesión, con h > 1 las
        # ventanas se solapan y hay menos observaciones independientes que
        # filas: el Sharpe y el Sortino a h > 1 son una aproximación, no una
        # estrategia con rebalanceo cada h sesiones.
        price_series = prices.set_index("date")["adj_close"]
        fwd_log_return = np.log(price_series.shift(-horizon) / price_series)
        actual_fwd_return = fwd_log_return.reindex(val_ds["date"]).to_numpy()

        for variant in variants:
            params = best_hp[variant]["best_params"]
            model, metrics = train_and_eval_final(
                variant, params, train_t, val_t, device, use_embedding_layernorm, text_proj_dim=text_proj_dim
            )

            model.eval()
            with torch.no_grad():
                val_batch = {k: val_t[k] for k in TENSOR_KEYS_ALL}
                dir_pred_proba = torch.sigmoid(model(val_batch)["direction_logit"]).cpu().numpy()
            y_dir_true = val_t["y_direction"].cpu().numpy()

            fin_metrics = compute_all_financial_metrics(
                dir_pred_proba,
                y_dir_true,
                actual_fwd_return,
                periods_per_year=TRADING_DAYS_PER_YEAR / horizon,
            )

            row = {"ticker": ticker, "horizon": horizon, "variant": variant, **metrics, **fin_metrics}

            if dev_thresholds is not None and (variant, ticker) in dev_thresholds:
                # Se hereda la tasa de clasificación positiva calibrada en
                # desarrollo (qué fracción de sesiones se predicen al alza), no
                # el umbral: el modelo final, entrenado con todo 2015-2021,
                # puede dar probabilidades en otra escala que los modelos de
                # fold, y el umbral tal cual podría quedar fuera de su rango.
                tasa_positiva, umbral_dev = dev_thresholds[(variant, ticker)]
                # Umbral causal: en cada sesión, el corte que reproduce la tasa
                # heredada usando solo las probabilidades observadas hasta esa
                # sesión. Calcular el cuantil con todo el holdout haría que el
                # umbral del primer día dependiera de sesiones de hasta dos
                # años después (ver expanding_calibrated_threshold).
                thr_series = expanding_calibrated_threshold(dir_pred_proba, tasa_positiva, umbral_dev)
                thr = float(thr_series[-1])  # el vigente al cierre del holdout
                pred_cal = (dir_pred_proba > thr_series).astype(float)
                fin_metrics_cal = compute_all_financial_metrics(
                    dir_pred_proba,
                    y_dir_true,
                    actual_fwd_return,
                    periods_per_year=TRADING_DAYS_PER_YEAR / horizon,
                    threshold=thr_series,
                )
                row.update(
                    {
                        "umbral_calibrado": thr,
                        "accuracy_calibrado": accuracy_score(y_dir_true, pred_cal),
                        "f1_calibrado": f1_score(y_dir_true, pred_cal, zero_division=0),
                        "sharpe_ratio_calibrado": fin_metrics_cal["sharpe_ratio"],
                        "sortino_ratio_calibrado": fin_metrics_cal["sortino_ratio"],
                        "pt_statistic_calibrado": fin_metrics_cal["pt_statistic"],
                        "pt_p_value_calibrado": fin_metrics_cal["pt_p_value"],
                        "cohen_kappa_calibrado": fin_metrics_cal["cohen_kappa"],
                    }
                )

            if save_checkpoints:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                ckpt_path = MODELS_DIR / f"{variant}_{ticker}_holdout_h{horizon}_L{lookback}.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "variant": variant,
                        "params": params,
                        "ticker": ticker,
                        "horizon": horizon,
                        "lookback": lookback,
                    },
                    ckpt_path,
                )

            all_rows.append(row)
            print(
                f"  {variant}: acc={metrics['accuracy']:.3f} f1={metrics['f1']:.3f} "
                f"rmse_rv={metrics['rmse_rv']:.4f} mae_rv={metrics['mae_rv']:.4f} "
                f"sharpe={fin_metrics['sharpe_ratio']:.3f} sortino={fin_metrics['sortino_ratio']:.3f} "
                f"pt_p={fin_metrics['pt_p_value']:.3f} auc={fin_metrics['roc_auc']:.3f} "
                f"kappa={fin_metrics['cohen_kappa']:.3f}"
            )
            # también se muestran por consola las métricas calibradas: si no,
            # solo se vería el Kappa con umbral 0,5, que en las combinaciones
            # colapsadas es nulo
            if "cohen_kappa_calibrado" in row:
                colapso = "  <- COLAPSA" if abs(row["cohen_kappa_calibrado"]) < 1e-12 else ""
                print(
                    f"      calibrado (umbral {row['umbral_calibrado']:.3f}): "
                    f"acc={row['accuracy_calibrado']:.3f} f1={row['f1_calibrado']:.3f} "
                    f"pt_p={row['pt_p_value_calibrado']:.3f} "
                    f"kappa={row['cohen_kappa_calibrado']:.3f}{colapso}"
                )

    df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if horizon == 1 else f"_h{horizon}"
    # si los hiperparámetros son los del fichero de referencia de este
    # horizonte, no se añade nada al nombre; si son otros, se añade el nombre
    # de su fichero
    canonico = hyperparameters_path_for(horizon, RESULTS_DIR)
    hp_tag = "" if hp_path.name == canonico.name else f"_{hp_path.stem}"
    # los activos van siempre en el nombre, para saber de qué activo es cada
    # fichero y que dos ejecuciones distintas no se sobrescriban
    tickers_tag = f"_{'_'.join(t.lower() for t in tickers)}"
    lookback_tag = "" if lookback == L else f"_L{lookback}"
    # si se evalúa solo una parte de las variantes (por ejemplo, para
    # paralelizar) o se calibra el umbral, el nombre lo indica, para no
    # sobrescribir el resultado completo con uno parcial
    variants_tag = "" if set(variants) == set(VARIANTS) else f"_{'_'.join(v for v in variants)}"
    calibrado_tag = "_calibrado" if dev_thresholds is not None else ""
    improved_tag = (
        ("_gjr" if garch_suffix else "")
        + ("_layernorm" if use_embedding_layernorm else "")
        + (f"_textproj{text_proj_dim}" if text_proj_dim is not None else "")
    )
    out_path = RESULTS_DIR / f"holdout_final_evaluation{suffix}{tickers_tag}{lookback_tag}{improved_tag}{variants_tag}{calibrado_tag}{hp_tag}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")
    return df


if __name__ == "__main__":
    import sys

    # se pueden indicar varios horizontes por línea de comandos, por ejemplo
    # «python -m src.training.holdout_evaluation 5 20»; sin argumentos se
    # evalúa h=1
    horizons = [int(h) for h in sys.argv[1:]] or [1]
    for h in horizons:
        main(horizon=h)
        print()
