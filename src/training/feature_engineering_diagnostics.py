"""Diagnóstico de las variables de entrada de la rama cuantitativa: comprueba
con datos si añadir otras variables mejora el modelo o si el diseño de tres
variables (`QUANT_INPUT_DIM=3`) se sostiene.

Condiciones que se pueden comparar:
- base: las tres variables del modelo.
- volumen: añade log(1 + volumen).
- tecnicos: añade el RSI-14 y el ATR-14 normalizado (technical_indicators.py);
  ambos: volumen e indicadores.
- garch_fcst: añade el pronóstico GARCH de volatilidad a h sesiones, la misma
  magnitud que predice ARIMA-GARCH, para ver si reduce la desventaja de la LSTM
  frente a esa línea base en volatilidad.
- L10, L20, L40 y L60: cambian la ventana L en lugar de las variables, para
  comprobar la elección de L=20.

Dos comprobaciones:

1. `quant_feature_correlation`: correlaciones entre las variables actuales y
   las candidatas, por activo y en conjunto. No entrena nada y muestra si una
   candidata es redundante, como el ATR con `cond_vol`.
2. `run_ablation`: entrena las variantes indicadas en los folds de desarrollo,
   una vez por condición. Por defecto reutiliza los hiperparámetros guardados;
   con `--hpo` busca hiperparámetros propios para cada condición, una
   comparación más justa porque el óptimo puede depender del número de
   variables.

Nunca usa el holdout: elegir variables con él sería una fuga de información.
Ninguna condición se incorpora al modelo final sin una decisión explícita.

Uso:
    python -m src.training.feature_engineering_diagnostics SPY QQQ KO GS
    python -m src.training.feature_engineering_diagnostics SPY --horizon 20 --conditions volumen tecnicos
    python -m src.training.feature_engineering_diagnostics SPY QQQ KO GS --hpo --n-trials 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.fusion_model.base_model import VARIANTS
from src.ingestion.download_prices import TICKERS
from src.training.hpo import precompute_all_data, run_study
from src.training.hyperparameters_io import hyperparameters_path_for, load_best_hyperparameters
from src.training.sequence_dataset import build_quant_feature_table
from src.training.train_final_models import train_and_eval_final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

QUANT_FEATURE_COLS = ["log_return", "cond_vol", "std_resid", "log_volume", "rsi_14", "atr_14_norm"]
ABLATION_VARIANT = "price_only"
ABLATION_HORIZON = 5
ABLATION_LOOKBACK = 20
ABLATION_N_TRIALS = 20

# etiqueta de condición -> opciones para precompute_all_data y
# build_quant_feature_table. La clave «lookback» no se pasa tal cual:
# run_ablation la usa para cambiar L en esa condición.
CONDITION_KWARGS = {
    "base": {},
    "volumen": {"include_volume": True},
    "tecnicos": {"include_technical": True},
    "ambos": {"include_volume": True, "include_technical": True},
    "garch_fcst": {"include_garch_forecast": True},
    "L10": {"lookback": 10},
    "L20": {"lookback": 20},
    "L40": {"lookback": 40},
    "L60": {"lookback": 60},
}
DEFAULT_CONDITIONS = ("base", "volumen", "tecnicos")


def quant_feature_correlation(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Correlaciones de Pearson entre log_return, cond_vol, std_resid,
    log_volume, rsi_14 y atr_14_norm, sobre la serie completa (calentamiento
    y folds) de cada activo y de todos juntos. Una correlación alta con
    `cond_vol` indica que la variable candidata no aporta información nueva
    frente al GARCH."""
    per_ticker = {}
    pooled_frames = []
    for ticker in tickers:
        quant_df = build_quant_feature_table(ticker, include_volume=True, include_technical=True)
        corr = quant_df[QUANT_FEATURE_COLS].corr()
        per_ticker[ticker] = corr
        pooled_frames.append(quant_df[QUANT_FEATURE_COLS])

    pooled_corr = pd.concat(pooled_frames, ignore_index=True).corr()
    per_ticker["POOLED"] = pooled_corr
    return per_ticker


def run_ablation(
    tickers: tuple[str, ...],
    hp_path: Path,
    conditions: dict[str, dict],
    variants: tuple[str, ...] = (ABLATION_VARIANT,),
    horizon: int = ABLATION_HORIZON,
    lookback: int = ABLATION_LOOKBACK,
    fresh_hpo: bool = False,
    n_trials: int = ABLATION_N_TRIALS,
) -> pd.DataFrame:
    """Entrena cada variante en los folds de desarrollo de cada activo, una
    vez por condición. Cada condición indica qué se añade a `quant_seq` (por
    ejemplo, {"volumen": {"include_volume": True}}), así que, para un mismo
    activo, fold y variante, solo cambian las variables de entrada. Una
    condición puede incluir la clave "lookback" para cambiar la ventana L en
    lugar de las variables.

    Con `fresh_hpo=False` (por defecto) todas las condiciones usan los
    hiperparámetros guardados en `hp_path`: es rápido, pero esos
    hiperparámetros se eligieron para tres variables. Con `fresh_hpo=True`
    se busca con Optuna para cada condición y variante: es mucho más caro,
    pero compara cada especificación con sus propios hiperparámetros
    óptimos."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ablation de features en {device}: variantes={variants}, h={horizon}, L={lookback}, tickers={tickers}")
    print(f"Condiciones: {list(conditions)}, fresh_hpo={fresh_hpo}")

    best_hp = None if fresh_hpo else load_best_hyperparameters(hp_path)

    rows = []
    for label, raw_kwargs in conditions.items():
        extra_kwargs = dict(raw_kwargs)
        cond_lookback = extra_kwargs.pop("lookback", lookback)
        all_data = precompute_all_data(
            device, tickers=tickers, horizon=horizon, lookback=cond_lookback, **extra_kwargs
        )
        for variant in variants:
            if fresh_hpo:
                print(f"  [HPO] {label}/{variant}: {n_trials} trials propios...")
                study = run_study(variant, all_data, device, n_trials)
                params = study.best_params
                print(f"  [HPO] {label}/{variant}: mejor pérdida val={study.best_value:.4f} params={params}")
            else:
                params = best_hp[variant]["best_params"]

            for d in all_data:
                _, metrics = train_and_eval_final(variant, params, d["train"], d["val"], device)
                row = {
                    "ticker": d["ticker"], "fold": d["fold"], "variant": variant,
                    "condicion": label, "lookback": cond_lookback,
                    "hiperparametros": "propios" if fresh_hpo else "reutilizados",
                    **metrics,
                }
                rows.append(row)
                print(
                    f"  [{label}/{variant}] {d['ticker']} {d['fold']} (L={cond_lookback}): acc={metrics['accuracy']:.3f} "
                    f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f} "
                    f"val_loss={metrics['best_val_loss']:.4f}"
                )

    return pd.DataFrame(rows)


def main() -> None:
    """Punto de entrada: calcula las correlaciones (salvo con `--skip-
    correlation`) y ejecuta el diagnóstico con las condiciones indicadas.
    Nunca usa el holdout."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a diagnosticar (por defecto: TICKERS)")
    parser.add_argument(
        "--horizon", type=int, default=ABLATION_HORIZON,
        help=f"Horizonte de predicción en sesiones (por defecto: {ABLATION_HORIZON})",
    )
    parser.add_argument(
        "--lookback", type=int, default=ABLATION_LOOKBACK,
        help=f"Ventana L que ve la LSTM (por defecto: {ABLATION_LOOKBACK}); "
        "las condiciones L10, L20, L40 y L60 la sustituyen, ver --conditions",
    )
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=[ABLATION_VARIANT],
        help=f"Variantes que se entrenan en el diagnóstico (por defecto: {ABLATION_VARIANT}, la más barata)",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=list(CONDITION_KWARGS), default=list(DEFAULT_CONDITIONS),
        help=f"Condiciones a comparar (por defecto: {list(DEFAULT_CONDITIONS)}). "
        "'base': las 3 variables del modelo; 'volumen': +log_volume; 'tecnicos': +rsi_14 y atr_14_norm; "
        "'ambos': las 4 variables extra; 'garch_fcst': +pronóstico GARCH a --horizon sesiones (garch_fcst_vol); "
        "'L10', 'L20', 'L40', 'L60': cambian la ventana en lugar de las variables (no mezclar los dos tipos en la misma ejecución)",
    )
    parser.add_argument(
        "--hpo", action="store_true",
        help="Busca hiperparámetros propios con Optuna para cada condición y variante, en lugar de reutilizar "
        "los de --hyperparams (más caro, pero la comparación es más justa).",
    )
    parser.add_argument("--n-trials", type=int, default=ABLATION_N_TRIALS, help=f"Trials de Optuna si se usa --hpo (por defecto: {ABLATION_N_TRIALS})")
    parser.add_argument(
        "--hyperparams", type=str, default=None,
        help="Ruta al JSON de hiperparámetros que se reutilizan (por defecto: results/best_hyperparameters_h{H}.json, el del horizonte indicado). Se ignora con --hpo.",
    )
    parser.add_argument("--skip-correlation", action="store_true", help="Omite la matriz de correlación y ejecuta solo el diagnóstico")
    args = parser.parse_args()

    tickers = tuple(t.upper() for t in args.tickers)
    tickers_tag = "_".join(t.lower() for t in tickers)
    variants_tag = "_".join(args.variants)
    conditions_tag = "_".join(args.conditions)
    hp_path = Path(args.hyperparams) if args.hyperparams else hyperparameters_path_for(args.horizon, RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.hpo and not hp_path.exists():
        raise FileNotFoundError(
            f"No existe {hp_path}. Pasa --hpo para buscar hiperparámetros propios, "
            f"o --hyperparams <ruta> para apuntar a un JSON ya calculado."
        )

    if not args.skip_correlation:
        print("=== 1. Matriz de correlación entre features cuantitativas (incl. volumen e indicadores técnicos) ===")
        corr_by_ticker = quant_feature_correlation(tickers)
        corr_path = RESULTS_DIR / f"quant_feature_correlation_{tickers_tag}.csv"
        with open(corr_path, "w", encoding="utf-8") as f:
            for name, corr in corr_by_ticker.items():
                print(f"\n-- {name} --")
                print(corr.round(3).to_string())
                f.write(f"# {name}\n")
                corr.round(4).to_csv(f)
                f.write("\n")
        print(f"\n-> {corr_path}")

    print(f"\n=== 2. Ablation: variantes={args.variants}, condiciones={args.conditions}, h={args.horizon}, L={args.lookback} ===")
    conditions = {label: CONDITION_KWARGS[label] for label in args.conditions}
    ablation_df = run_ablation(
        tickers, hp_path, conditions,
        variants=tuple(args.variants), horizon=args.horizon, lookback=args.lookback,
        fresh_hpo=args.hpo, n_trials=args.n_trials,
    )
    hpo_tag = "_hpopropio" if args.hpo else ""
    # si alguna condición cambia L, el nombre del fichero usa «Lvaria» en lugar
    # de un valor único; el CSV guarda igualmente la L real de cada fila
    lookback_tag = "Lvaria" if any("lookback" in kw for kw in conditions.values()) else f"L{args.lookback}"
    ablation_path = RESULTS_DIR / f"feature_ablation_{tickers_tag}_{variants_tag}_{conditions_tag}_h{args.horizon}_{lookback_tag}{hpo_tag}.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"\n{len(ablation_df)} filas -> {ablation_path}")

    summary = ablation_df.groupby(["variant", "condicion"])[["accuracy", "f1", "rmse_rv", "mae_rv", "best_val_loss"]].median().round(4)
    print("\n=== Resumen (mediana across ticker x fold) ===")
    print(summary.to_string())
    summary_path = RESULTS_DIR / f"feature_ablation_summary_{tickers_tag}_{variants_tag}_{conditions_tag}_h{args.horizon}_{lookback_tag}{hpo_tag}.csv"
    summary.to_csv(summary_path)
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
