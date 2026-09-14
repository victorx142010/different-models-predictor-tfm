"""Recalcula todas las métricas, contrastes y tablas de resultados del holdout
a partir de las predicciones de consolidate_holdout.py, que debe ejecutarse
antes.

Calcula las métricas de cada combinación, con el umbral por defecto y con el
calibrado, el test de Diebold-Mariano interno (variantes con texto frente a
price_only) y el externo (variantes frente a líneas base), y escribe las tablas
en Markdown, listas para copiar sin transcribir número a número.

Tras calcular las métricas, las compara con la ejecución de referencia
(holdout_recalibrado_todas_variantes.csv) e imprime la mayor diferencia. Si la
inferencia desde checkpoint es fiel, la diferencia es del orden del error de
redondeo de la máquina; una mayor indica que la reproducción no es exacta.

Salida:
    results/consolidado_metricas_variantes.csv
        60 filas, una por activo × horizonte × variante
    results/consolidado_metricas_baselines.csv
        36 filas, una por activo × horizonte × línea base
    results/consolidado_dm_interno.csv
        36 contrastes de las variantes con texto frente a price_only
    results/consolidado_dm_baselines.csv
        180 contrastes de las variantes frente a las líneas base
    results/tablas_capitulo5.md
        todas las tablas en Markdown
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error

from src.evaluation.financial_metrics import (
    expanding_calibrated_threshold,
    TRADING_DAYS_PER_YEAR,
    compute_all_financial_metrics,
    diebold_mariano_test,
    pesaran_timmermann_test,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

VARIANTS = ("price_only", "text_only", "early_fusion", "late_fusion", "cross_attention")
TEXT_VARIANTS = ("early_fusion", "cross_attention", "text_only")
BASELINES = ("random_walk", "naive_persistente", "arima_garch")
BASELINE_LABEL = {
    "random_walk": "Camino aleatorio",
    "naive_persistente": "Persistente (naïve)",
    "arima_garch": "ARIMA-GARCH",
}
TICKERS = ("SPY", "QQQ", "KO", "GS")
HORIZONS = (5, 20, 60)
N_COMBOS = 60


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _f(x: float, d: int = 4) -> str:
    """Devuelve el número como texto, con coma decimal."""
    if pd.isna(x):
        return "n/a"
    return f"{x:.{d}f}".replace(".", ",")


# --------------------------------------------------------------------------
# 1. Una fila por combinación activo × horizonte × variante, con y sin
#    calibrar el umbral
# --------------------------------------------------------------------------
def variant_metrics(var_df: pd.DataFrame, thr_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada combinación activo × horizonte × variante a partir
    de sus predicciones, con el umbral de 0,5 y con el umbral causal
    calibrado (tasa heredada de desarrollo)."""
    rates = {
        (r["variant"], r["ticker"], r["horizon"]): (r["tasa_positiva_calibrada"], r["umbral_calibrado"])
        for _, r in thr_df.iterrows()
    }
    rows = []
    for (ticker, horizon, variant), g in var_df.groupby(["ticker", "horizon", "variant"]):
        # el umbral expansivo depende del orden temporal: se ordena
        # explícitamente en lugar de confiar en el orden del CSV
        g = g.sort_values("date")
        # float32 a propósito: es el tipo con el que el modelo produce las
        # probabilidades y con el que holdout_evaluation.main calcula el
        # percentil del umbral. Con float64 el percentil cambia lo justo para
        # que unas pocas sesiones en la frontera cambien de clase, y las cifras
        # no reproducirían la ejecución original.
        proba = g["proba_dir"].to_numpy(dtype=np.float32)
        y_dir = g["y_dir"].to_numpy()
        vol_pred = g["vol_pred"].to_numpy(dtype=np.float32)
        y_vol = g["y_vol"].to_numpy(dtype=np.float32)
        fwd = g["fwd_return"].to_numpy()
        ppy = TRADING_DAYS_PER_YEAR / horizon

        base = compute_all_financial_metrics(proba, y_dir, fwd, periods_per_year=ppy)
        pred05 = (proba > 0.5).astype(float)

        tasa, umbral_dev = rates[(variant, ticker, horizon)]
        # umbral causal: cada sesión usa solo las probabilidades observadas
        # hasta ella
        thr_series = expanding_calibrated_threshold(proba, tasa, umbral_dev)
        thr = float(thr_series[-1])  # el vigente al cierre, para la tabla
        cal = compute_all_financial_metrics(proba, y_dir, fwd, periods_per_year=ppy, threshold=thr_series)
        predc = (proba > thr_series).astype(float)

        rows.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "variant": variant,
                "n": len(g),
                "accuracy": accuracy_score(y_dir, pred05),
                "f1": f1_score(y_dir, pred05, zero_division=0),
                "cohen_kappa": cohen_kappa_score(y_dir, pred05),
                "rmse_rv": _rmse(y_vol, vol_pred),
                "mae_rv": mean_absolute_error(y_vol, vol_pred),
                "roc_auc": base["roc_auc"],
                "sharpe_ratio": base["sharpe_ratio"],
                "sortino_ratio": base["sortino_ratio"],
                "pt_p_value": base["pt_p_value"],
                "umbral_calibrado": thr,
                "accuracy_calibrado": accuracy_score(y_dir, predc),
                "f1_calibrado": f1_score(y_dir, predc, zero_division=0),
                "cohen_kappa_calibrado": cohen_kappa_score(y_dir, predc),
                "sharpe_ratio_calibrado": cal["sharpe_ratio"],
                "sortino_ratio_calibrado": cal["sortino_ratio"],
                "pt_p_value_calibrado": cal["pt_p_value"],
                "colapsado_05": len(np.unique(pred05)) == 1,
                "colapsado_cal": len(np.unique(predc)) == 1,
            }
        )
    return pd.DataFrame(rows)


def baseline_metrics(base_df: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada combinación activo × horizonte × línea base."""
    rows = []
    for (ticker, horizon, model), g in base_df.groupby(["ticker", "horizon", "model"]):
        d = g["dir_pred"].to_numpy()
        y_dir = g["y_dir"].to_numpy()
        v = g["vol_pred"].to_numpy()
        y_vol = g["y_vol"].to_numpy()
        m = ~np.isnan(d) & ~np.isnan(y_dir)
        mv = ~np.isnan(v) & ~np.isnan(y_vol)
        pt = pesaran_timmermann_test(d[m], y_dir[m])
        rows.append(
            {
                "ticker": ticker,
                "horizon": horizon,
                "model": model,
                "accuracy": accuracy_score(y_dir[m], d[m]),
                "f1": f1_score(y_dir[m], d[m], zero_division=0),
                "cohen_kappa": cohen_kappa_score(y_dir[m], d[m]),
                "pt_p_value": pt["p_value"],
                "rmse_rv": _rmse(y_vol[mv], v[mv]),
                "mae_rv": mean_absolute_error(y_vol[mv], v[mv]),
                "colapsado": len(np.unique(d[m])) == 1,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Los dos contrastes de Diebold-Mariano sobre el error de volatilidad
# --------------------------------------------------------------------------
def dm_internal(var_df: pd.DataFrame) -> pd.DataFrame:
    """Test de Diebold-Mariano de cada variante con texto frente a
    price_only sobre el error de volatilidad: indica si el texto aporta algo
    dentro de la propia familia de modelos."""
    rows = []
    for (ticker, horizon), g in var_df.groupby(["ticker", "horizon"]):
        piv = g.pivot(index="date", columns="variant", values="vol_pred")
        y_vol = g[g["variant"] == "price_only"].set_index("date")["y_vol"].reindex(piv.index).to_numpy()
        e_price = piv["price_only"].to_numpy() - y_vol
        for variant in TEXT_VARIANTS:
            e_var = piv[variant].to_numpy() - y_vol
            dm = diebold_mariano_test(e_var, e_price, h=horizon)
            rows.append(
                {
                    "ticker": ticker,
                    "horizon": horizon,
                    "variant": variant,
                    "dm_statistic": dm["statistic"],
                    "dm_p_value": dm["p_value"],
                    "mean_loss_diff": dm["mean_loss_diff"],
                }
            )
    return pd.DataFrame(rows)


def dm_vs_baselines(var_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """Test de Diebold-Mariano de cada variante frente a cada línea base
    sobre el error de volatilidad."""
    rows = []
    for (ticker, horizon), g in var_df.groupby(["ticker", "horizon"]):
        piv = g.pivot(index="date", columns="variant", values="vol_pred")
        y_vol = g[g["variant"] == "price_only"].set_index("date")["y_vol"].reindex(piv.index).to_numpy()
        bsub = base_df[(base_df["ticker"] == ticker) & (base_df["horizon"] == horizon)]
        bpiv = bsub.pivot(index="date", columns="model", values="vol_pred").reindex(piv.index)
        for variant in VARIANTS:
            e_var = piv[variant].to_numpy() - y_vol
            for model in BASELINES:
                e_base = bpiv[model].to_numpy() - y_vol
                valid = ~np.isnan(e_var) & ~np.isnan(e_base)
                dm = diebold_mariano_test(e_var[valid], e_base[valid], h=horizon)
                rows.append(
                    {
                        "ticker": ticker,
                        "horizon": horizon,
                        "variant": variant,
                        "baseline": model,
                        "dm_statistic": dm["statistic"],
                        "dm_p_value": dm["p_value"],
                        "mean_loss_diff": dm["mean_loss_diff"],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. La inferencia desde checkpoint tiene que reproducir la ejecución de
#    referencia, o las tablas no valen
# --------------------------------------------------------------------------
def validate(met: pd.DataFrame) -> None:
    """Compara las métricas recalculadas con las de la ejecución de
    referencia e imprime la mayor diferencia de cada métrica."""
    ref_path = RESULTS_DIR / "holdout_recalibrado_todas_variantes.csv"
    if not ref_path.exists():
        print("[validación] no se encuentra el fichero de referencia, se omite")
        return
    ref = pd.read_csv(ref_path)
    k = ["ticker", "horizon", "variant"]
    cols = [
        "accuracy", "f1", "rmse_rv", "mae_rv", "roc_auc",
        "accuracy_calibrado", "f1_calibrado", "cohen_kappa_calibrado",
        "sharpe_ratio_calibrado", "pt_p_value_calibrado",
    ]
    m = met[k + cols].merge(ref[k + cols], on=k, suffixes=("_new", "_ref"))
    print(f"\n[validación] {len(m)} combinaciones cotejadas contra la ejecución del 2026-09-05")
    worst = 0.0
    for c in cols:
        d = (m[f"{c}_new"] - m[f"{c}_ref"]).abs()
        d = d[~d.isna()]
        worst = max(worst, float(d.max()) if len(d) else 0.0)
        print(f"    {c:26s} max abs dif = {d.max():.2e}" if len(d) else f"    {c:26s} (sin datos)")
    verdict = "IDÉNTICA" if worst < 1e-9 else ("compatible" if worst < 1e-4 else "DISCREPA")
    print(f"  -> desajuste maximo global: {worst:.2e}  ({verdict})")


# --------------------------------------------------------------------------
# 4. Las tablas en Markdown, ya formateadas para copiarlas
# --------------------------------------------------------------------------
def _md(headers: list[str], rows: list[list[str]]) -> str:
    """Tabla en formato Markdown a partir de cabeceras y filas."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_markdown(met: pd.DataFrame, bmet: pd.DataFrame, dmi: pd.DataFrame, dmb: pd.DataFrame) -> str:
    """Construye en Markdown todas las tablas de resultados."""
    P = []
    A = "\n\n"

    # medianas por variante
    g = met.groupby("variant")[
        ["accuracy_calibrado", "f1_calibrado", "rmse_rv", "roc_auc",
         "sharpe_ratio_calibrado", "sortino_ratio_calibrado"]
    ].median().reindex(VARIANTS)
    P.append("## Tabla 5.1 — Resultados descriptivos por variante (mediana sobre 12 combinaciones activo × horizonte)" + A +
             _md(["Variante", "Accuracy", "F1-Score", "RMSE (RV)", "AUC-ROC", "Sharpe", "Sortino"],
                 [[f"*{v}*", _f(g.loc[v, "accuracy_calibrado"]), _f(g.loc[v, "f1_calibrado"]),
                   _f(g.loc[v, "rmse_rv"]), _f(g.loc[v, "roc_auc"]),
                   _f(g.loc[v, "sharpe_ratio_calibrado"], 3), _f(g.loc[v, "sortino_ratio_calibrado"], 3)]
                  for v in VARIANTS]))

    # medianas por horizonte
    gh = met.groupby("horizon")[
        ["accuracy_calibrado", "f1_calibrado", "rmse_rv", "roc_auc", "accuracy"]
    ].median()
    col05 = met.groupby("horizon")["colapsado_05"].sum()
    P.append("## Tabla 5.2 — Resultados descriptivos por horizonte (mediana sobre 20 combinaciones activo × variante)" + A +
             _md(["Horizonte", "Accuracy", "F1-Score", "RMSE (RV)", "AUC",
                  "Accuracy sin calibrar", "Colapsadas con umbral por defecto"],
                 [[f"h = {h}", _f(gh.loc[h, "accuracy_calibrado"]), _f(gh.loc[h, "f1_calibrado"]),
                   _f(gh.loc[h, "rmse_rv"]), _f(gh.loc[h, "roc_auc"]), _f(gh.loc[h, "accuracy"]),
                   f"{int(col05.loc[h])}/20"] for h in HORIZONS]))

    # colapso a clase constante por variante
    c05 = met.groupby("variant")["colapsado_05"].sum()
    ccal = met.groupby("variant")["colapsado_cal"].sum()
    order = c05.sort_values(ascending=False).index.tolist()
    P.append(f"## Tabla 5.3 — Recuento de colapso a clase constante por variante "
             f"(total {int(c05.sum())}/60 con umbral 0,5)" + A +
             _md(["Variante", "Comb. colapsadas, umbral 0,5", "Comb. colapsadas, umbral calibrado"],
                 [[f"*{v}*", str(int(c05[v])), str(int(ccal[v]))] for v in order]))

    # Pesaran-Timmermann de las variantes, con el umbral calibrado
    alpha_b = 0.05 / N_COMBOS
    sig = met[met["pt_p_value_calibrado"] < 0.05].sort_values("pt_p_value_calibrado")
    P.append(f"## Tabla 5.4 — Test de Pesaran-Timmermann tras calibrar el umbral "
             f"({len(sig)} significativas al nivel nominal; Bonferroni alfa = 0,05/60 = {_f(alpha_b, 5)})" + A +
             _md(["Activo", "Horizonte", "Variante", "Accuracy", "Kappa (κ)", "AUC", "p-valor", "Bonferroni"],
                 [[r.ticker, str(r.horizon), f"*{r.variant}*", _f(r.accuracy_calibrado),
                   _f(r.cohen_kappa_calibrado), _f(r.roc_auc), _f(r.pt_p_value_calibrado, 6),
                   "✓" if r.pt_p_value_calibrado < alpha_b else "✗"] for r in sig.itertuples()]))

    # Pesaran-Timmermann de las líneas base
    n_ok = int((~bmet["colapsado"]).sum())
    alpha_bl = 0.05 / n_ok
    rows55 = []
    for m_ in BASELINES:
        s = bmet[bmet["model"] == m_]
        ncol = int(s["colapsado"].sum())
        live = s[~s["colapsado"]]
        nsig = int((live["pt_p_value"] < 0.05).sum())
        nbon = int((live["pt_p_value"] < alpha_bl).sum())
        rows55.append([BASELINE_LABEL[m_], f"{ncol}/12",
                       "n/a" if len(live) == 0 else f"{nsig}/{len(live)}",
                       "n/a" if len(live) == 0 else str(nbon)])
    P.append(f"## Tabla 5.5 — Test de Pesaran-Timmermann sobre las líneas base "
             f"(36 combinaciones; Bonferroni sobre las {n_ok} no colapsadas, alfa = {_f(alpha_bl, 5)})" + A +
             _md(["Línea base", "Colapsadas (Kappa = 0)", "Significativas", "Sobreviven Bonferroni"], rows55))

    # Diebold-Mariano de las variantes con texto frente a price_only
    alpha_i = 0.05 / len(dmi)
    rows56 = []
    for v in TEXT_VARIANTS:
        s = dmi[dmi["variant"] == v]
        peor_n = int(((s.dm_p_value < 0.05) & (s.mean_loss_diff > 0)).sum())
        mejor_n = int(((s.dm_p_value < 0.05) & (s.mean_loss_diff < 0)).sum())
        peor_b = int(((s.dm_p_value < alpha_i) & (s.mean_loss_diff > 0)).sum())
        mejor_b = int(((s.dm_p_value < alpha_i) & (s.mean_loss_diff < 0)).sum())
        rows56.append([f"*{v}*", str(peor_n), str(mejor_n), str(peor_b), str(mejor_b)])
    P.append(f"## Tabla 5.6 — Diebold-Mariano frente a `price_only` sobre volatilidad "
             f"(12 combinaciones x 3 variantes; Bonferroni sobre {len(dmi)} tests, alfa = {_f(alpha_i, 5)})" + A +
             _md(["Variante", "Peor (nominal)", "Mejor (nominal)", "Peor (Bonferroni)", "Mejor (Bonferroni)"], rows56))

    # ranking de todos los modelos por RMSE de volatilidad
    rank = pd.concat([
        met.groupby("variant")["rmse_rv"].median().rename("rmse").reset_index().rename(columns={"variant": "modelo"}),
        bmet.groupby("model")["rmse_rv"].median().rename("rmse").reset_index().rename(columns={"model": "modelo"}),
    ]).sort_values("rmse").reset_index(drop=True)
    P.append("## Tabla 5.7 — Ranking de modelos por RMSE de volatilidad (mediana, 12 combinaciones)" + A +
             _md(["Puesto", "Modelo", "RMSE (RV)"],
                 [[str(i + 1), BASELINE_LABEL.get(r.modelo, f"*{r.modelo}*"), _f(r.rmse)]
                  for i, r in enumerate(rank.itertuples())]))

    # Diebold-Mariano de las variantes frente a las líneas base
    alpha_e = 0.05 / len(dmb)
    hdr = ["Variante"]
    for m_ in BASELINES:
        hdr += [f"{BASELINE_LABEL[m_]} — peor", f"{BASELINE_LABEL[m_]} — mejor"]
    hdr += ["Total peor", "Total mejor"]
    rows58, tp, tm = [], [0] * 3, [0] * 3
    order58 = met.groupby("variant")["rmse_rv"].median().sort_values(ascending=False).index.tolist()
    for v in order58:
        row = [f"*{v}*"]
        sp = sm = 0
        for j, m_ in enumerate(BASELINES):
            s = dmb[(dmb["variant"] == v) & (dmb["baseline"] == m_)]
            p = int(((s.dm_p_value < alpha_e) & (s.mean_loss_diff > 0)).sum())
            q = int(((s.dm_p_value < alpha_e) & (s.mean_loss_diff < 0)).sum())
            row += [str(p), str(q)]
            sp += p
            sm += q
            tp[j] += p
            tm[j] += q
        row += [str(sp), str(sm)]
        rows58.append(row)
    total_row = ["**Total**"]
    for j in range(3):
        total_row += [f"**{tp[j]}**", f"**{tm[j]}**"]
    total_row += [f"**{sum(tp)}**", f"**{sum(tm)}**"]
    rows58.append(total_row)
    P.append(f"## Tabla 5.8 — Diebold-Mariano, las cinco variantes frente a las tres líneas base "
             f"({len(dmb)} contrastes, bajo Bonferroni alfa = {_f(alpha_e, 6)})" + A +
             _md(hdr, rows58))

    # desglose por activo de las medianas por variante, una tabla por horizonte
    for i, h in enumerate(HORIZONS, start=1):
        s = met[met["horizon"] == h]
        rows = []
        for v in VARIANTS:
            for tk in sorted(TICKERS):
                r = s[(s["variant"] == v) & (s["ticker"] == tk)].iloc[0]
                rows.append([tk, v, _f(r.accuracy_calibrado), _f(r.f1_calibrado), _f(r.rmse_rv),
                             _f(r.roc_auc), _f(r.sharpe_ratio_calibrado, 3), _f(r.sortino_ratio_calibrado, 3)])
        P.append(f"## Tabla 8.{i} — Resultados descriptivos por variante y ticker, horizonte h={h}" + A +
                 _md(["Ticker", "Variante", "Accuracy", "F1", "RMSE (RV)", "AUC", "Sharpe", "Sortino"], rows))

    # RMSE de las líneas base por activo, una tabla por horizonte
    for i, h in enumerate(HORIZONS, start=4):
        rows = []
        for m_ in BASELINES:
            for tk in sorted(TICKERS):
                r = bmet[(bmet.model == m_) & (bmet.ticker == tk) & (bmet.horizon == h)].iloc[0]
                rows.append([tk, BASELINE_LABEL[m_], _f(r.rmse_rv)])
        for v in VARIANTS:
            for tk in sorted(TICKERS):
                r = met[(met.variant == v) & (met.ticker == tk) & (met.horizon == h)].iloc[0]
                rows.append([tk, v, _f(r.rmse_rv)])
        P.append(f"## Tabla 8.{i} — RMSE de volatilidad por modelo y ticker, horizonte h={h}" + A +
                 _md(["Ticker", "Modelo", "RMSE (RV)"], rows))

    header = (
        "# Tablas del capítulo 5 y del Anexo B — ejecución consolidada\n\n"
        "Todas las cifras de este documento proceden de una única fuente: las predicciones\n"
        "sesión a sesión de `results/consolidado_predicciones_variantes.csv` y\n"
        "`results/consolidado_predicciones_baselines.csv`, generadas por\n"
        "`src/evaluation/consolidate_holdout.py` a partir de los 60 checkpoints del holdout\n"
        "(`models/{variant}_{ticker}_holdout_h{horizon}_L20.pt`). No hay mezcla de ejecuciones.\n\n"
        "*Fuente de todas las tablas: elaboración propia.*\n"
    )
    return header + A + (A + "\n").join(P) + "\n"


def main() -> None:
    """Lee las predicciones consolidadas y los umbrales, calcula métricas y
    contrastes, los valida y guarda los ficheros de resultados."""
    var_df = pd.read_csv(RESULTS_DIR / "consolidado_predicciones_variantes.csv", parse_dates=["date"])
    base_df = pd.read_csv(RESULTS_DIR / "consolidado_predicciones_baselines.csv", parse_dates=["date"])
    thr_df = pd.read_csv(RESULTS_DIR / "dev_calibrated_thresholds.csv")

    met = variant_metrics(var_df, thr_df)
    bmet = baseline_metrics(base_df)
    dmi = dm_internal(var_df)
    dmb = dm_vs_baselines(var_df, base_df)

    met.to_csv(RESULTS_DIR / "consolidado_metricas_variantes.csv", index=False)
    bmet.to_csv(RESULTS_DIR / "consolidado_metricas_baselines.csv", index=False)
    dmi.to_csv(RESULTS_DIR / "consolidado_dm_interno.csv", index=False)
    dmb.to_csv(RESULTS_DIR / "consolidado_dm_baselines.csv", index=False)

    validate(met)

    md_path = RESULTS_DIR / "tablas_capitulo5.md"
    md_path.write_text(build_markdown(met, bmet, dmi, dmb), encoding="utf-8")
    print(f"\n-> {md_path}")
    print(f"-> consolidado_metricas_variantes.csv ({len(met)} filas)")
    print(f"-> consolidado_metricas_baselines.csv ({len(bmet)} filas)")
    print(f"-> consolidado_dm_interno.csv ({len(dmi)} filas)")
    print(f"-> consolidado_dm_baselines.csv ({len(dmb)} filas)")


if __name__ == "__main__":
    main()
