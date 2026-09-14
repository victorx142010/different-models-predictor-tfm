"""Genera las cuatro figuras de resultados del holdout y guarda aparte los
datos exactos de cada una, para poder contrastarlas con las tablas.

Lee las métricas de rebuild_chapter5_tables.py y las predicciones de
consolidate_holdout.py, así que las figuras salen de la misma ejecución que las
tablas: si se regenera una cosa, hay que regenerar la otra.

Las curvas ROC se calculan con las probabilidades de cada sesión, y el punto
marcado en cada curva es el punto de operación real: la tasa de aciertos y de
falsas alarmas que produjo el umbral calibrado.

Salida (figuras, en la carpeta ../TFM/imagenes/media):
    image8.png   AUC por variante y horizonte, con cada activo
    image9.png   curvas ROC de las combinaciones que superan Bonferroni
    image10.png  Kappa antes y después de calibrar el umbral
    image11.png  mapa de calor del RMSE de los ocho modelos

Salida (datos, en results/figuras):
    datos_figura_5_*.csv            valores exactos de cada figura
    datos_figuras_capitulo5.xlsx    los mismos datos y las tablas de métricas
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

from src.evaluation.financial_metrics import expanding_calibrated_threshold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGDATA_DIR = RESULTS_DIR / "figuras"
IMG_DIR = PROJECT_ROOT.parent / "TFM" / "imagenes" / "media"

VARIANTS = ("price_only", "text_only", "early_fusion", "late_fusion", "cross_attention")
HORIZONS = (5, 20, 60)
TICKERS = ("SPY", "QQQ", "KO", "GS")
BASELINE_LABEL = {
    "random_walk": "Camino aleatorio",
    "naive_persistente": "Persistente (naïve)",
    "arima_garch": "ARIMA-GARCH",
}

# la paleta va escrita a mano y no se deja la de matplotlib por defecto: los
# azules de los horizontes van de oscuro a claro para que h=5, 20 y 60 se lean
# como una progresión aunque se imprima en blanco y negro
C_HOR = {5: "#1b4965", 20: "#5fa8d3", 60: "#bee9e8"}
C_TICKER = {"SPY": "#d1495b", "QQQ": "#edae49", "KO": "#00798c", "GS": "#7d5ba6"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    """Guarda la figura en la carpeta de imágenes y la cierra."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------
# AUC por variante, con cada activo por separado encima de la barra, porque
# la mediana sola esconde la dispersión entre activos
# --------------------------------------------------------------------------
def figura_auc(met: pd.DataFrame) -> pd.DataFrame:
    """AUC mediano por variante y horizonte, con el valor de cada activo.
    Devuelve los datos dibujados."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.26
    x = np.arange(len(VARIANTS))

    for j, h in enumerate(HORIZONS):
        offs = (j - 1) * width
        meds = [met[(met.variant == v) & (met.horizon == h)]["roc_auc"].median() for v in VARIANTS]
        ax.bar(x + offs, meds, width * 0.92, color=C_HOR[h], edgecolor="#1b4965",
               linewidth=0.6, label=f"h = {h}", zorder=2)
        for i, v in enumerate(VARIANTS):
            sub = met[(met.variant == v) & (met.horizon == h)]
            jitter = np.linspace(-width * 0.24, width * 0.24, len(sub))
            for k, (_, r) in enumerate(sub.iterrows()):
                ax.plot(x[i] + offs + jitter[k], r["roc_auc"], "o", ms=4.2,
                        mfc=C_TICKER[r["ticker"]], mec="white", mew=0.6, zorder=4)

    linea_azar = ax.axhline(0.5, color="#444444", ls="--", lw=1.1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in VARIANTS])
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.36, 0.71)
    ax.yaxis.grid(True, ls=":", lw=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    h1, l1 = ax.get_legend_handles_labels()
    h1.append(linea_azar)
    l1.append("azar (AUC = 0,5)")
    leg1 = ax.legend(h1, l1, title="Mediana por horizonte", frameon=False, ncols=4,
                     columnspacing=1.1, loc="lower left", bbox_to_anchor=(0.0, 1.01))
    ax.add_artist(leg1)
    dots = [Line2D([], [], marker="o", ls="", ms=5.5, mfc=C_TICKER[t], mec="white", label=t)
            for t in TICKERS]
    ax.legend(handles=dots, title="Activo (valor individual)", frameon=False, ncols=4,
              columnspacing=0.8, loc="lower right", bbox_to_anchor=(1.0, 1.01))

    _save(fig, "image8.png")
    return met[["ticker", "horizon", "variant", "roc_auc"]].sort_values(["variant", "horizon", "ticker"])


# --------------------------------------------------------------------------
# curvas ROC de las combinaciones que superan Bonferroni, con el punto de
# operación real marcado sobre cada curva
# --------------------------------------------------------------------------
def figura_roc(met: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Curvas ROC de las combinaciones que superan Bonferroni, con su punto
    de operación real. Devuelve los datos dibujados."""
    alpha_b = 0.05 / 60
    dev = pd.read_csv(RESULTS_DIR / "dev_calibrated_thresholds.csv").set_index(
        ["variant", "ticker", "horizon"])
    top = met[met.pt_p_value_calibrado < alpha_b].sort_values("pt_p_value_calibrado")

    # Un panel por cada combinación que supera Bonferroni, en una rejilla de
    # hasta tres columnas. El número de paneles se calcula porque puede cambiar
    # al reejecutar, y en una sola fila seis paneles quedarían demasiado
    # estrechos para leerse impresos.
    n = len(top)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.9 * nrows),
                                  sharex=True, sharey=True, squeeze=False)
    axes = axes_grid.ravel()
    for ax_sobrante in axes[n:]:  # si la rejilla queda incompleta, se ocultan los paneles sobrantes
        ax_sobrante.set_visible(False)
    rows = []
    for ax, (_, r) in zip(axes, top.iterrows()):
        g = preds[(preds.ticker == r.ticker) & (preds.horizon == r.horizon) & (preds.variant == r.variant)]
        proba = g["proba_dir"].to_numpy(dtype=np.float32)
        y = g["y_dir"].to_numpy()
        fpr, tpr, thr = roc_curve(y, proba)

        ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color="#999999", zorder=1)
        ax.plot(fpr, tpr, lw=2.0, color="#1b4965", zorder=3)
        ax.fill_between(fpr, tpr, alpha=0.10, color="#1b4965", zorder=2)

        # el umbral calibrado cambia en cada sesión (ver
        # expanding_calibrated_threshold), así que ningún punto de la curva lo
        # representa: se marca el punto de operación real, la tasa de aciertos
        # y de falsas alarmas de las predicciones
        d = dev.loc[(r.variant, r.ticker, r.horizon)]
        thr_series = expanding_calibrated_threshold(
            proba, float(d.tasa_positiva_calibrada), float(d.umbral_calibrado))
        pred = proba > thr_series
        tp = float(((pred == 1) & (y == 1)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
        fpr_real = fp / (fp + tn) if (fp + tn) else 0.0
        tpr_real = tp / (tp + fn) if (tp + fn) else 0.0
        ax.plot(fpr_real, tpr_real, "o", ms=8, mfc="#d1495b", mec="white", mew=1.2, zorder=5)

        ax.set_title(f"{r.ticker} · h = {r.horizon}\n{r.variant}", fontsize=10)
        ax.text(0.97, 0.05,
                f"AUC = {r.roc_auc:.3f}\nκ = {r.cohen_kappa_calibrado:.3f}\np = {r.pt_p_value_calibrado:.6f}".replace(".", ","),
                transform=ax.transAxes, fontsize=8.2, ha="right", va="bottom")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

        rows.append(pd.DataFrame({
            "ticker": r.ticker, "horizon": r.horizon, "variant": r.variant,
            "fpr": fpr, "tpr": tpr, "threshold": thr,
        }))

    # etiquetas solo en los bordes: eje Y en la primera columna y eje X en el
    # panel inferior de cada columna, que no siempre está en la última fila si
    # la rejilla queda incompleta
    for fila in range(nrows):
        axes_grid[fila][0].set_ylabel("Tasa de verdaderos positivos")
    for col in range(ncols):
        ultima = max(f for f in range(nrows) if f * ncols + col < n)
        axes_grid[ultima][col].set_xlabel("Tasa de falsos positivos")
    # un solo rótulo para todos los paneles: con paneles estrechos, uno por
    # panel se solaparía con el recuadro de AUC, Kappa y p
    punto = Line2D([], [], marker="o", ls="", ms=8, mfc="#d1495b", mec="white", mew=1.2,
                   label="Punto de operación con el umbral calibrado causal")
    fig.legend(handles=[punto], frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.03 / nrows), fontsize=9)
    _save(fig, "image9.png")
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Kappa antes y después de calibrar, para ver de un vistazo en cuántas
# combinaciones la calibración ayuda y en cuántas empeora
# --------------------------------------------------------------------------
def figura_kappa(met: pd.DataFrame) -> pd.DataFrame:
    """Kappa de las 60 combinaciones con el umbral de 0,5 frente al umbral
    calibrado. Devuelve los datos dibujados."""
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    k0 = met["cohen_kappa"].to_numpy()
    k1 = met["cohen_kappa_calibrado"].to_numpy()
    mejora = (k1 - k0) > 1e-12

    lim = (-0.235, 0.275)
    ax.plot(lim, lim, ls="--", lw=1.1, color="#999999", zorder=1)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.fill_between(lim, lim, lim[1], color="#00798c", alpha=0.05, zorder=0)

    ax.scatter(k0[mejora], k1[mejora], s=46, c="#00798c", edgecolor="white",
               linewidth=0.7, zorder=3, label=f"Mejora ({int(mejora.sum())})")
    ax.scatter(k0[~mejora], k1[~mejora], s=46, c="#d1495b", edgecolor="white",
               linewidth=0.7, marker="s", zorder=3, label=f"Empeora ({int((~mejora).sum())})")

    peor = met.loc[(met.cohen_kappa_calibrado - met.cohen_kappa).idxmin()]
    ax.annotate(f"peor caso: {peor.ticker} h={peor.horizon}\n{peor.variant}",
                xy=(peor.cohen_kappa, peor.cohen_kappa_calibrado),
                xytext=(-0.225, -0.168), fontsize=8.4, color="#d1495b",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="#d1495b", lw=0.9))

    qqq60 = met[(met.ticker == "QQQ") & (met.horizon == 60)]
    ax.scatter(qqq60["cohen_kappa"], qqq60["cohen_kappa_calibrado"], s=150,
               facecolors="none", edgecolors="#d1495b", linewidth=1.3, zorder=2)
    ax.text(-0.225, 0.212, "círculos: las 5 variantes\nde QQQ a h = 60 (cambio\nde régimen desarrollo/prueba)",
            fontsize=8.2, color="#d1495b", ha="left", va="top", linespacing=1.35)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Kappa de Cohen con umbral por defecto (0,5)")
    ax.set_ylabel("Kappa de Cohen con umbral calibrado")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, ls=":", lw=0.6, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

    _save(fig, "image10.png")
    return met[["ticker", "horizon", "variant", "cohen_kappa", "cohen_kappa_calibrado"]]


# --------------------------------------------------------------------------
# RMSE de los ocho modelos en las doce combinaciones, que muestra que la
# ventaja de las líneas base no es homogénea
# --------------------------------------------------------------------------
def figura_rmse(met: pd.DataFrame, bmet: pd.DataFrame) -> pd.DataFrame:
    """Mapa de calor del RMSE de volatilidad de los ocho modelos en las doce
    combinaciones activo × horizonte. Devuelve los datos dibujados."""
    a = met[["ticker", "horizon", "variant", "rmse_rv"]].rename(columns={"variant": "modelo"})
    b = bmet[["ticker", "horizon", "model", "rmse_rv"]].rename(columns={"model": "modelo"})
    b["modelo"] = b["modelo"].map(BASELINE_LABEL)
    full = pd.concat([a, b], ignore_index=True)
    full["combo"] = full["ticker"] + "\nh=" + full["horizon"].astype(str)

    order_mod = full.groupby("modelo")["rmse_rv"].median().sort_values().index.tolist()
    order_combo = [f"{t}\nh={h}" for h in HORIZONS for t in TICKERS]
    M = full.pivot(index="modelo", columns="combo", values="rmse_rv").loc[order_mod, order_combo]

    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    im = ax.imshow(M.to_numpy(), cmap="YlOrRd", aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=0.009, vmax=0.115))

    ax.set_xticks(np.arange(len(order_combo)))
    ax.set_xticklabels(order_combo, fontsize=8.4)
    ax.set_yticks(np.arange(len(order_mod)))
    ax.set_yticklabels(order_mod, fontsize=9)
    for i in range(len(order_mod)):
        for j in range(len(order_combo)):
            val = M.to_numpy()[i, j]
            ax.text(j, i, f"{val:.4f}".replace(".", ","), ha="center", va="center",
                    fontsize=7.4, color="white" if val > 0.045 else "#222222")
    for j in (3.5, 7.5):
        ax.axvline(j, color="white", lw=2.4)
    # las líneas base se distinguen por el color de su etiqueta y no con una
    # línea divisoria, porque las filas se ordenan por RMSE mediano y en otra
    # ejecución podrían no quedar agrupadas
    for lab in ax.get_yticklabels():
        es_base = lab.get_text() in BASELINE_LABEL.values()
        lab.set_color("#1b4965" if es_base else "#7d5ba6")
        lab.set_fontweight("bold" if es_base else "normal")

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.028)
    cb.set_label("RMSE de volatilidad realizada (escala log)", fontsize=9)
    ax.set_xlabel("Activo y horizonte de predicción")
    ax.tick_params(axis="both", length=0)

    _save(fig, "image11.png")
    return full[["ticker", "horizon", "modelo", "rmse_rv"]].sort_values(["modelo", "horizon", "ticker"])


def main() -> None:
    """Genera las cuatro figuras y guarda sus datos en CSV y en un libro de
    Excel."""
    FIGDATA_DIR.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(RESULTS_DIR / "consolidado_metricas_variantes.csv")
    bmet = pd.read_csv(RESULTS_DIR / "consolidado_metricas_baselines.csv")
    preds = pd.read_csv(RESULTS_DIR / "consolidado_predicciones_variantes.csv", parse_dates=["date"])

    print("Generando figuras:")
    d1 = figura_auc(met)
    d2 = figura_roc(met, preds)
    d3 = figura_kappa(met)
    d4 = figura_rmse(met, bmet)

    datos = {
        "figura_5_1_auc": d1,
        "figura_5_2_roc": d2,
        "figura_5_3_kappa": d3,
        "figura_5_4_rmse": d4,
    }
    print("\nGenerando datos de respaldo:")
    for name, df in datos.items():
        p = FIGDATA_DIR / f"datos_{name}.csv"
        df.to_csv(p, index=False)
        print(f"  -> {p} ({len(df)} filas)")

    xlsx = FIGDATA_DIR / "datos_figuras_capitulo5.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in datos.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
        met.to_excel(xw, sheet_name="metricas_variantes", index=False)
        bmet.to_excel(xw, sheet_name="metricas_baselines", index=False)
    print(f"  -> {xlsx}")


if __name__ == "__main__":
    main()
