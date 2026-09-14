"""Lanza en paralelo recalibrate_holdout_worker.py para las cinco variantes, un
proceso por variante.

Cada proceso entrena su variante para los 4 activos y los 3 horizontes (12
combinaciones) sobre 2015-2021 y la evalúa sobre el holdout 2022-2023, con el
umbral por defecto y con el umbral calibrado en desarrollo: 60 combinaciones en
total. Es el paso 3 para reproducir los resultados del holdout, después de
main.py y compute_dev_thresholds.py.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
VARIANTS = ("price_only", "text_only", "late_fusion", "early_fusion", "cross_attention")


def main() -> None:
    """Lanza los cinco procesos, espera a que terminen y une sus resultados
    en results/holdout_recalibrado_todas_variantes.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {v: RESULTS_DIR / f"holdout_recalibrado_{v}.csv" for v in VARIANTS}

    print(f"Lanzando {len(VARIANTS)} procesos en paralelo (3 horizontes x 4 tickers c/u): {VARIANTS}")
    t0 = time.time()
    procs = {}
    for v in VARIANTS:
        cmd = [sys.executable, "-m", "src.training.recalibrate_holdout_worker", "--variant", v, "--out", str(out_paths[v])]
        log_path = RESULTS_DIR / f"recalibrate_holdout_{v}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        procs[v] = (subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT), log_file)
        print(f"  [{v}] PID {procs[v][0].pid}, log en {log_path}")

    for v, (p, log_file) in procs.items():
        ret = p.wait()
        log_file.close()
        print(f"  [{v}] terminado (exit={ret})")

    print(f"\nTotal: {time.time() - t0:.0f}s")

    dfs = []
    for v, path in out_paths.items():
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"  AVISO: no se generó {path}, revisar su .log")
    if not dfs:
        return

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(RESULTS_DIR / "holdout_recalibrado_todas_variantes.csv", index=False)

    print("\n=== Resumen: original (0,5) vs calibrado, por variante (mediana sobre 12 combinaciones) ===")
    for v in VARIANTS:
        sub = full[full["variant"] == v]
        if sub.empty:
            continue
        n_colapso_05 = (sub["cohen_kappa"] == 0).sum()
        n_colapso_cal = (sub["cohen_kappa_calibrado"] == 0).sum() if "cohen_kappa_calibrado" in sub else None
        print(
            f"{v:16s}  acc {sub['accuracy'].median():.4f}->{sub['accuracy_calibrado'].median():.4f}  "
            f"f1 {sub['f1'].median():.4f}->{sub['f1_calibrado'].median():.4f}  "
            f"kappa {sub['cohen_kappa'].median():.4f}->{sub['cohen_kappa_calibrado'].median():.4f}  "
            f"colapso {n_colapso_05}/12->{n_colapso_cal}/12"
        )


if __name__ == "__main__":
    main()
