"""Lanza calibration_check_worker.py para las cinco variantes en paralelo, cada
una en un proceso independiente, y junta sus resultados.

Las cinco comprobaciones no dependen unas de otras y los modelos son pequeños,
así que caben a la vez en la GPU.

Salida: results/calibration_check_{variante}_h{H}.csv y .log de cada variante,
y results/calibration_check_all_variants_h{H}.csv con todas juntas.

Uso:
    python -m src.training.calibration_check_launcher --horizon 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
VARIANTS = ("price_only", "text_only", "late_fusion", "early_fusion", "cross_attention")


def main() -> None:
    """Lanza los cinco procesos, espera a que terminen, junta sus resultados
    y muestra cuántas combinaciones colapsan con el umbral de 0,5 y con el
    calibrado."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=5)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {v: RESULTS_DIR / f"calibration_check_{v}_h{args.horizon}.csv" for v in VARIANTS}

    print(f"Lanzando {len(VARIANTS)} procesos en paralelo (h={args.horizon}): {VARIANTS}")
    t0 = time.time()
    procs = {}
    for v in VARIANTS:
        cmd = [
            sys.executable, "-m", "src.training.calibration_check_worker",
            "--variant", v, "--horizon", str(args.horizon), "--out", str(out_paths[v]),
        ]
        log_path = RESULTS_DIR / f"calibration_check_{v}_h{args.horizon}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        procs[v] = (subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT), log_file)
        print(f"  [{v}] PID {procs[v][0].pid}, log en {log_path}")

    for v, (p, log_file) in procs.items():
        ret = p.wait()
        log_file.close()
        status = "OK" if ret == 0 else f"exit={ret}"
        print(f"  [{v}] terminado ({status})")

    t1 = time.time()
    print(f"\nTotal: {t1 - t0:.0f}s")

    dfs = []
    for v, path in out_paths.items():
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            print(f"  AVISO: no se generó {path} ({v} pudo haber fallado, revisar su .log)")
    if not dfs:
        print("Ningún resultado generado.")
        return

    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(RESULTS_DIR / f"calibration_check_all_variants_h{args.horizon}.csv", index=False)

    print(f"\n=== Resumen por variante (16 combinaciones c/u, h={args.horizon}) ===")
    for v in VARIANTS:
        sub = full[full["variant"] == v]
        if sub.empty:
            continue
        print(
            f"{v:16s}  umbral 0.5: {sub['colapsado_05'].sum()}/16 colapsadas, kappa medio={sub['kappa_05'].mean():.3f}"
            f"   |   calibrado: {sub['colapsado_cal'].sum()}/16 colapsadas, kappa medio={sub['kappa_cal'].mean():.3f}"
        )


if __name__ == "__main__":
    main()
