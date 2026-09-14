"""Lectura y escritura de los ficheros de hiperparámetros, que usan main.py,
train_final_models.py, holdout_evaluation.py y los diagnósticos.

Cada horizonte tiene su propio fichero,
results/best_hyperparameters{etiqueta}_h{H}.json. Además de los hiperparámetros
de cada variante, contiene un bloque `_meta` que registra con qué activos,
horizonte y configuración se obtuvieron. Antes de sobrescribir un fichero con
contenido distinto se pide confirmación, para que reemplazar unos
hiperparámetros sea siempre una decisión consciente.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_best_hyperparameters(path: Path) -> dict:
    """Carga el JSON de hiperparámetros tal cual. La clave `_meta` queda
    incluida, pero no interfiere: el código que los usa accede solo por
    nombre de variante."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_best_hyperparameters(path: Path, best_params_all: dict, meta: dict) -> None:
    """Guarda los hiperparámetros de cada variante junto con el bloque
    `_meta`. Si el fichero ya existe con contenido distinto, pide
    confirmación antes de reemplazarlo y, si se responde que no, no escribe
    nada. Si el contenido es idéntico, lo sobrescribe sin preguntar."""
    payload = {**best_params_all, "_meta": meta}

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return

        existing_meta = existing.get("_meta", {})
        print(f"\n{path.name} ya existe y se va a reemplazar:")
        print(f"  actual: {existing_meta}")
        print(f"  nuevo:  {meta}")
        answer = input("¿Confirmas que quieres sobrescribirlo? [s/N] ").strip().lower()
        if answer not in ("s", "si", "sí", "y", "yes"):
            print(f"Cancelado: {path.name} no se ha modificado.")
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hiperparámetros guardados en {path}")


def check_hyperparameters_provenance(
    best_hp: dict,
    *,
    horizon: int | None = None,
    tickers: tuple[str, ...] | None = None,
    lookback: int | None = None,
    garch: str | None = None,
) -> list[str]:
    """Compara el bloque `_meta` de unos hiperparámetros con la
    configuración con la que se van a usar y devuelve la lista de
    diferencias, vacía si coinciden.

    Reutilizar hiperparámetros de otro horizonte puede ser razonable en una
    exploración rápida, pero debe ser explícito, porque el óptimo cambia con
    el horizonte: por eso se avisa en lugar de fallar. Si falta `_meta`, la
    procedencia se considera desconocida."""
    meta = best_hp.get("_meta")
    if meta is None:
        return ["el fichero no tiene bloque `_meta`: procedencia desconocida"]

    desajustes = []
    if horizon is not None and meta.get("horizon") != horizon:
        desajustes.append(f"horizonte: se pide h={horizon}, pero se buscaron para h={meta.get('horizon')}")
    if tickers is not None and meta.get("tickers") is not None:
        if tuple(meta["tickers"]) != tuple(tickers):
            desajustes.append(f"activos: se piden {list(tickers)}, pero se buscaron con {meta['tickers']}")
    if lookback is not None and meta.get("lookback") != lookback:
        desajustes.append(f"lookback: se pide L={lookback}, pero se buscaron con L={meta.get('lookback')}")
    if garch is not None and meta.get("garch") != garch:
        desajustes.append(f"GARCH: se pide '{garch}', pero se buscaron con '{meta.get('garch')}'")
    return desajustes


def warn_if_provenance_mismatch(best_hp: dict, path: Path, **esperado) -> list[str]:
    """Muestra un aviso si `check_hyperparameters_provenance` encuentra
    diferencias y devuelve la lista."""
    desajustes = check_hyperparameters_provenance(best_hp, **esperado)
    if desajustes:
        print(f"  !! AVISO: {path.name} no corresponde a esta configuración:")
        for d in desajustes:
            print(f"     - {d}")
        print("     Se usan igualmente. Si quieres los propios, lanza la búsqueda con --hpo")
        print("     o pasa el fichero correcto con --hyperparams.")
    return desajustes


DEFAULT_RESULTS_DIR = Path("results")


def hyperparameters_path_for(horizon: int, results_dir: Path | None = None, tag: str = "") -> Path:
    """Ruta de los hiperparámetros de un horizonte:
    `results/best_hyperparameters{tag}_h{H}.json`.

    El horizonte figura siempre en el nombre, para que no se puedan
    confundir los hiperparámetros de horizontes distintos. `tag` identifica
    búsquedas alternativas, como `_textproj32`."""
    base = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    return base / f"best_hyperparameters{tag}_h{horizon}.json"


def available_horizons(results_dir: Path | None = None, tag: str = "") -> list[int]:
    """Horizontes que ya tienen fichero de hiperparámetros, ordenados. Se
    usa para que los mensajes de error indiquen qué hay disponible."""
    base = DEFAULT_RESULTS_DIR if results_dir is None else results_dir
    if not base.exists():
        return []
    encontrados = []
    for p in base.glob(f"best_hyperparameters{tag}_h*.json"):
        resto = p.stem[len(f"best_hyperparameters{tag}_h"):]
        if resto.isdigit():
            encontrados.append(int(resto))
    return sorted(encontrados)


def require_hyperparameters_for(
    horizon: int,
    results_dir: Path | None = None,
    tag: str = "",
    tickers: tuple[str, ...] | None = None,
) -> tuple[Path, dict]:
    """Devuelve `(ruta, hiperparámetros)` del horizonte pedido o falla con
    un mensaje que explica qué ejecutar.

    Nunca recurre al fichero de otro horizonte: el óptimo cambia con el
    horizonte, así que usar los de un horizonte vecino no está justificado."""
    path = hyperparameters_path_for(horizon, results_dir, tag)
    if path.exists():
        return path, load_best_hyperparameters(path)

    disponibles = available_horizons(results_dir, tag)
    lineas = [
        f"No hay hiperparámetros para h={horizon}: falta {path}.",
        "",
        "La búsqueda de Optuna se hace una vez por horizonte, y no se reutilizan",
        "los de otro: el régimen óptimo del modelo cambia con el horizonte.",
        "",
        f"Lánzala para h={horizon} con:",
        f"    python main.py {' '.join(tickers) if tickers else 'SPY QQQ KO GS'} --horizon {horizon} --hpo",
    ]
    if disponibles:
        lineas += ["", f"Horizontes que sí tienen hiperparámetros ya calculados: {disponibles}."]
        lineas += ["Si de verdad quieres reutilizar uno de esos (exploración, no resultado", 
                   "definitivo), pásalo explícitamente con --hyperparams y quedará registrado."]
    else:
        lineas += ["", "Todavía no hay hiperparámetros calculados para ningún horizonte."]
    raise FileNotFoundError("\n".join(lineas))
