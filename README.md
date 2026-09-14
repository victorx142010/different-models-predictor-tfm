# Framework híbrido de análisis predictivo para activos financieros

Código del Trabajo Fin de Máster del Máster Universitario en Tecnologías del
Sector Financiero: Fintech (UC3M, 2025-2026).

El modelo combina el histórico de precios (GARCH + LSTM) con titulares de
noticias financieras (FinBERT) para predecir la **dirección** y la
**volatilidad** de SPY, QQQ, KO y GS a 5, 20 y 60 sesiones. Compara cinco
formas de fusionar precio y texto frente a tres líneas base clásicas.

La metodología y los resultados están explicados en la memoria del TFM.

## Instalación

Requiere Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para usar GPU NVIDIA, reinstala PyTorch con CUDA:

```bash
.venv\Scripts\python.exe -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0+cu130
```

## Uso

Todo el pipeline se lanza desde `main.py`:

```bash
.venv\Scripts\python.exe main.py SPY QQQ KO GS --horizon 5 --hpo --baselines --holdout
```

La primera vez que se usa un activo, descarga sus precios (Yahoo Finance) y
sus noticias (FNSPID, en Hugging Face).

| Opción | Qué hace |
|---|---|
| `--horizon H` | Sesiones hacia delante que se predicen |
| `--variants ...` | Entrena solo algunas variantes (por defecto, las cinco) |
| `--hpo` | Busca hiperparámetros con Optuna (sin esta opción, reutiliza los guardados en `results/`) |
| `--baselines` | Evalúa también las tres líneas base |
| `--holdout` | Evalúa una única vez sobre 2022-2023 |
| `--device cpu` | Fuerza la CPU |

Para ver todas las opciones: `main.py --help`.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

## Estructura

```
main.py        punto de entrada del pipeline
src/           código: ingesta, GARCH, FinBERT, modelos, validación y evaluación
tests/         pruebas automatizadas
data/          precios y noticias descargados
features/      variables calculadas (GARCH y embeddings)
models/        modelos entrenados
results/       métricas, hiperparámetros y figuras
```

`data/`, `features/`, `models/` y `results/` no se versionan: se generan al
ejecutar el pipeline.

## Autor

Victor Sevillano Macuri · Tutor: Fernando Fernández Rebollo
