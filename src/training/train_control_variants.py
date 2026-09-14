"""Funciones de entrenamiento compartidas y script de la comparación inicial de
arquitecturas.

Algunas funciones, como `scale_quant_seq`, las reutilizan otros módulos. Como
script, entrena las cuatro variantes de control (price_only, text_only,
late_fusion y early_fusion) en los cuatro folds, con pesos de pérdida 1:1,
hiperparámetros por defecto, horizonte h=1 y ventana L=20. Fue la primera
comparación de arquitecturas, previa a la búsqueda de hiperparámetros; los
resultados finales se obtienen con main.py.

El `StandardScaler` de `quant_seq` se ajusta solo con el entrenamiento de cada
fold y después se aplica, fijo, a la validación. Ajustarlo con datos de
validación sería una fuga de información.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.evaluation.financial_metrics import classification_agreement_metrics
from sklearn.preprocessing import StandardScaler

from src.fusion_model.base_model import VARIANTS, MultiModalModel, multitask_loss
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.training.sequence_dataset import build_dataset, build_quant_feature_table, load_daily_agg
from src.validation.splitter import WalkForwardSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

L = 20
HORIZON = 1
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
LR = 5e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
DROPOUT = 0.2
SEED = 42

TENSOR_KEYS = (
    "quant_seq",
    "text_today_emb",
    "text_today_has_news",
    "text_seq_emb",
    "text_seq_mask",
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raíz del error cuadrático medio."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def scale_quant_seq(train_seq: np.ndarray, val_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estandariza las secuencias de precio (media 0, varianza 1). El
    escalador se ajusta solo con `train_seq` y se aplica igual a `val_seq`,
    para no filtrar estadísticas de validación. Como StandardScaler trabaja
    con matrices 2D, las secuencias [N, L, variables] se aplanan a [N·L,
    variables] y se recomponen después."""
    scaler = StandardScaler()
    n, l, f = train_seq.shape
    scaler.fit(train_seq.reshape(-1, f))
    train_scaled = scaler.transform(train_seq.reshape(-1, f)).reshape(n, l, f)
    n2, l2, f2 = val_seq.shape
    val_scaled = scaler.transform(val_seq.reshape(-1, f2)).reshape(n2, l2, f2)
    return train_scaled.astype("float32"), val_scaled.astype("float32")


def to_tensors(ds: dict, device: str) -> dict[str, torch.Tensor]:
    """Convierte el diccionario de arrays de `build_dataset` en tensores de
    PyTorch en el dispositivo indicado."""
    out = {
        "quant_seq": torch.tensor(ds["quant_seq"], dtype=torch.float32, device=device),
        "text_today_emb": torch.tensor(ds["text_today_emb"], dtype=torch.float32, device=device),
        "text_today_has_news": torch.tensor(ds["text_today_has_news"], dtype=torch.bool, device=device),
        "text_seq_emb": torch.tensor(ds["text_seq_emb"], dtype=torch.float32, device=device),
        "text_seq_mask": torch.tensor(ds["text_seq_mask"], dtype=torch.bool, device=device),
        "y_direction": torch.tensor(ds["y_direction"], dtype=torch.float32, device=device),
        "y_vol": torch.tensor(ds["y_vol"], dtype=torch.float32, device=device),
    }
    return out


def train_one(variant: str, train_t: dict, val_t: dict, device: str) -> tuple[MultiModalModel, dict]:
    """Entrena una variante en un fold con parada temprana: si la pérdida de
    validación no mejora en `PATIENCE` épocas seguidas, se detiene y
    recupera el mejor estado, no el último. Devuelve el modelo y sus
    métricas de validación."""
    torch.manual_seed(SEED)
    model = MultiModalModel(variant=variant, dense_dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n = train_t["quant_seq"].shape[0]
    idx = np.arange(n)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(idx)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start : start + BATCH_SIZE]
            batch = {k: train_t[k][batch_idx] for k in TENSOR_KEYS}
            y_dir = train_t["y_direction"][batch_idx]
            y_vol = train_t["y_vol"][batch_idx]

            optimizer.zero_grad()
            out = model(batch)
            loss, _ = multitask_loss(out["direction_logit"], out["vol_pred"], y_dir, y_vol)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_batch = {k: val_t[k] for k in TENSOR_KEYS}
            out = model(val_batch)
            val_loss, _ = multitask_loss(
                out["direction_logit"], out["vol_pred"], val_t["y_direction"], val_t["y_vol"]
            )
        val_loss_val = val_loss.item()

        if val_loss_val < best_val_loss - 1e-5:
            best_val_loss = val_loss_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_batch = {k: val_t[k] for k in TENSOR_KEYS}
        out = model(val_batch)
        # se guarda la probabilidad, no solo la clase: la necesita el AUC, que
        # es la métrica que detecta el colapso a clase constante (una
        # predicción siempre igual da una accuracy engañosamente alta, pero
        # Kappa 0)
        dir_proba = torch.sigmoid(out["direction_logit"]).cpu().numpy()
        dir_pred = (dir_proba > 0.5).astype(float)
        vol_pred = out["vol_pred"].cpu().numpy()

    y_dir_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_dir_true, dir_pred),
        "f1": f1_score(y_dir_true, dir_pred, zero_division=0),
        "rmse_rv": _rmse(y_vol_true, vol_pred),
        "mae_rv": mean_absolute_error(y_vol_true, vol_pred),
        **classification_agreement_metrics(dir_proba, dir_pred, y_dir_true),
        "n_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
    }
    return model, metrics


def main() -> pd.DataFrame:
    """Entrena las cuatro variantes de control en los folds de desarrollo de
    los activos indicados (por defecto, `TICKERS`) y guarda sus métricas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a entrenar (por defecto: TICKERS)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en {device}")
    splitter = WalkForwardSplitter()

    all_rows = []
    for ticker in args.tickers:
        quant_df = build_quant_feature_table(ticker)
        daily_agg = load_daily_agg(ticker)
        targets = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        prices = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )

        for fold in splitter.folds:
            train_dates = splitter.slice_train(prices, fold)["date"]
            val_dates = splitter.slice_val(prices, fold)["date"]

            train_ds = build_dataset(train_dates, quant_df, daily_agg, targets, L, HORIZON)
            val_ds = build_dataset(val_dates, quant_df, daily_agg, targets, L, HORIZON)
            if train_ds is None or val_ds is None:
                print(f"[WARN] sin muestras para {ticker} {fold.name}, saltando")
                continue

            train_quant_scaled, val_quant_scaled = scale_quant_seq(
                train_ds["quant_seq"], val_ds["quant_seq"]
            )
            train_ds = dict(train_ds, quant_seq=train_quant_scaled)
            val_ds = dict(val_ds, quant_seq=val_quant_scaled)

            train_t = to_tensors(train_ds, device)
            val_t = to_tensors(val_ds, device)

            n_news_train = int(train_t["text_today_has_news"].sum().item())
            n_news_val = int(val_t["text_today_has_news"].sum().item())

            for variant in VARIANTS:
                _, metrics = train_one(variant, train_t, val_t, device)
                row = {
                    "ticker": ticker,
                    "fold": fold.name,
                    "variant": variant,
                    "n_train": len(train_ds["date"]),
                    "n_val": len(val_ds["date"]),
                    "n_news_days_train": n_news_train,
                    "n_news_days_val": n_news_val,
                    **metrics,
                }
                all_rows.append(row)
                print(
                    f"{ticker} {fold.name} {variant}: acc={metrics['accuracy']:.3f} "
                    f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f} "
                    f"(epochs={metrics['n_epochs']})"
                )

    metrics_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_path = RESULTS_DIR / f"control_variants_metrics_{tickers_tag}.parquet"
    metrics_df.to_parquet(out_path, index=False)
    print(f"\n{len(metrics_df)} filas -> {out_path}")

    summary = metrics_df.groupby("variant")[["accuracy", "f1", "rmse_rv", "mae_rv"]].mean().round(4)
    print("\nPromedio por variante (across ticker x fold):")
    print(summary.to_string())

    summary_path = RESULTS_DIR / f"control_variants_summary_{tickers_tag}.csv"
    summary.to_csv(summary_path)
    print(f"-> {summary_path}")

    return metrics_df


if __name__ == "__main__":
    main()
