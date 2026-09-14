"""Entrena la variante cross_attention en los cuatro folds y la compara con las
variantes de control de train_control_variants.py.

Usa la misma metodología que las variantes de control (pérdida 1:1, AdamW, la
misma parada temprana, hiperparámetros por defecto y h=1), para que la única
diferencia sea la arquitectura. Como las otras cuatro variantes ya están
guardadas en results/control_variants_metrics_{activos}.parquet, solo entrena
cross_attention y, si existe ese fichero para los mismos activos, genera una
tabla con las cinco.

Formó parte de la comparación inicial de arquitecturas. Su función `to_tensors`
la reutilizan otros módulos, porque incluye las claves del conjunto de
noticias.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.fusion_model.base_model import MultiModalModel, multitask_loss
from src.ingestion.download_prices import DOWNLOAD_END, DOWNLOAD_START, TICKERS
from src.training.sequence_dataset import (
    build_dataset,
    build_quant_feature_table,
    load_daily_agg,
    load_variable_set,
)
from src.training.train_control_variants import (
    BATCH_SIZE,
    DROPOUT,
    GRAD_CLIP_NORM,
    HORIZON,
    L,
    LR,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    _rmse,
    scale_quant_seq,
)
from src.validation.splitter import WalkForwardSplitter
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
RESULTS_DIR = PROJECT_ROOT / "results"

VARIANT = "cross_attention"

TENSOR_KEYS = (
    "quant_seq",
    "text_today_emb",
    "text_today_has_news",
    "text_seq_emb",
    "text_seq_mask",
    "news_set_emb",
    "news_set_mask",
)


def to_tensors(ds: dict, device: str) -> dict[str, torch.Tensor]:
    """Como `to_tensors` de train_control_variants.py, pero incluye también
    news_set_emb y news_set_mask, que solo usa cross_attention."""
    return {
        "quant_seq": torch.tensor(ds["quant_seq"], dtype=torch.float32, device=device),
        "text_today_emb": torch.tensor(ds["text_today_emb"], dtype=torch.float32, device=device),
        "text_today_has_news": torch.tensor(ds["text_today_has_news"], dtype=torch.bool, device=device),
        "text_seq_emb": torch.tensor(ds["text_seq_emb"], dtype=torch.float32, device=device),
        "text_seq_mask": torch.tensor(ds["text_seq_mask"], dtype=torch.bool, device=device),
        "news_set_emb": torch.tensor(ds["news_set_emb"], dtype=torch.float32, device=device),
        "news_set_mask": torch.tensor(ds["news_set_mask"], dtype=torch.bool, device=device),
        "y_direction": torch.tensor(ds["y_direction"], dtype=torch.float32, device=device),
        "y_vol": torch.tensor(ds["y_vol"], dtype=torch.float32, device=device),
    }


def train_one(train_t: dict, val_t: dict, device: str) -> tuple[MultiModalModel, dict]:
    """Entrena cross_attention en un fold con el mismo bucle y la misma
    parada temprana que `train_one` de train_control_variants.py. Se
    mantiene aparte porque aquí la variante es fija y los tensores incluyen
    el conjunto de noticias."""
    torch.manual_seed(SEED)
    model = MultiModalModel(variant=VARIANT, dense_dropout=DROPOUT).to(device)
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
        dir_pred = (torch.sigmoid(out["direction_logit"]) > 0.5).float().cpu().numpy()
        vol_pred = out["vol_pred"].cpu().numpy()

    y_dir_true = val_t["y_direction"].cpu().numpy()
    y_vol_true = val_t["y_vol"].cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_dir_true, dir_pred),
        "f1": f1_score(y_dir_true, dir_pred, zero_division=0),
        "rmse_rv": _rmse(y_vol_true, vol_pred),
        "mae_rv": mean_absolute_error(y_vol_true, vol_pred),
        "n_epochs": epoch + 1,
        "best_val_loss": best_val_loss,
    }
    return model, metrics


def main() -> pd.DataFrame:
    """Entrena cross_attention en los folds de los activos indicados (por
    defecto, `TICKERS`) y guarda sus métricas. Si existen las métricas de
    las variantes de control para los mismos activos, genera además una
    tabla conjunta con una comparación directa frente a late_fusion, la
    variante de control más parecida: también separa precio y texto, pero
    los combina de forma fija."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=list(TICKERS), help="Tickers a entrenar (por defecto: TICKERS)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando {VARIANT} en {device}")
    splitter = WalkForwardSplitter()

    all_rows = []
    for ticker in args.tickers:
        quant_df = build_quant_feature_table(ticker)
        daily_agg = load_daily_agg(ticker)
        variable_set = load_variable_set(ticker)
        targets = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_targets_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )
        prices = pd.read_parquet(
            DATA_INTERIM_DIR / f"{ticker}_prices_clean_{DOWNLOAD_START}_{DOWNLOAD_END}.parquet"
        )

        for fold in splitter.folds:
            train_dates = splitter.slice_train(prices, fold)["date"]
            val_dates = splitter.slice_val(prices, fold)["date"]

            train_ds = build_dataset(
                train_dates, quant_df, daily_agg, targets, L, HORIZON, variable_set=variable_set
            )
            val_ds = build_dataset(
                val_dates, quant_df, daily_agg, targets, L, HORIZON, variable_set=variable_set
            )
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

            _, metrics = train_one(train_t, val_t, device)
            row = {
                "ticker": ticker,
                "fold": fold.name,
                "variant": VARIANT,
                "n_train": len(train_ds["date"]),
                "n_val": len(val_ds["date"]),
                "n_news_days_train": n_news_train,
                "n_news_days_val": n_news_val,
                **metrics,
            }
            all_rows.append(row)
            print(
                f"{ticker} {fold.name} {VARIANT}: acc={metrics['accuracy']:.3f} "
                f"f1={metrics['f1']:.3f} rmse_rv={metrics['rmse_rv']:.4f} "
                f"(epochs={metrics['n_epochs']})"
            )

    metrics_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tickers_tag = "_".join(t.lower() for t in args.tickers)
    out_path = RESULTS_DIR / f"cross_attention_metrics_{tickers_tag}.parquet"
    metrics_df.to_parquet(out_path, index=False)
    print(f"\n{len(metrics_df)} filas -> {out_path}")

    # Solo se combina con las métricas de control si son de los mismos activos;
    # si no, la comparación no sería válida. Como los activos forman parte del
    # nombre del fichero, basta con comprobar que existe.
    control_variants_path = RESULTS_DIR / f"control_variants_metrics_{tickers_tag}.parquet"
    if control_variants_path.exists():
        control_variants_df = pd.read_parquet(control_variants_path)
        combined = pd.concat([control_variants_df, metrics_df], ignore_index=True)
        combined_path = RESULTS_DIR / f"all_variants_metrics_{tickers_tag}.parquet"
        combined.to_parquet(combined_path, index=False)

        print("\n=== Comparación completa (mediana across ticker x fold) ===")
        summary = combined.groupby("variant")[["accuracy", "f1", "rmse_rv", "mae_rv"]].median().round(4)
        print(summary.to_string())
        summary.to_csv(RESULTS_DIR / f"all_variants_summary_median_{tickers_tag}.csv")

        print("\n=== late_fusion vs. cross_attention (comparación directa) ===")
        key_compare = combined[combined["variant"].isin(["late_fusion", "cross_attention"])]
        pivot = key_compare.pivot_table(
            index=["ticker", "fold"], columns="variant", values=["accuracy", "rmse_rv"]
        ).round(4)
        print(pivot.to_string())

    return metrics_df


if __name__ == "__main__":
    main()
