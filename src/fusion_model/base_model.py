"""Arquitectura del modelo: una sola clase, `MultiModalModel`, que implementa
las cinco variantes según el parámetro `variant`.

Usar una única clase garantiza que las variantes se comparan en igualdad de
condiciones: comparten la LSTM, las capas densas finales y el entrenamiento, y
solo cambia cómo se combinan el precio y las noticias. Con cinco clases
separadas sería fácil que se colara una diferencia accidental, como una capa de
más o una inicialización distinta.

Las cinco variantes:
- `price_only`: solo la LSTM sobre precio y GARCH. Es el control que responde a
  si el texto aporta algo.
- `text_only`: solo el embedding diario de noticias, sin precio. Mide cuánta
  señal hay en el texto por sí solo.
- `early_fusion`: una única LSTM que recibe, en cada sesión, las variables de
  precio junto con el embedding de texto de ese mismo día.
- `late_fusion`: precio y texto se procesan por separado y se concatenan antes
  de las capas finales. Comprueba si separar y concatenar basta, sin un
  mecanismo más elaborado.
- `cross_attention`: la variante principal. La LSTM resume el precio en `h_L`,
  que actúa como consulta de una atención sobre todas las noticias del día. El
  contexto resultante se concatena con `h_L` (conexión residual), de modo que
  el modelo nunca pierde la información de precio.

En los días sin noticias no se usa un valor fijo, como un vector de ceros, sino
un vector que el modelo aprende durante el entrenamiento (`NullEmbedding`). Así
es el propio optimizador el que decide cómo representar la ausencia de
noticias.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fusion_model.cross_attention import CrossAttention
from src.quant_module.lstm import QUANT_INPUT_DIM, QuantLSTMEncoder

VARIANTS = ("early_fusion", "late_fusion", "price_only", "text_only", "cross_attention")
EMBEDDING_DIM = 768


class NullEmbedding(nn.Module):
    """Vector aprendido que sustituye al embedding en los días sin noticias.
    Es un parámetro más del modelo, que el optimizador ajusta junto con el
    resto.

    Funciona tanto con un embedding por muestra [B, dim] (late_fusion y
    text_only) como con una secuencia [B, L, dim] (early_fusion, donde se
    sustituye en cada sesión sin noticias)."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.null_vec = nn.Parameter(torch.zeros(dim))

    def forward(self, emb: torch.Tensor, has_news: torch.Tensor) -> torch.Tensor:
        """Deja el embedding real donde has_news es True y lo sustituye por
        el vector aprendido donde es False."""
        mask = has_news.unsqueeze(-1)
        null = self.null_vec.expand_as(emb)
        return torch.where(mask, emb, null)


class MultiModalModel(nn.Module):
    """Modelo completo. En `__init__` construye solo las piezas que necesita
    cada variante (text_only no tiene LSTM y price_only no tiene capas de
    texto), y `forward` sigue el camino correspondiente. Las capas densas
    finales y las dos cabezas de salida, dirección y volatilidad, son las
    mismas en las cinco variantes: solo cambia el vector que llega hasta
    ellas."""

    def __init__(
        self,
        variant: str,
        quant_input_dim: int = QUANT_INPUT_DIM,
        embedding_dim: int = EMBEDDING_DIM,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        lstm_dropout: float = 0.0,
        text_proj_dim: int = 64,
        dense_hidden: int = 64,
        dense_dropout: float = 0.2,
        attn_d_k: int = 64,
        attn_n_heads: int = 4,
        attn_dropout: float = 0.0,
        use_embedding_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant debe ser uno de {VARIANTS}, recibido '{variant}'")
        self.variant = variant
        self.uses_text = variant in ("early_fusion", "late_fusion", "text_only")

        # Los embeddings de FinBERT llegan sin normalizar (norma L2 de unos
        # 17), mientras que las variables de precio sí están estandarizadas.
        # Ese desajuste de escala afecta sobre todo a cross_attention, cuyas
        # puntuaciones Q·K dependen de la magnitud de K y V. Con
        # use_embedding_layernorm=True se aplica un LayerNorm al embedding
        # antes de cualquier rama de texto; por defecto está desactivado.
        self.use_embedding_layernorm = use_embedding_layernorm
        needs_emb_norm = use_embedding_layernorm and (self.uses_text or variant == "cross_attention")
        if needs_emb_norm:
            self.emb_norm = nn.LayerNorm(embedding_dim)

        if self.uses_text:
            self.null_embedding = NullEmbedding(embedding_dim)
            self.text_proj = nn.Linear(embedding_dim, text_proj_dim)

        if variant == "early_fusion":
            self.lstm = QuantLSTMEncoder(
                quant_input_dim + text_proj_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout
            )
            head_input_dim = lstm_hidden_size
        elif variant == "late_fusion":
            self.lstm = QuantLSTMEncoder(quant_input_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout)
            head_input_dim = lstm_hidden_size + text_proj_dim
        elif variant == "price_only":
            self.lstm = QuantLSTMEncoder(quant_input_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout)
            head_input_dim = lstm_hidden_size
        elif variant == "cross_attention":
            self.lstm = QuantLSTMEncoder(quant_input_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout)
            self.cross_attn = CrossAttention(
                query_dim=lstm_hidden_size,
                kv_dim=embedding_dim,
                d_k=attn_d_k,
                n_heads=attn_n_heads,
                dropout=attn_dropout,
            )
            self.null_context = nn.Parameter(torch.zeros(attn_d_k))
            head_input_dim = lstm_hidden_size + attn_d_k  # el contexto de atención se concatena con h_L, no lo sustituye (conexión residual)
        else:  # text_only
            head_input_dim = text_proj_dim

        self.dense = nn.Sequential(
            nn.Linear(head_input_dim, dense_hidden),
            nn.GELU(),
            nn.Dropout(dense_dropout),
        )
        self.direction_head = nn.Linear(dense_hidden, 1)
        self.vol_head = nn.Sequential(nn.Linear(dense_hidden, 1), nn.Softplus())

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Cada rama del if/elif implementa una variante, pero todas
        producen un único tensor `features`, que pasa por las mismas capas
        densas y las mismas dos cabezas de salida."""
        attn_weights = None

        if self.variant == "early_fusion":
            text_seq_emb = batch["text_seq_emb"]
            if self.use_embedding_layernorm:
                text_seq_emb = self.emb_norm(text_seq_emb)
            # en cada sesión sin noticias se sustituye por el vector nulo, se
            # proyecta a la dimensión de texto y se concatena con el precio
            # antes de la LSTM
            null_seq = self.null_embedding(text_seq_emb, batch["text_seq_mask"])
            text_proj_seq = self.text_proj(null_seq)  # [B, L, text_proj_dim]
            lstm_input = torch.cat([batch["quant_seq"], text_proj_seq], dim=-1)
            features, _ = self.lstm(lstm_input)
        elif self.variant == "late_fusion":
            # la LSTM procesa solo el precio; el texto del día se procesa
            # aparte y las dos ramas se unen al final
            h_L, _ = self.lstm(batch["quant_seq"])
            text_today_emb = batch["text_today_emb"]
            if self.use_embedding_layernorm:
                text_today_emb = self.emb_norm(text_today_emb)
            null_today = self.null_embedding(text_today_emb, batch["text_today_has_news"])
            text_feat = self.text_proj(null_today)
            features = torch.cat([h_L, text_feat], dim=-1)
        elif self.variant == "price_only":
            features, _ = self.lstm(batch["quant_seq"])
        elif self.variant == "cross_attention":
            h_L, _ = self.lstm(batch["quant_seq"])
            news_set_emb = batch["news_set_emb"]
            if self.use_embedding_layernorm:
                news_set_emb = self.emb_norm(news_set_emb)
            context, attn_weights = self.cross_attn(h_L, news_set_emb, batch["news_set_mask"])
            # si el día no tiene ninguna noticia no hay nada a lo que atender
            # (la atención daría una distribución uniforme sobre relleno, sin
            # significado): se usa directamente el contexto nulo aprendido
            has_any_news = batch["news_set_mask"].any(dim=-1)
            context = torch.where(has_any_news.unsqueeze(-1), context, self.null_context.expand_as(context))
            features = torch.cat([context, h_L], dim=-1)  # conexión residual: el modelo nunca pierde el acceso directo a h_L
        else:  # text_only
            text_today_emb = batch["text_today_emb"]
            if self.use_embedding_layernorm:
                text_today_emb = self.emb_norm(text_today_emb)
            null_today = self.null_embedding(text_today_emb, batch["text_today_has_news"])
            features = self.text_proj(null_today)

        dense_out = self.dense(features)
        direction_logit = self.direction_head(dense_out).squeeze(-1)
        vol_pred = self.vol_head(dense_out).squeeze(-1)
        return {"direction_logit": direction_logit, "vol_pred": vol_pred, "attn_weights": attn_weights}


def multitask_loss(
    direction_logit: torch.Tensor,
    vol_pred: torch.Tensor,
    y_direction: torch.Tensor,
    y_vol: torch.Tensor,
    lambda_dir: float = 1.0,
    lambda_vol: float = 1.0,
    pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pérdida multitarea con pesos fijos: entropía cruzada binaria (BCE)
    para la dirección y Huber para la volatilidad, combinadas con los pesos
    `lambda_dir` y `lambda_vol`. Se usa Huber en lugar del error cuadrático
    porque penaliza menos los errores grandes puntuales, frecuentes en los
    picos de volatilidad.

    `pos_weight` (por defecto None) cambia el peso de los ejemplos de subida
    dentro de la BCE: un valor menor que 1 resta peso a la clase
    mayoritaria. Solo se usó en los diagnósticos pos_weight_*.py, como
    alternativa a calibrar el umbral."""
    pw = None if pos_weight is None else torch.tensor(pos_weight, device=direction_logit.device)
    bce = F.binary_cross_entropy_with_logits(direction_logit, y_direction, pos_weight=pw)
    huber = F.smooth_l1_loss(vol_pred, y_vol)
    total = lambda_dir * bce + lambda_vol * huber
    return total, {"bce": bce.item(), "huber": huber.item()}


class UncertaintyWeightedLoss(nn.Module):
    """Alternativa a los pesos fijos: el entrenamiento aprende cuánto pesa
    cada tarea, ponderándolas por su incertidumbre:

        L = e^(-s_dir) BCE + s_dir + e^(-s_vol) Huber + s_vol

    `s_dir` y `s_vol` son parámetros entrenables. Si una tarea es más
    ruidosa, el modelo sube su `s` y así reduce su peso; el término `+ s`
    impide subirlo sin límite para anular la tarea.

    Como estos parámetros pertenecen a la pérdida y no al modelo, hay que
    añadirlos a los parámetros del optimizador; si no, nunca se actualizan."""

    def __init__(self) -> None:
        super().__init__()
        self.s_dir = nn.Parameter(torch.zeros(()))
        self.s_vol = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        direction_logit: torch.Tensor,
        vol_pred: torch.Tensor,
        y_direction: torch.Tensor,
        y_vol: torch.Tensor,
        pos_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calcula BCE y Huber como `multitask_loss`, pero las combina con
        s_dir y s_vol. `pos_weight` funciona igual que en `multitask_loss`."""
        pw = None if pos_weight is None else torch.tensor(pos_weight, device=direction_logit.device)
        bce = F.binary_cross_entropy_with_logits(direction_logit, y_direction, pos_weight=pw)
        huber = F.smooth_l1_loss(vol_pred, y_vol)
        total = (
            torch.exp(-self.s_dir) * bce + self.s_dir + torch.exp(-self.s_vol) * huber + self.s_vol
        )
        return total, {
            "bce": bce.item(),
            "huber": huber.item(),
            "s_dir": self.s_dir.item(),
            "s_vol": self.s_vol.item(),
        }
