"""Atención cruzada: permite a la variante cross_attention mirar cada noticia
del día por separado y decidir cuánto peso darle, en lugar de usar un único
vector medio.

Sigue el esquema consulta-clave-valor de los Transformer:
- La consulta (Q) sale del último estado de la LSTM (`h_L`), que resume la
  historia del precio hasta hoy. Con ella se pregunta qué noticias del día son
  relevantes dado ese contexto.
- Las claves y los valores (K, V) salen de los embeddings FinBERT de las
  noticias del día: hasta M_MAX=32, con su máscara de relleno
  (variable_set_view.py).

    CrossAttn(Q, K, V) = softmax( QK^T / sqrt(d_k / n_heads) + M ) V

El resultado es un contexto que resume las noticias del día, con más peso para
las más relevantes. La atención es multicabeza (2 o 4 cabezas, elegidas en la
búsqueda de hiperparámetros), de modo que cada cabeza puede fijarse en un
aspecto distinto.

Los pesos de atención se pueden inspeccionar después para ver qué noticias
concretas influyeron más en la predicción de un día.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Se usa un valor muy negativo pero finito en lugar de -inf para anular el
# relleno en el softmax. Con -inf, un día sin ninguna noticia real (fila
# enmascarada entera) daría NaN (0/0); con -1e9 da una distribución casi
# uniforme sin NaN. En cualquier caso, MultiModalModel sustituye esos días por
# el contexto nulo aprendido.
MASK_FILL_VALUE = -1e9


class CrossAttention(nn.Module):
    """Atención multicabeza con una sola consulta por muestra (no una
    secuencia de consultas, como en un Transformer completo). Devuelve el
    contexto ponderado y los pesos de atención."""

    def __init__(
        self, query_dim: int, kv_dim: int, d_k: int = 64, n_heads: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_k % n_heads != 0:
            raise ValueError(f"d_k ({d_k}) debe ser divisible por n_heads ({n_heads})")
        self.d_k = d_k
        self.n_heads = n_heads
        self.head_dim = d_k // n_heads

        self.W_Q = nn.Linear(query_dim, d_k)
        self.W_K = nn.Linear(kv_dim, d_k)
        self.W_V = nn.Linear(kv_dim, d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, keys_values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula el contexto de atención de cada muestra.

        - query: [B, query_dim], el estado h_L.
        - keys_values: [B, M, kv_dim], embeddings de las noticias del día,
          con relleno.
        - mask: [B, M], booleana: True si es noticia real y False si es
          relleno.

        Devuelve context [B, d_k] y attn_weights [B, n_heads, M]."""
        B, M, _ = keys_values.shape

        # proyecta Q, K, V y los reparte entre las n_heads cabezas: cada
        # cabeza trabaja con una porción de tamaño head_dim = d_k / n_heads
        Q = self.W_Q(query).view(B, self.n_heads, 1, self.head_dim)
        K = self.W_K(keys_values).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(keys_values).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)

        # producto escalar Q·K escalado por sqrt(head_dim), para que el softmax
        # no se sature cuando head_dim es grande
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)  # [B, H, 1, M]
        mask_expanded = mask.view(B, 1, 1, M)
        scores = scores.masked_fill(~mask_expanded, MASK_FILL_VALUE)

        attn = torch.softmax(scores, dim=-1)  # [B, H, 1, M]: cuánto peso le da cada cabeza a cada noticia
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # [B, H, 1, head_dim]: promedio ponderado de V según los pesos de atención
        context = context.transpose(1, 2).reshape(B, self.d_k)  # se juntan de nuevo las cabezas en un solo vector

        return context, attn.squeeze(2)  # attn: [B, H, M]
