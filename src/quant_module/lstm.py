"""Codificador LSTM de la rama cuantitativa.

Recibe una ventana de L sesiones y, en cada una, un vector con tres valores:
[log_return, cond_vol, std_resid], es decir, el retorno del día, la volatilidad
condicional GARCH y el residuo estandarizado. Para añadir otra variable
bastaría con cambiar `QUANT_INPUT_DIM`.

Tiene una o dos capas. El dropout entre capas solo se aplica con dos, porque
con una sola no hay nada entre lo que aplicarlo.

Devuelve dos salidas:
- `h_L`: el último estado oculto, que resume toda la ventana. Lo usan
  price_only, early_fusion y late_fusion, y en cross_attention hace de
  consulta.
- `hidden_seq`: los estados ocultos de todas las sesiones. Hoy no se usa; queda
  disponible por si se quisiera una atención que consulte varias sesiones y no
  solo la última.
"""

from __future__ import annotations

import torch
import torch.nn as nn

QUANT_INPUT_DIM = 3  # [log_return, cond_vol, std_resid]


class QuantLSTMEncoder(nn.Module):
    """Envoltorio de nn.LSTM que devuelve por separado el último estado
    oculto y la secuencia completa de estados."""

    def __init__(
        self,
        input_dim: int = QUANT_INPUT_DIM,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Entrada [B, L, input_dim]; salida (h_L [B, hidden_size],
        hidden_seq [B, L, hidden_size])."""
        hidden_seq, (h_n, c_n) = self.lstm(x)
        h_L = h_n[-1]  # con varias capas, h_n trae el estado final de cada una; se usa el de la última
        return h_L, hidden_seq
