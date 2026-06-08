"""RiskPredictor architecture for the FinSight Phase 4 risk model.

Pipeline: LSTM over the time axis, multi-head self-attention over the LSTM
outputs, mean pool across time, and three independent linear heads that
predict (volatility, VaR-95, CVaR-95). The forward returns a single tensor
of shape [batch, 3] so the module is straightforward to export via
torch.onnx.export with opset 17.
"""

from __future__ import annotations

import torch
from torch import nn


class RiskPredictor(nn.Module):
    """Sequence-to-scalar model predicting three portfolio risk targets.

    Input shape:  [batch, T, input_dim] daily feature sequence.
    Output shape: [batch, 3] columns (volatility, var_95, cvar_95).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.lstm_layers: int = lstm_layers

        # nn.LSTM ignores the dropout argument when num_layers == 1 and emits
        # a warning. Pass 0.0 in that case so callers do not see noise.
        effective_lstm_dropout: float = lstm_dropout if lstm_layers > 1 else 0.0

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=effective_lstm_dropout,
            batch_first=True,
        )

        self.attention: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.vol_head: nn.Linear = nn.Linear(hidden_dim, 1)
        self.var_head: nn.Linear = nn.Linear(hidden_dim, 1)
        self.cvar_head: nn.Linear = nn.Linear(hidden_dim, 1)

        self._init_head_weights()

    @property
    def output_columns(self) -> tuple[str, str, str]:
        """Names of the three output columns, in the order forward() returns them."""
        return ("volatility", "var_95", "cvar_95")

    def _init_head_weights(self) -> None:
        for head in (self.vol_head, self.var_head, self.cvar_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full pipeline.

        Args:
            x: feature tensor of shape [batch, T, input_dim].

        Returns:
            Tensor of shape [batch, 3] with columns (volatility, var_95, cvar_95).
        """
        # LSTM: [batch, T, input_dim] -> [batch, T, hidden_dim].
        lstm_out, _ = self.lstm(x)

        # Self-attention with Q = K = V = lstm_out.
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, need_weights=False)

        # Mean pool across time: [batch, T, hidden_dim] -> [batch, hidden_dim].
        pooled: torch.Tensor = attn_out.mean(dim=1)

        # Three independent heads. Each produces [batch, 1]; squeeze last dim.
        vol: torch.Tensor = self.vol_head(pooled).squeeze(-1)
        var: torch.Tensor = self.var_head(pooled).squeeze(-1)
        cvar: torch.Tensor = self.cvar_head(pooled).squeeze(-1)

        # Stack along a new last dim -> [batch, 3].
        return torch.stack([vol, var, cvar], dim=-1)
