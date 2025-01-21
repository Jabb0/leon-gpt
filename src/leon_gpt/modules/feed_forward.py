import torch
import torch.nn as nn

from jaxtyping import Float

class FeedForward(nn.Module):
    def __init__(self, embedding_features: int, dropout: float) -> None:
        super().__init__()
        self._layer = nn.Sequential(
            nn.Linear(embedding_features, 4 * embedding_features),
            nn.ReLU(),
            nn.Linear(4 * embedding_features, embedding_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: Float[torch.Tensor, "batch tokens embedding_features"]) -> Float[torch.Tensor, "batch tokens embedding_features"]:
        return self._layer(x)
