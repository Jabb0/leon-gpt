from typing import Any, Union, Sequence

import torch
from composer.core import Batch
from composer.models import ComposerModel
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float

from leon_gpt.modules.self_attention import Block


class BigramLanguageModel(ComposerModel):

    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 num_layers: int,
                 num_heads_per_layer: int,
                 max_sequence_length: int,
                 dropout: float) -> None:
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self._positional_embeddings = nn.Embedding(max_sequence_length, embedding_size)
        self.max_sequence_length = max_sequence_length

        # This is the transformer
        self._blocks = nn.Sequential(
            *[Block(embedding_size, num_heads_per_layer, max_sequence_length, dropout) for _ in range(num_layers)],
            nn.LayerNorm(embedding_size)
        )

        self._linear_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, token_idx: Float[torch.Tensor, "batch tokens"]) -> Float[torch.Tensor, "batch tokens vocab_size"]:
        """
        Lookup the distribution of next tokens given the current token.
        :param token_idx: Indices of tokens in range [0, vocab_size)
        :return:
        """
        B, T = token_idx.shape

        token_embedding = self._token_embedding_table(token_idx)  # (B, T, C)
        position_embedding = self._positional_embeddings(torch.arange(T, device=token_idx.device))  # (T, C)
        x = token_embedding + position_embedding  # (B, T, C)
        x = self._blocks(x)
        # For each token predict a distribution over all available tokens.
        # If trained to predict the next token, this should be the next token.
        logits = self._linear_head(x)  # (B, T, vocab_size)

        return logits

    def loss(self, outputs: Any, batch: Float[torch.Tensor, "batch tokens"], *args, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        target = target.view(B * T)
        loss = F.cross_entropy(logits, target)
        return loss

