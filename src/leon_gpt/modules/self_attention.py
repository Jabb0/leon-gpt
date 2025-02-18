"""
Let's give Leon some attention.
"""
import torch
from jaxtyping import Float
from torch import nn as nn
from torch.nn import functional as F

from leon_gpt.modules.feed_forward import FeedForward


class SelfAttentionBlock(nn.Module):

    def __init__(self, input_features: int, max_sequence_length: int,
                       head_size: int, mask_future: bool,
                        dropout: float) -> None:
        super().__init__()
        self._keys = nn.Linear(input_features, head_size, bias=False)
        self._queries = nn.Linear(input_features, head_size, bias=False)
        self._values = nn.Linear(input_features, head_size, bias=False)
        if mask_future:
            self.register_buffer("tril", torch.tril(torch.ones(max_sequence_length, max_sequence_length)))
        self._mask_future = mask_future

        self._dropout = nn.Dropout(dropout)

    def forward(self, x: Float[torch.Tensor, "batch tokens features"]) -> Float[torch.Tensor, "batch tokens head_size"]:
        _, T, C = x.shape
        assert T <= self.tril.size(dim=0), "token dimension needs to be smaller than max_sequence_length"

        # (batch, tokens, head_size)
        k = self._keys(x)
        q = self._queries(x)
        # (batch, tokens, head_size) @ (batch, head_size, tokens) -> (batch, tokens, tokens)
        # Is the scalar product between query and key for all tokens combinations.
        # The normalization retains a smooth distribution independent of the number of input features
        weights = q @ k.transpose(-2, -1) * C**-0.5

        if self._mask_future:
            # Mask out tokens a token should not attend to. Which are tokens in the future.
            # Average over all tokens up to and including the token weighted as inferred above.
            weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # Softmax along the "attends to" dimension.
        # (batch, tokens, tokens)
        weights = F.softmax(weights, dim=-1)
        weights = self._dropout(weights)

        # (batch, tokens, head_size)
        v = self._values(x)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_features: int,
                       max_sequence_length: int, head_size: int, mask_future: bool,
                       dropout: float) -> None:
        super().__init__()
        self._heads = nn.ModuleList([SelfAttentionBlock(input_features, max_sequence_length, head_size, mask_future, dropout)
                                     for _ in range(num_heads)])
        embedding_dimension = num_heads * head_size
        self._proj = nn.Linear(embedding_dimension, embedding_dimension)
        self._dropout = nn.Dropout(dropout)

    def forward(self, x:  Float[torch.Tensor, "batch tokens features"]) -> Float[torch.Tensor, "batch tokens multi_head_size"]:
        out = torch.cat([h(x) for h in self._heads], dim=-1)
        out = self._dropout(self._proj(out))
        return out


class DecoderSelfAttentionBlock(SelfAttentionBlock):
    def __init__(self, input_features: int, max_sequence_length: int, head_size: int):
        super().__init__(input_features, max_sequence_length, head_size, mask_future=True)


class EncoderSelfAttentionBlock(SelfAttentionBlock):
    def __init__(self, input_features: int, max_sequence_length: int, head_size: int):
        super().__init__(input_features, max_sequence_length, head_size, mask_future=False)


class Block(nn.Module):
    def __init__(self,
                 embedding_features: int,
                 num_heads: int,
                 max_sequence_length: int,
                 dropout: float):
        """

        :param embedding_features: Input dimension C.
        :param num_heads:  Number of heads.
        :param max_sequence_length:  Largest sequence length T.
        """
        super().__init__()
        if embedding_features % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads.")
        head_size = embedding_features // num_heads
        self._self_attention = MultiHeadAttention(num_heads, embedding_features,
                                                  max_sequence_length,
                                                  head_size,
                                                  mask_future=True,
                                                  dropout=dropout
                                                  )

        self._feed_forward = FeedForward(embedding_features, dropout)
        self._ln1 = nn.LayerNorm(embedding_features)
        self._ln2 = nn.LayerNorm(embedding_features)

    def forward(self, x):
        x = x + self._self_attention(self._ln1(x))
        x = x + self._feed_forward(self._ln2(x))
        return x
