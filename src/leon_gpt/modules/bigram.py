import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):
    """
    Really simple language mode. Each token reads off the logits for the next token from a lookup table.
    There is no context involved.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self._token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Lookup the distribution of next tokens given the current token.
        :param idx:
        :return:
        """
        return self._token_embedding_table(idx)


def demo_model() -> None:
    model = BigramLanguageModel(512)




if __name__ == '__main__':
    demo_model()
