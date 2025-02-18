import abc
import torch

from leon_gpt.data.text_dataset import TextDataset


class Tokenizer(abc.ABC):
    """
    Tokenizer transform that turns a string into tokens.
    Tokenizers need to be initialized on the data before being used in a dataloader for training.
    Some tokenizers include optimization, others use fixed dictionaries.
    """

    @abc.abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """
        Turns a texts into tokens.
        Requires the tokenizer to be initialized.
        :param text: utf-8 encoded string that should be tokenized.
        :return: integer tokens inside the text.
        """

    @abc.abstractmethod
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Turns a tensor of tokens into text.
        :param tokens: tokens to decode.
        :return: utf-8 encoded string.
        """


class SingleCharacterTokenizer(Tokenizer):
    """
    Tokenizer that turns each character into a distinct token.
    The vocabulary has a max size.
    """

    def __init__(self, max_vocabulary_size: int):
        self._max_vocabulary_size = max_vocabulary_size

        self._stoi = None
        self._itos = None

    def fit(self, dataset: TextDataset) -> None:
        """
        Fits the tokenizer to textual data.
        If the vocabulary exceeds the max size an error is raised.
        TODO: Subsample instead of raising an error.
        """
        self._stoi = {ch: i for i, ch in enumerate(c for s in dataset for c in s)}
        vocab_size = len(self._stoi)
        if vocab_size > self._max_vocabulary_size:
            raise RuntimeError(f"Vocabulary of size {vocab_size} exceeds allowed maximum. Use a different tokenizer.")

        self._itos = {i: ch for ch, i in self._stoi.items()}

    def encode(self, text: str) -> torch.Tensor:
        if self._stoi is None:
            raise ValueError("Tokenizer is not fitted. Call .fit first.")
        return torch.tensor([self._stoi[c] for c in text])

    def decode(self, tokens: torch.Tensor) -> str:
        if self._itos is None:
            raise ValueError("Tokenizer is not fitted. Call .fit first.")
        return ''.join(self._itos[i.item()] for i in tokens)

    def vocabulary_size(self) -> int:
        if self._itos is None:
            raise ValueError("Tokenizer is not fitted. Call .fit first.")
        return len(self._stoi)
