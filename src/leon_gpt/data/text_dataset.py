from abc import ABC
from typing import Iterator

from torch.utils.data import IterableDataset

class TextDataset(IterableDataset[str], ABC):
    """
    Dataset that returns utf-8 encoded strings of a given size.

    """

    def __init__(self, sequence_len: int) -> None:
        super().__init__()
        self._sequence_len = sequence_len


class MemoryTextDataset(TextDataset):
    """
    Dataset that has the text known at construction time.
    """
    def __init__(self, sequence_len: int, text: str) -> None:
        super().__init__(sequence_len)

        self._text = text
        self._length = len(self._text) - self._sequence_len + 1
        if self._length <= 0:
            raise ValueError(f"Sequence length {sequence_len} is too large for text with length {len(self._text)}")

    def __len__(self) -> int:
        """
        :return: Number of sequences in this dataset.
        """
        return self._length

    def __getitem__(self, idx: int) -> str:
        """Returns an utf-8 encoded string of text with sequence_len length starting from character idx."""
        return self._text[idx:idx+self._sequence_len]

    def __iter__(self) -> Iterator[str]:
        for i in range(self._length):
            yield self[i]
