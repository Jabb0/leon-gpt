import json
from pathlib import Path
from typing import Iterator

from leon_gpt.data.text_dataset import TextDataset


class TelegramSingleChatDataset(TextDataset):
    """
    Dataset of all telegram chat messages from a single chat export.
    The export of all chats has a different format and requires a different dataloader.
    """

    def __init__(self, sequence_len: int, file: Path) -> None:
        super().__init__(sequence_len)
        with file.open(encoding="utf-8") as f:
            data = json.load(f)

        self._text = "\n\n".join(f"{entry['from']}:\n{entry['text']}" for entry in data)

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
