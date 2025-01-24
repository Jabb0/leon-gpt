import pytest

from leon_gpt.data.text_dataset import MemoryTextDataset
from leon_gpt.data.tokenizer import SingleCharacterTokenizer


def test__SingleCharacterTokenizer__example_sentence() -> None:
    tokenizer = SingleCharacterTokenizer(
        20
    )

    text = "Hey let's tokenize this please."

    dataset = MemoryTextDataset(
        5,
        text
    )
    tokenizer.fit(dataset)

    encoded = tokenizer.encode(text)
    assert len(encoded) == len(text)

    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test__SingleCharacterTokenizer__empty_sentence() -> None:
    tokenizer = SingleCharacterTokenizer(
        20
    )

    text = ""

    dataset = MemoryTextDataset(
        5,
        "Hey let's tokenize this please."
    )
    tokenizer.fit(dataset)

    encoded = tokenizer.encode(text)
    assert len(encoded) == len(text)

    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test__SingleCharacterTokenizer__vocabulary_size_exceeded() -> None:
    tokenizer = SingleCharacterTokenizer(
        3
    )

    dataset = MemoryTextDataset(
        5,
        "Hey let's tokenize this please."
    )
    with pytest.raises(RuntimeError):
        tokenizer.fit(dataset)
