import dataclasses
from pathlib import Path

import torch


@dataclasses.dataclass
class TrainerConfig:
    dataset_path: Path = Path("data/leon2017-2020.json")
    model_path: Path = Path("leon-gpt.pth")
    batch_size: int = 64
    embedding_size: int = 768
    num_layers: int = 12
    num_heads_per_layer: int = 12
    maximum_sequence_length: int = 128
    max_iterations: int = 10000
    eval_interval: int = 300
    checkpoint_interval: int = 300
    max_vocab_size: int = 1024
    eval_iters: int = 20
    learning_rate: float = 6e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dropout: float = 0.2
