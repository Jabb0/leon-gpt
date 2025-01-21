import dataclasses
from pathlib import Path

import torch
from torch.nn import functional as F

from leon_gpt.modules.bigram import BigramLanguageModel



@dataclasses.dataclass
class TrainerConfig:
    dataset_path: Path = Path("data/input.txt")
    batch_size: int = 32
    embedding_size: int = 512
    num_layers: int = 2
    num_heads_per_layer: int = 4
    maximum_sequence_length: int = 8
    max_iterations: int = 1000
    eval_interval: int = 300
    eval_iters: int = 200
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dropout: float = 0.


def compute_loss(logits, target):
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    target = target.view(B * T)
    loss = F.cross_entropy(logits, target)
    return loss

def generate(model: BigramLanguageModel, start_idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """
    Generates a sequence of up to max_new_tokens in addition to the start_idx sequence.
    :param model:
    :param start_idx:
    :param max_new_tokens:
    :return:
    """
    idx = start_idx.clone()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.max_sequence_length:])
        logits = logits[:, -1, :]  # Only predictions for next token after the last.
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    return idx


def train(config: TrainerConfig) -> None:
    # Dataloader
    # Tokenizer
    # Super yanky

    with config.dataset_path.open(encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = { ch:i for i, ch in enumerate(chars) }
    itos = { i:ch for ch, i in stoi.items() }
    encode = lambda s: [stoi[c] for c in s]  # encode string to integer list
    decode = lambda l: ''.join(itos[i] for i in l)  # decode a list of integers to a string.

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - config.maximum_sequence_length, (config.batch_size,))
        x = torch.stack([data[i:i+config.maximum_sequence_length] for i in ix])
        y = torch.stack([data[i+1:i+config.maximum_sequence_length + 1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)
        return x, y

    # Transformer
    model = BigramLanguageModel(
        vocab_size=vocab_size,
        embedding_size=config.embedding_size,
        num_layers=config.num_layers,
        num_heads_per_layer=config.num_heads_per_layer,
        max_sequence_length=config.maximum_sequence_length,
        dropout=config.dropout,
    )
    model = model.to(config.device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(config.eval_iters)
            for k in range(config.eval_iters):
                x, y = get_batch(split)
                eval_logits = model(x)
                estimated_loss = compute_loss(eval_logits, y)
                losses[k] = estimated_loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for n_iter in range(config.max_iterations):
        if n_iter % config.eval_interval == 0:
            all_losses = estimate_loss()
            print(f"Step {n_iter}: train loss {all_losses['train']:.4f}, val loss {all_losses['val']:.4f}")

        xb, yb = get_batch("train")
        logits = model(xb)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(logits, yb)
        loss.backward()

        optimizer.step()

    start_idx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    result_string = decode(generate(model, start_idx, max_new_tokens=100)[0].tolist())
    print(result_string)



if __name__ == '__main__':
    train(TrainerConfig())
