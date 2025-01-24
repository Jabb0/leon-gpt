import dataclasses
from pathlib import Path

import torch
from torch.nn import functional as F#

import json

from leon_gpt.modules.bigram import BigramLanguageModel






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
        data = json.load(f)

    text = "\n\n".join(f"{entry['from']}:\n{entry['text']}" for entry in data)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    if vocab_size > config.max_vocab_size:
        raise RuntimeError(f"Vocabulary of size {vocab_size} exceeds allowed maximum. Use a different tokenizer.")

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
    if config.model_path.exists():
        model.load_state_dict(torch.load(config.model_path, weights_only=True))

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

        if n_iter > 0 and n_iter % config.checkpoint_interval == 0:
            torch.save(model.state_dict(), config.model_path)

        xb, yb = get_batch("train")
        logits = model(xb)
        optimizer.zero_grad(set_to_none=True)
        loss = compute_loss(logits, yb)
        loss.backward()

        optimizer.step()

    torch.save(model.state_dict(), config.model_path)

    model.eval()
    start_string = "Leon Kaltenbrunn:\n"

    start_idx = torch.tensor([encode(start_string)], dtype=torch.long, device=config.device)
    result_string = decode(generate(model, start_idx, max_new_tokens=1024)[0].tolist())
    print(result_string)



def infer(config):
    with config.dataset_path.open(encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    if vocab_size > config.max_vocab_size:
        raise RuntimeError(f"Vocabulary of size {vocab_size} exceeds allowed maximum. Use a different tokenizer.")

    stoi = { ch:i for i, ch in enumerate(chars) }
    itos = { i:ch for ch, i in stoi.items() }
    encode = lambda s: [stoi[c] for c in s]  # encode string to integer list
    decode = lambda l: ''.join(itos[i] for i in l)  # decode a list of integers to a string.

    model = BigramLanguageModel(
        vocab_size=vocab_size,
        embedding_size=config.embedding_size,
        num_layers=config.num_layers,
        num_heads_per_layer=config.num_heads_per_layer,
        max_sequence_length=config.maximum_sequence_length,
        dropout=config.dropout,
    )
    model = model.to(config.device)
    model.load_state_dict(torch.load(config.model_path, weights_only=True))
    model.eval()

    prompt = ""
    while prompt != "quit":
        prompt = input("Leon:")

        start_string = f"Leon Kaltenbrunn:\n{prompt}\n\n"

        start_idx = torch.tensor([encode(start_string)], dtype=torch.long, device=config.device)
        result_string = decode(generate(model, start_idx, max_new_tokens=1024)[0].tolist())
        print(result_string)



if __name__ == '__main__':
    infer(TrainerConfig())
