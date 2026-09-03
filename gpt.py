"""Train and sample from a small, character-level GPT.

The transformer architecture follows the model built in Andrej Karpathy's
"Let's build GPT" lecture, with a command-line interface and checkpoints added
so training and generation can be run independently.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")


class CharTokenizer:
    """A reversible character-level tokenizer learned from a text corpus."""

    def __init__(self, chars: list[str]) -> None:
        if not chars:
            raise ValueError("Tokenizer vocabulary cannot be empty")
        self.chars = sorted(set(chars))
        self.stoi = {char: index for index, char in enumerate(self.chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(list(text))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[char] for char in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc.args[0]!r} is not in the tokenizer vocabulary"
            ) from exc

    def decode(self, token_ids: list[int]) -> str:
        try:
            return "".join(self.itos[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(f"Unknown token id: {exc.args[0]}") from exc


class AttentionHead(nn.Module):
    """One causal self-attention head."""

    def __init__(self, config: GPTConfig, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time_steps, _ = x.shape
        keys = self.key(x)
        queries = self.query(x)

        attention = queries @ keys.transpose(-2, -1)
        attention = attention * (keys.shape[-1] ** -0.5)
        attention = attention.masked_fill(
            ~self.causal_mask[:time_steps, :time_steps], float("-inf")
        )
        attention = self.dropout(F.softmax(attention, dim=-1))
        return attention @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Several causal attention heads evaluated in parallel."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList(
            [AttentionHead(config, head_size) for _ in range(config.n_head)]
        )
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(attended))


class FeedForward(nn.Module):
    """Position-wise multilayer perceptron used inside a transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm_1 = nn.LayerNorm(config.n_embd)
        self.layer_norm_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.final_layer_norm = nn.LayerNorm(config.n_embd)
        self.language_model_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time_steps = idx.shape
        if time_steps > self.config.block_size:
            raise ValueError(
                f"Sequence length {time_steps} exceeds block size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(time_steps, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.final_layer_norm(self.blocks(x))
        logits = self.language_model_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        for _ in range(max_new_tokens):
            idx_context = idx[:, -self.config.block_size :]
            logits, _ = self(idx_context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                cutoff = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def choose_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_data(text: str, tokenizer: CharTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(0.9 * len(data))
    return data[:split_index], data[split_index:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) <= block_size:
        raise ValueError("Dataset split must be longer than block_size")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[i : i + block_size] for i in starts])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    estimates: dict[str, float] = {}
    for name, data in (("train", train_data), ("validation", validation_data)):
        losses = torch.zeros(eval_iters)
        for iteration in range(eval_iters):
            inputs, targets = get_batch(
                data, batch_size, model.config.block_size, device
            )
            _, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Model did not return a training loss")
            losses[iteration] = loss.item()
        estimates[name] = losses.mean().item()
    model.train()
    return estimates


def save_checkpoint(
    path: Path,
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "chars": tokenizer.chars,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def load_checkpoint(
    path: Path, device: str
) -> tuple[GPTLanguageModel, CharTokenizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["chars"])
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer, checkpoint


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    text = args.input.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    train_data, validation_data = split_data(text, tokenizer)

    start_step = 0
    if args.resume:
        model, checkpoint_tokenizer, checkpoint = load_checkpoint(
            args.checkpoint, device
        )
        if checkpoint_tokenizer.chars != tokenizer.chars:
            raise ValueError("Checkpoint vocabulary does not match the input corpus")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint["step"])
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        )
        model = GPTLanguageModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} | vocabulary={tokenizer.vocab_size} | "
        f"parameters={parameter_count / 1e6:.2f}M"
    )

    model.train()
    final_step = start_step + args.max_iters
    for step in range(start_step, final_step):
        if step % args.eval_interval == 0 or step == final_step - 1:
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                args.batch_size,
                args.eval_iters,
                device,
            )
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"validation loss {losses['validation']:.4f}"
            )

        inputs, targets = get_batch(
            train_data, args.batch_size, model.config.block_size, device
        )
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_checkpoint(args.checkpoint, model, tokenizer, optimizer, final_step)
    print(f"Saved checkpoint to {args.checkpoint}")
    print(generate_text(model, tokenizer, args.prompt, args.max_new_tokens, device))


def generate_text(
    model: GPTLanguageModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    if not prompt:
        prompt = tokenizer.chars[0]
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(context, max_new_tokens, temperature, top_k)
    return tokenizer.decode(generated[0].tolist())


def generate_from_checkpoint(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            device,
            args.temperature,
            args.top_k,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "generate"), default="train")
    parser.add_argument("--input", type=Path, default=project_dir / "input.txt")
    parser.add_argument(
        "--checkpoint", type=Path, default=project_dir / "checkpoints" / "gpt.pt"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        generate_from_checkpoint(args)


if __name__ == "__main__":
    main()
