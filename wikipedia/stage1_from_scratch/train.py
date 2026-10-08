"""Train the tiny GPT on the prepared Wikipedia tokens.

Training is one loop repeated thousands of times:

  1. take random snippets of text,
  2. ask the model to predict the next token at every position,
  3. measure how wrong it was (the loss),
  4. nudge every parameter slightly in the direction that reduces the loss.

    python stage1_from_scratch/train.py --max-iters 5000
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from generate import stream
from model import GPT, GPTConfig
from tokenizer import Tokenizer


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick random snippets; the target is the same snippet shifted one token ahead."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).long()
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts]).long()
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batch_size: int, batches: int) -> float:
    """Average the loss over several batches for a steadier reading."""
    model.eval()
    losses = [
        model(*get_batch(data, batch_size, model.config.block_size))[1].item() for _ in range(batches)
    ]
    model.train()
    return sum(losses) / len(losses)


def learning_rate(step: int, max_iters: int, peak: float, warmup: int) -> float:
    """Ramp up linearly, then decay along a cosine curve to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_iters - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out/stage1.pt"))
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=GPTConfig.block_size)
    parser.add_argument("--n-layer", type=int, default=GPTConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=GPTConfig.n_head)
    parser.add_argument("--n-embd", type=int, default=GPTConfig.n_embd)
    parser.add_argument("--dropout", type=float, default=GPTConfig.dropout)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.load(args.data_dir / "tokenizer.json")
    train_data = torch.load(args.data_dir / "train.pt")
    val_data = torch.load(args.data_dir / "val.pt")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    print(f"Model: {model.num_parameters():,} parameters, {len(train_data):,} training tokens")
    # Before any training the model guesses uniformly, so the loss starts near this.
    print(f"Loss of a random guess: {math.log(config.vocab_size):.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    start = time.time()
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            train_loss = estimate_loss(model, train_data, args.batch_size, args.eval_batches)
            val_loss = estimate_loss(model, val_data, args.batch_size, args.eval_batches)
            print(
                f"\nstep {step}: train loss {train_loss:.3f}, val loss {val_loss:.3f}, "
                f"{(time.time() - start) / 60:.1f} min elapsed"
            )
            model.eval()
            sample = "".join(stream(model, tokenizer, "The history of Canada", max_new_tokens=60))
            model.train()
            print(f"  sample: The history of Canada{sample}".replace("\n", " "))
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": asdict(config), "merges": tokenizer.merges},
                    args.out,
                )
        if step == args.max_iters:
            break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.max_iters, args.lr, args.warmup)
        x, y = get_batch(train_data, args.batch_size, config.block_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # work out how each parameter contributed to the error
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # nudge the parameters

        if step % 50 == 0:
            print(f"\r  step {step}/{args.max_iters}, loss {loss.item():.3f}", end="", flush=True)

    print(f"\nBest validation loss {best_val:.3f}; checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
