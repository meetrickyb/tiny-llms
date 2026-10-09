"""Train the tiny GPT to add numbers.

The loop is the same as for the Wikipedia model: show examples, measure how
wrong the predictions were, nudge the parameters. Two things differ.

The training data is not downloaded. Addition problems are made up on the
spot, so the supply is endless and no example needs to be shown twice.

Success can be measured exactly. A tenth of all possible problems are never
trained on, and the model is scored on how many of those it gets right. A
model that memorised its examples would score nothing there.

    python train.py --digits 3
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from calculate import pick_device, solve
from data import VOCAB, encode, sample_problems
from model import GPT, GPTConfig


def get_batch(batch_size: int, digits: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Make up fresh problems; the target is the same text shifted one token ahead."""
    tokens = encode(*sample_problems(batch_size, digits), digits)
    x, y = tokens[:, :-1], tokens[:, 1:].clone()
    # The numbers being added are random, so nothing can predict them. Only
    # the answer digits count towards the loss; -100 marks the rest as ignored.
    y[:, : 2 * digits + 1] = -100
    return x.to(device), y.to(device)


def accuracy(model: GPT, digits: int, count: int, from_held_out: bool) -> float:
    """The share of problems answered exactly right, every digit included."""
    a, b = sample_problems(count, digits, from_held_out)
    return float((solve(model, a, b, digits) == a + b).float().mean())


def learning_rate(step: int, max_iters: int, peak: float, warmup: int) -> float:
    """Ramp up linearly, then decay along a cosine curve to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_iters - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--digits", type=int, default=3, help="largest size of number to add")
    parser.add_argument("--out", type=Path, default=Path("out/gpt-0.5.pt"))
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-problems", type=int, default=2000)
    parser.add_argument("--n-layer", type=int, default=GPTConfig.n_layer)
    parser.add_argument("--n-head", type=int, default=GPTConfig.n_head)
    parser.add_argument("--n-embd", type=int, default=GPTConfig.n_embd)
    parser.add_argument("--seed", type=int, default=0)
    # The model is too small for a GPU to pay off, so the CPU is the default.
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda or auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    config = GPTConfig(
        vocab_size=len(VOCAB),
        block_size=3 * args.digits + 2,  # the whole problem and answer, less the final token
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    problems = 10 ** (2 * args.digits)
    print(f"Model: {model.num_parameters():,} parameters, training on: {device}")
    print(f"{problems:,} possible problems, of which a tenth are held out and never trained on")
    print(f"Training will show it at most {args.max_iters * args.batch_size:,} of them")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    start = time.time()
    for step in range(args.max_iters + 1):
        if step % args.eval_interval == 0:
            model.eval()
            seen = accuracy(model, args.digits, args.eval_problems, from_held_out=False)
            unseen = accuracy(model, args.digits, args.eval_problems, from_held_out=True)
            a, b = sample_problems(3, args.digits, from_held_out=True)
            answers = solve(model, a, b, args.digits)
            model.train()
            print(
                f"\nstep {step}: {seen:.1%} right on training problems, {unseen:.1%} on held-out ones, "
                f"{time.time() - start:.0f}s elapsed"
            )
            for x, y, answer in zip(a.tolist(), b.tolist(), answers.tolist()):
                print(f"  {x} + {y} = {answer}  {'✓' if answer == x + y else f'✗ ({x + y})'}")
            if unseen > best:
                best = unseen
                torch.save({"model": model.state_dict(), "config": asdict(config), "digits": args.digits}, args.out)
        if step == args.max_iters:
            break

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.max_iters, args.lr, args.warmup)
        x, y = get_batch(args.batch_size, args.digits, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # work out how each parameter contributed to the error
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # nudge the parameters

        if step % 50 == 0:
            print(f"\r  step {step}/{args.max_iters}, loss {loss.item():.3f}", end="", flush=True)

    print(f"\nBest held-out accuracy {best:.1%}; checkpoint saved to {args.out}")


if __name__ == "__main__":
    main()
