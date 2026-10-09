"""Ask the trained model to add two numbers.

Without --prompt it opens an interactive loop. With --test it works through
problems that were held out of training and reports how many it got right.

    python calculate.py --model gpt-0.5-math --prompt "347 + 589"
    python calculate.py --test 10000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from data import decode_answer, encode_question, sample_problems
from model import GPT, GPTConfig


def pick_device(name: str = "auto") -> torch.device:
    """Choose where the maths runs: "mps" is the GPU built into Apple Silicon Macs."""
    if name == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    return torch.device(name)


def load_model(checkpoint_path: Path, device: torch.device | str = "cpu") -> tuple[GPT, int]:
    """Rebuild the model saved by train.py, and the number of digits it was trained on."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint["digits"]


@torch.no_grad()
def solve(model: GPT, a: torch.Tensor, b: torch.Tensor, digits: int) -> torch.Tensor:
    """Have the model add each pair of numbers, writing the answer one digit at a time."""
    idx = encode_question(a, b, digits).to(model.token_emb.weight.device)
    for _ in range(digits + 1):
        logits, _ = model(idx)
        # Always take the likeliest digit. Prose benefits from some randomness; sums do not.
        next_digit = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_digit], dim=1)
    return decode_answer(idx[:, -(digits + 1) :].cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="gpt-0.5-math", help="model name; loads out/<name>.pt")
    parser.add_argument("--prompt", default=None, help='one sum to work out, such as "347 + 589"')
    parser.add_argument("--test", type=int, default=None, metavar="N", help="check N held-out problems")
    # The model is too small for a GPU to pay off, so the CPU is the default.
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda or auto")
    args = parser.parse_args()

    checkpoint = Path("out") / f"{args.model}.pt"
    if not checkpoint.exists():
        available = ", ".join(sorted(path.stem for path in Path("out").glob("*.pt"))) or "none"
        raise SystemExit(f"No model named {args.model!r} in out/. Available: {available}")
    model, digits = load_model(checkpoint, pick_device(args.device))

    if args.test is not None:
        a, b = sample_problems(args.test, digits, from_held_out=True)
        correct = int((solve(model, a, b, digits) == a + b).sum())
        print(f"{correct:,} of {args.test:,} held-out problems correct ({correct / args.test:.2%})")
        return

    def run(text: str) -> None:
        numbers = [int(n) for n in re.findall(r"\d+", text)]
        if len(numbers) != 2 or max(numbers) >= 10**digits:
            print(f"Give two numbers of up to {digits} digits, such as 347 + 589.")
            return
        a, b = torch.tensor(numbers[:1]), torch.tensor(numbers[1:])
        print(f"The answer is {int(solve(model, a, b, digits))}")

    if args.prompt is not None:
        run(args.prompt)
        return

    print(f"{args.model} with {model.num_parameters():,} parameters. Type a sum; empty line to quit.")
    while True:
        try:
            text = input("> ")
        except EOFError:
            break
        if not text:
            break
        run(text)


if __name__ == "__main__":
    main()
