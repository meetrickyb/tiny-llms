"""Generate text with the trained model.

Give it the start of a sentence and it keeps writing, one token at a time.
Without --prompt it opens an interactive loop.

    python stage1_from_scratch/generate.py --prompt "The history of Canada"
"""

from __future__ import annotations

import argparse
import codecs
from collections.abc import Iterator
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import Tokenizer


def load_model(checkpoint_path: Path) -> tuple[GPT, Tokenizer]:
    """Rebuild the model and tokenizer saved by train.py."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer([(a, b) for a, b in checkpoint["merges"]])
    return model, tokenizer


def stream(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> Iterator[str]:
    """Yield the continuation of `prompt` piece by piece as it is generated."""
    # An empty prompt starts from END_OF_TEXT, i.e. the beginning of a new article.
    ids = tokenizer.encode(prompt) or [tokenizer.eot_id]
    idx = torch.tensor([ids], dtype=torch.long)
    # A token can end in the middle of a multi-byte character, so bytes are
    # decoded incrementally rather than token by token.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    for _ in range(max_new_tokens):
        next_id = model.next_token(idx, temperature, top_k)
        if next_id.item() == tokenizer.eot_id:
            break
        idx = torch.cat([idx, next_id], dim=1)
        yield decoder.decode(tokenizer.token_bytes(int(next_id.item())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("out/stage1.pt"))
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)

    def run(prompt: str) -> None:
        print(prompt, end="", flush=True)
        for piece in stream(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_k):
            print(piece, end="", flush=True)
        print("\n")

    if args.prompt is not None:
        run(args.prompt)
        return

    print(f"Model with {model.num_parameters():,} parameters. Type the start of a sentence; empty line to quit.")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            break
        if not prompt:
            break
        run(prompt)


if __name__ == "__main__":
    main()
