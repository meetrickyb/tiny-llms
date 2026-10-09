"""Turn the downloaded articles into training data.

Trains the tokenizer on the articles, then encodes every article into one
long sequence of token ids. A slice of whole articles is held out as a
validation set, which the model never trains on, so we can tell learning
apart from memorising.

    python wikipedia-llm/prepare.py --vocab-size 4096
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tokenizer import Tokenizer


def load_articles(path: Path) -> list[str]:
    """Read the non-empty article texts, each prefixed with its title."""
    articles: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["text"]:
                articles.append(f"{record['title']}\n\n{record['text']}")
    return articles


def encode_articles(tokenizer: Tokenizer, articles: list[str]) -> torch.Tensor:
    """Encode articles into one id sequence, with END_OF_TEXT after each."""
    ids: list[int] = []
    for article in articles:
        ids.extend(tokenizer.encode(article))
        ids.append(tokenizer.eot_id)
    return torch.tensor(ids, dtype=torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles", type=Path, default=Path("data/articles.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    articles = load_articles(args.articles)
    random.Random(0).shuffle(articles)
    num_val = max(1, int(len(articles) * args.val_fraction))
    val_articles, train_articles = articles[:num_val], articles[num_val:]
    train_chars = sum(len(a) for a in train_articles)
    print(f"{len(train_articles)} training articles ({train_chars / 1e6:.1f} MB), {num_val} validation")

    print(f"Training tokenizer with a vocabulary of {args.vocab_size}")
    start = time.time()
    tokenizer = Tokenizer.train("\n".join(train_articles), args.vocab_size)
    tokenizer.save(args.out_dir / "tokenizer.json")
    print(f"  done in {time.time() - start:.0f}s")

    train_ids = encode_articles(tokenizer, train_articles)
    val_ids = encode_articles(tokenizer, val_articles)
    torch.save(train_ids, args.out_dir / "train.pt")
    torch.save(val_ids, args.out_dir / "val.pt")
    print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")
    print(f"{train_chars / len(train_ids):.2f} characters per token on average")

    sample = "The Hudson's Bay Company was founded in 1670."
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode(sample)]
    print(f"\nHow the tokenizer splits a sentence:\n  {sample}\n  {' | '.join(pieces)}")


if __name__ == "__main__":
    main()
