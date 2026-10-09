"""Turn the articles into question-and-answer pairs for fine-tuning.

The model trained by train.py continues text; it does not answer questions,
because it has never seen one. Fine-tuning fixes that by continuing the
training on text shaped like the behaviour we want:

    Question: What is the Hudson's Bay Company? Answer: The Hudson's Bay Company is ...

No human wrote these pairs. Wikipedia articles open by saying what their
subject is, and their sections open by summarising the section, so questions
can be built from titles and headings and answered with those opening lines.

    python wikipedia-llm/prepare_qa.py
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

import torch

from prepare import encode_articles, load_articles
from tokenizer import Tokenizer

LEAD_QUESTIONS = [
    "What is {title}?",
    "Tell me about {title}.",
    "Who or what is {title}?",
    "Describe {title}.",
    "What do you know about {title}?",
    "{title}",
]
SECTION_QUESTIONS = [
    "Tell me about the {section} of {title}.",
    "What is the {section} of {title}?",
    "{title}: {section}",
]

HEADING = re.compile(r"^== (.+?) ==$", re.MULTILINE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


def format_prompt(question: str) -> str:
    """The text the model is given; it has learned to write the answer after it."""
    return f"Question: {question} Answer:"


def first_paragraph(text: str) -> str:
    """The first line of prose, skipping blank lines and sub-headings."""
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("="):
            return line
    return ""


def shorten(paragraph: str, min_chars: int = 120, max_chars: int = 400) -> str:
    """Keep whole sentences from the start until the answer is long enough; "" if none fit."""
    answer = ""
    for sentence in SENTENCE_END.split(paragraph):
        answer = f"{answer} {sentence}".strip()
        if len(answer) >= min_chars:
            break
    if not min_chars <= len(answer) <= max_chars or answer[-1] not in ".!?":
        return ""
    return answer


def vary(question: str, rng: random.Random) -> str:
    """People type questions carelessly, so some are lower-cased or lose their punctuation."""
    if rng.random() < 0.3:
        question = question.lower()
    if rng.random() < 0.3:
        question = question.rstrip("?.")
    return question


def article_pairs(article: str, rng: random.Random) -> list[str]:
    """Build the training examples for one article."""
    title, _, text = article.partition("\n\n")
    title = re.sub(r" \(.*?\)$", "", title)  # "Canadiana (web series)" -> "Canadiana"
    lead, *sections = HEADING.split(text)

    pairs: list[tuple[str, str]] = []
    answer = shorten(first_paragraph(lead))
    if answer:
        # Every phrasing gets the same answer, so the model learns that the
        # wording of a question matters less than its subject.
        pairs.extend((template.format(title=title), answer) for template in LEAD_QUESTIONS)
    for heading, body in zip(sections[::2], sections[1::2]):
        answer = shorten(first_paragraph(body))
        if answer:
            template = rng.choice(SECTION_QUESTIONS)
            pairs.append((template.format(title=title, section=heading.lower()), answer))
    return [f"{format_prompt(vary(question, rng))} {answer}" for question, answer in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles", type=Path, default=Path("data/articles.jsonl"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/qa"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    # Same shuffle as prepare.py, so the articles held out there are held out
    # here too: the validation questions are about subjects never trained on.
    articles = load_articles(args.articles)
    random.Random(0).shuffle(articles)
    num_val = max(1, int(len(articles) * args.val_fraction))

    rng = random.Random(0)
    val_pairs = [pair for article in articles[:num_val] for pair in article_pairs(article, rng)]
    train_pairs = [pair for article in articles[num_val:] for pair in article_pairs(article, rng)]
    rng.shuffle(train_pairs)
    print(f"{len(train_pairs):,} training pairs, {len(val_pairs):,} validation pairs")

    # The tokenizer is reused unchanged: a fine-tuned model must keep reading
    # the same token ids it was first trained on.
    tokenizer = Tokenizer.load(args.tokenizer)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.tokenizer, args.out_dir / "tokenizer.json")
    train_ids = encode_articles(tokenizer, train_pairs)
    val_ids = encode_articles(tokenizer, val_pairs)
    torch.save(train_ids, args.out_dir / "train.pt")
    torch.save(val_ids, args.out_dir / "val.pt")
    print(f"{len(train_ids):,} training tokens, {len(val_ids):,} validation tokens")

    print("\nThree of the training pairs:")
    for pair in train_pairs[:3]:
        print(f"  {pair}")


if __name__ == "__main__":
    main()
