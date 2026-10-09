"""Addition problems written as sequences of tokens.

The model reads and writes one character at a time, from a vocabulary of
twelve: the ten digits, "+" and "=". A problem and its answer look like this
for three-digit numbers:

    347+589=6390

Two things about that text make the job learnable for a tiny model.

Numbers are padded with zeros to a fixed width, so the hundreds digit is
always in the same place.

The answer is written backwards (936 is padded to 0936, then reversed to
6390). People add from the right, carrying leftwards, because each digit of
the answer depends only on the digits to its right. A model that writes one
token at a time can follow the same procedure only if it writes the units
digit first.
"""

from __future__ import annotations

import torch

VOCAB = "0123456789+="
PLUS = VOCAB.index("+")
EQUALS = VOCAB.index("=")


def held_out(a: torch.Tensor, b: torch.Tensor, digits: int) -> torch.Tensor:
    """True for the tenth of all problems that are never trained on.

    A fixed scramble of the two numbers decides, so the same problems are held
    out on every run and no pattern links them. Getting these right is the
    evidence that the model learned to add rather than memorised answers.
    """
    scrambled = (a * 10**digits + b) * 2654435761 % 2**32
    return (scrambled >> 16) % 10 == 0


def sample_problems(count: int, digits: int, from_held_out: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick random pairs of numbers, from the training problems or the held-out ones."""
    a = b = torch.empty(0, dtype=torch.long)
    while len(a) < count:
        new_a = torch.randint(10**digits, (2 * count,))
        new_b = torch.randint(10**digits, (2 * count,))
        keep = held_out(new_a, new_b, digits) == from_held_out
        a, b = torch.cat([a, new_a[keep]]), torch.cat([b, new_b[keep]])
    return a[:count], b[:count]


def to_digits(numbers: torch.Tensor, width: int) -> torch.Tensor:
    """Split each number into `width` digit tokens, most significant first."""
    powers = 10 ** torch.arange(width - 1, -1, -1)
    return (numbers[:, None] // powers) % 10


def encode_question(a: torch.Tensor, b: torch.Tensor, digits: int) -> torch.Tensor:
    """The tokens for "347+589=", one row per problem."""
    plus = torch.full((len(a), 1), PLUS)
    equals = torch.full((len(a), 1), EQUALS)
    return torch.cat([to_digits(a, digits), plus, to_digits(b, digits), equals], dim=1)


def encode(a: torch.Tensor, b: torch.Tensor, digits: int) -> torch.Tensor:
    """The tokens for "347+589=6390": the question, then the answer backwards."""
    # The sum of two n-digit numbers can need n + 1 digits.
    answer = to_digits(a + b, digits + 1).flip(1)
    return torch.cat([encode_question(a, b, digits), answer], dim=1)


def decode_answer(tokens: torch.Tensor) -> torch.Tensor:
    """Turn rows of backwards answer digits into numbers."""
    powers = 10 ** torch.arange(tokens.size(1))
    return (tokens * powers).sum(dim=1)
