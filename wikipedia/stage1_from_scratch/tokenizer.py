"""A byte-pair-encoding (BPE) tokenizer written from scratch.

A language model does not read letters or words; it reads integers. The
tokenizer decides what those integers stand for.

BPE starts with 256 tokens, one per possible byte, so any text can be
encoded. It then repeatedly finds the pair of tokens that most often sit next
to each other in the training text and glues them into one new token. After a
few thousand merges, common words are a single token and rare words are a
handful of pieces. This is the same scheme GPT models use, at a smaller size.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Text is first cut into word-sized chunks (a word keeps its leading space),
# and merges never cross a chunk boundary. The three alternatives cover every
# character, so joining the chunks gives back the original text exactly.
CHUNK_PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

END_OF_TEXT = "<|endoftext|>"

Pair = tuple[int, int]


class Tokenizer:
    """Encodes text to token ids and back using a learned list of merges."""

    def __init__(self, merges: list[Pair]) -> None:
        self.merges = merges
        # The rank of a merge is the order it was learned in; earlier merges
        # must be applied first when encoding.
        self.ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.eot_id = len(self.vocab)
        self.vocab_size = len(self.vocab) + 1
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> Tokenizer:
        """Learn `vocab_size - 257` merges from `text`."""
        # Work on unique chunks with their counts rather than the raw text:
        # "the" appears a million times but only needs processing once.
        chunk_counts = Counter(CHUNK_PATTERN.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        pair_counts: Counter[Pair] = Counter()
        pair_words: defaultdict[Pair, set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            for pair in zip(word, word[1:]):
                pair_counts[pair] += counts[i]
                pair_words[pair].add(i)

        merges: list[Pair] = []
        num_merges = vocab_size - 256 - 1  # one id is kept for END_OF_TEXT
        for step in range(num_merges):
            best = max(pair_counts, key=lambda p: (pair_counts[p], p), default=None)
            # Stop early once every chunk has collapsed into a single token.
            if best is None or pair_counts[best] <= 0:
                break
            new_id = 256 + len(merges)
            merges.append(best)

            # Only the words containing the winning pair change, so only
            # their pair counts are recomputed.
            for i in pair_words.pop(best):
                word, count = words[i], counts[i]
                for pair in zip(word, word[1:]):
                    pair_counts[pair] -= count
                word = _merge(word, best, new_id)
                words[i] = word
                for pair in zip(word, word[1:]):
                    pair_counts[pair] += count
                    pair_words[pair].add(i)
            del pair_counts[best]

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{num_merges}")
        return cls(merges)

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids."""
        ids: list[int] = []
        for chunk in CHUNK_PATTERN.findall(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            ids.extend(self._cache[chunk])
        return ids

    def _encode_chunk(self, chunk: str) -> list[int]:
        word = list(chunk.encode("utf-8"))
        while len(word) > 1:
            # Apply the earliest-learned merge that is present, until none are.
            pair = min(zip(word, word[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            word = _merge(word, pair, 256 + self.ranks[pair])
        return word

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes one token stands for (empty for END_OF_TEXT)."""
        return b"" if token_id == self.eot_id else self.vocab[token_id]

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        data = b"".join(self.token_bytes(i) for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        merges = json.loads(path.read_text(encoding="utf-8"))["merges"]
        return cls([(a, b) for a, b in merges])


def _merge(word: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `word` with `new_id`."""
    out: list[int] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
