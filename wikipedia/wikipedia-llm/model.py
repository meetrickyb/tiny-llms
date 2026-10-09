"""A GPT-style language model, small enough to train on a laptop CPU.

The model does one thing: given a sequence of tokens, it predicts the next
one. Everything an LLM appears to do comes from repeating that prediction.

The architecture is the same as GPT-2's, only with far fewer and far smaller
layers:

    token ids -> embeddings -> N x [attention, feed-forward] -> next-token scores
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Size of the model. The defaults give roughly 2.6 million parameters."""

    vocab_size: int = 4096
    block_size: int = 128  # context window: how many tokens the model can see at once
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Lets every token gather information from the tokens before it."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular matrix: position i may only look at positions <= i.
        # Without it the model could cheat by reading the token it must predict.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding size
        # Each token produces a query ("what am I looking for?"), a key
        # ("what do I contain?") and a value ("what do I pass on?").
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split into heads, so several attention patterns are learned in parallel.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # How well each query matches each key, turned into weights that sum to 1.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))

        out = weights @ v  # weighted average of the values
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


class Block(nn.Module):
    """One transformer layer: attention to gather context, then a feed-forward net to process it."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The additions are residual connections: each layer adjusts the
        # running representation rather than replacing it.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: embeddings, a stack of blocks, and a next-token prediction head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Attention has no built-in sense of order, so position is added explicitly.
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # The input embedding and output head share one weight matrix.
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return next-token scores for every position, and the loss if targets are given."""
        T = idx.size(1)
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            # Cross-entropy: how surprised the model was by the real next token.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def next_token(
        self, idx: torch.Tensor, temperature: float = 0.8, top_k: int | None = 50
    ) -> torch.Tensor:
        """Sample one more token for each sequence in `idx`."""
        logits, _ = self(idx[:, -self.config.block_size :])
        # Low temperature makes the likeliest tokens even likelier (safer,
        # more repetitive); high temperature flattens the odds (more varied).
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits[logits < cutoff] = float("-inf")
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
