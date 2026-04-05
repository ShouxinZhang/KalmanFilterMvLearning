#!/usr/bin/env python3

"""Minimal GRU sentiment demo for the note.

This task is intentionally easy, so plain RNN and GRU can look similar here.
The goal is to illustrate the shared recurrent mechanism, not to claim that
this tiny example is a decisive benchmark.
"""

from __future__ import annotations

import torch
from torch import nn

from toy_sentiment import (
    DEMO_SENTENCES,
    TRAINING_SAMPLES,
    build_vocab,
    predict_probability,
    print_demo_predictions,
    set_seed,
    train_classifier,
)

# Hyperparameters for the intentionally easy toy demo.
EMBED_DIM = 16
HIDDEN_DIM = 16


class GRUClassifier(nn.Module):
    """A minimal GRU classifier for short sentiment sequences."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        _, hidden = self.gru(embedded)
        logits = self.output(hidden[-1]).squeeze(-1)
        return logits


def main() -> None:
    set_seed(11)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(TRAINING_SAMPLES)
    model = GRUClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    print("Training GRU on the easy toy sentiment dataset")
    print("This demo is intentionally simple, so plain RNN and GRU may look similar.")
    print(f"device={device} vocab_size={len(vocab)} samples={len(TRAINING_SAMPLES)}")

    history = train_classifier(model, TRAINING_SAMPLES, vocab, device=device)
    print(f"epoch=  1 loss={history[0]:.4f}")
    for epoch in (50, 100, 150, 200, 250):
        print(f"epoch={epoch:3d} loss={history[epoch - 1]:.4f}")

    print_demo_predictions(model, DEMO_SENTENCES, vocab, device)
    probability = predict_probability(model, "movie is not good", vocab, device)
    print(f"\nNegation check: p(positive | 'movie is not good') = {probability:.3f}")


if __name__ == "__main__":
    main()
