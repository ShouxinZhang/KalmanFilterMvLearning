#!/usr/bin/env python3

"""Shared utilities for the easy toy sentiment demos.

The task is deliberately tiny: learn the sentiment of short movie-review
snippets and highlight how recurrent memory handles negation such as ``not``.
Because the dataset is so small and the sentences are so short, this code is
best read as a mechanism demo. It is not meant to cleanly separate plain RNN
and GRU in practical performance.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import torch
from torch import nn

TrainingSample = tuple[str, float]

TRAINING_SAMPLES: list[TrainingSample] = [
    ("movie is good", 1.0),
    ("movie is very good", 1.0),
    ("movie is really good", 1.0),
    ("movie is not bad", 1.0),
    ("movie is not very bad", 1.0),
    ("movie feels good", 1.0),
    ("movie is bad", 0.0),
    ("movie is very bad", 0.0),
    ("movie is really bad", 0.0),
    ("movie is not good", 0.0),
    ("movie is not very good", 0.0),
    ("movie feels bad", 0.0),
]

DEMO_SENTENCES: list[str] = [
    "movie is good",
    "movie is bad",
    "movie is not good",
    "movie is not bad",
    "movie is very good",
    "movie is not very good",
]


def set_seed(seed: int = 0) -> None:
    """Set random seeds for reproducible toy-demo runs."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vocab(samples: Iterable[TrainingSample]) -> dict[str, int]:
    """Build a tiny vocabulary from the training samples."""

    vocab = {"<unk>": 0}
    for sentence, _ in samples:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_sentence(sentence: str, vocab: dict[str, int], device: torch.device) -> torch.Tensor:
    """Convert one sentence into a `(1, seq_len)` token tensor."""

    token_ids = [vocab.get(token, vocab["<unk>"]) for token in sentence.split()]
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def train_classifier(
    model: nn.Module,
    samples: list[TrainingSample],
    vocab: dict[str, int],
    *,
    epochs: int = 250,
    learning_rate: float = 0.03,
    device: torch.device,
) -> list[float]:
    """Train a sequence classifier on the tiny sentiment dataset."""

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    model.train()
    for _ in range(epochs):
        total_loss = 0.0
        for sentence, label in samples:
            tokens = encode_sentence(sentence, vocab, device)
            target = torch.tensor([label], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        history.append(total_loss / len(samples))
    return history


@torch.no_grad()
def predict_probability(
    model: nn.Module, sentence: str, vocab: dict[str, int], device: torch.device
) -> float:
    """Predict the positive-class probability for one sentence."""

    model.eval()
    tokens = encode_sentence(sentence, vocab, device)
    logits = model(tokens)
    probability = torch.sigmoid(logits).item()
    return probability


def print_demo_predictions(
    model: nn.Module, sentences: list[str], vocab: dict[str, int], device: torch.device
) -> None:
    """Print demo predictions for a small list of sentences."""

    print("\nPredictions:")
    for sentence in sentences:
        positive_prob = predict_probability(model, sentence, vocab, device)
        label = "positive" if positive_prob >= 0.5 else "negative"
        print(f"  {sentence:<24} -> {positive_prob:0.3f} ({label})")
