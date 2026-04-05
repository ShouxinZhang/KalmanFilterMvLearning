#!/usr/bin/env python3

"""Harder long-memory benchmark for plain RNN versus GRU.

The task is deliberately simple to describe but harder than the toy sentiment
demo: the model must remember a bit written near the beginning of the sequence
and report it only after a long stream of filler tokens.

Training sees only short sequences. Evaluation includes much longer sequences,
so the benchmark measures length generalization and memory retention rather
than simple memorization of tiny training examples.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 123
VOCAB_SIZE = 4
STORE_TOKEN = 0
BIT_ZERO = 1
BIT_ONE = 2
QUERY_TOKEN = 3

EMBED_DIM = 24
HIDDEN_DIM = 32
BATCH_SIZE = 128
EPOCHS = 18
LEARNING_RATE = 0.003

TRAIN_LENGTHS = (12,)
TEST_LENGTHS = (12, 16, 20, 24, 28, 32, 40, 48, 64, 96)
TRAIN_SAMPLES = 4096
TEST_SAMPLES_PER_LENGTH = 1024


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sequence(total_length: int, rng: random.Random) -> tuple[list[int], int]:
    label = rng.randint(0, 1)
    memory_token = BIT_ONE if label == 1 else BIT_ZERO
    distractors = [rng.choice((BIT_ZERO, BIT_ONE)) for _ in range(total_length - 3)]
    sequence = [STORE_TOKEN, memory_token, *distractors, QUERY_TOKEN]
    return sequence, label


def build_dataset(lengths: tuple[int, ...], num_samples: int, seed: int) -> TensorDataset:
    rng = random.Random(seed)
    if len(set(lengths)) != 1:
        raise ValueError("This benchmark expects a fixed sequence length per dataset.")
    total_length = lengths[0]

    sequences: list[list[int]] = []
    labels: list[int] = []
    for _ in range(num_samples):
        sequence, label = generate_sequence(total_length, rng)
        sequences.append(sequence)
        labels.append(label)

    tokens = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.float32)
    return TensorDataset(tokens, targets)


@dataclass
class AccuracyRow:
    name: str
    accuracy_by_length: dict[int, float]


class SequenceClassifier(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.core = core
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        _, hidden = self.core(embedded)
        final_hidden = hidden[-1]
        return self.output(final_hidden).squeeze(-1)


def make_plain_rnn() -> SequenceClassifier:
    return SequenceClassifier(nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True, nonlinearity="tanh"))


def make_gru() -> SequenceClassifier:
    return SequenceClassifier(nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True))


def train_model(
    model: nn.Module,
    dataset: TensorDataset,
    device: torch.device,
) -> list[float]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    history: list[float] = []
    for _ in range(EPOCHS):
        running_loss = 0.0
        for tokens, targets in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tokens.size(0)
        history.append(running_loss / len(dataset))
    return history


@torch.no_grad()
def evaluate_on_length(
    model: nn.Module,
    total_length: int,
    device: torch.device,
    *,
    seed: int,
) -> float:
    dataset = build_dataset((total_length,), TEST_SAMPLES_PER_LENGTH, seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    model.eval()
    correct = 0
    total = 0
    for tokens, targets in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)
        logits = model(tokens)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    return correct / total


def run_experiment(name: str, model: nn.Module, device: torch.device, seed_offset: int) -> AccuracyRow:
    train_dataset = build_dataset(TRAIN_LENGTHS, TRAIN_SAMPLES, seed=SEED + seed_offset)
    model = model.to(device)

    print(f"\nTraining {name}")
    print(f"train_lengths={TRAIN_LENGTHS} test_lengths={TEST_LENGTHS} device={device}")
    history = train_model(model, train_dataset, device)
    print(f"  epoch= 1 loss={history[0]:.4f}")
    for epoch in (6, 12, 18):
        print(f"  epoch={epoch:2d} loss={history[epoch - 1]:.4f}")

    accuracy_by_length = {
        length: evaluate_on_length(model, length, device, seed=SEED + 1000 + seed_offset + length)
        for length in TEST_LENGTHS
    }
    return AccuracyRow(name=name, accuracy_by_length=accuracy_by_length)


def print_summary(rows: list[AccuracyRow]) -> None:
    print("\nAccuracy by sequence length")
    header = "model".ljust(12) + "".join(f"{length:>10d}" for length in TEST_LENGTHS)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = row.name.ljust(12)
        for length in TEST_LENGTHS:
            line += f"{row.accuracy_by_length[length]:>10.3f}"
        print(line)

    rnn_row = next(row for row in rows if row.name == "plain_rnn")
    gru_row = next(row for row in rows if row.name == "gru")
    print("\nGRU minus plain-RNN accuracy")
    for length in TEST_LENGTHS:
        gap = gru_row.accuracy_by_length[length] - rnn_row.accuracy_by_length[length]
        print(f"  length={length:>3d}: {gap:+.3f}")

    for row in rows:
        mean_accuracy = sum(row.accuracy_by_length.values()) / len(TEST_LENGTHS)
        print(f"\nMean accuracy across test lengths for {row.name}: {mean_accuracy:.3f}")


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = [
        run_experiment("plain_rnn", make_plain_rnn(), device, seed_offset=0),
        run_experiment("gru", make_gru(), device, seed_offset=100),
    ]
    print_summary(results)


if __name__ == "__main__":
    main()
