"""Shared model + synthetic data for the HTTP federated learning example.

In a real deployment the server and every client live in separate processes
(or machines); they agree on a model contract beforehand. This module plays
that shared-contract role.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ClassificationMLP(nn.Module):
    """Small binary classifier on two features."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification: positive when x0 + x1 > 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    return x, y