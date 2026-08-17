"""Federated Learning with PyTorch models using FedAvg.

Three clients each hold private data. A server owns the global model. The
FederatedTrainer runs the whole loop in-process: distribute global weights,
train locally on each client, aggregate the updates with sample-count
weighted FedAvg, update the global model, repeat.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedTrainer,
)


class ClassificationMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def make_data(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Binary classification: x0 + x1 > 0."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    return x, y


def accuracy(model: nn.Module, x: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        preds = model(torch.as_tensor(x)).argmax(dim=1).numpy()
    return float((preds == y).mean())


def main() -> None:
    torch.manual_seed(0)

    n_clients = 3
    rounds = 5

    server = FederatedServer(model=ClassificationMLP())

    clients = [
        FederatedClient(
            client_id=f"client_{i}",
            model=ClassificationMLP(),
            train_data=make_data(40, seed=i),
            local_epochs=3,
            batch_size=16,
            learning_rate=0.2,
            criterion=nn.CrossEntropyLoss(),
            seed=0,
        )
        for i in range(n_clients)
    ]

    eval_x, eval_y = make_data(150, seed=99)
    before = accuracy(server.get_model(), eval_x, eval_y)
    print(f"n_clients={n_clients}  rounds={rounds}")
    print(f"accuracy before federated training: {before:.3f}")

    def log_round(round_index: int, weights: list[np.ndarray]) -> None:
        print(f"round {round_index + 1}: global weights updated "
              f"({len(weights)} parameter arrays)")

    trainer = FederatedTrainer(
        server=server,
        clients=clients,
        rounds=rounds,
        on_round=log_round,
    )
    trainer.fit()

    after = accuracy(trainer.get_model(), eval_x, eval_y)
    print(f"accuracy after  federated training: {after:.3f}")


if __name__ == "__main__":
    main()