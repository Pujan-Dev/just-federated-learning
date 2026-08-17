"""Local (non-federated) training with a PyTorch model.

Shows that a FederatedClient can be used purely for local training: no
server, no rounds, no networking.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from federated_learning import FederatedClient


class RegressionMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 3)).astype(np.float32)
    y = (2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2]).astype(np.float32)
    y = y.reshape(-1, 1)

    client = FederatedClient(
        client_id="local",
        model=RegressionMLP(),
        train_data=(x, y),
        local_epochs=10,
        batch_size=32,
        learning_rate=0.05,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss(),
        seed=0,
    )

    before = client.get_weights()
    client.train()
    after = client.get_weights()

    changed = sum(not np.allclose(a, b) for a, b in zip(before, after))
    print(f"Trained locally for {client.local_epochs} epochs.")
    print(f"Parameters changed after training: {changed}/{len(before)}")
    print(f"Local dataset size: {client.num_samples} samples")

    update = client.get_update()
    print(f"Update: client_id={update.client_id!r}, "
          f"num_samples={update.num_samples}, weights={len(update.weights)}")


if __name__ == "__main__":
    main()