"""High-level federated training orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from federated_learning.models.base import Weights
from federated_learning.server import FederatedServer

if TYPE_CHECKING:
    from federated_learning.client import FederatedClient


class FederatedTrainer:
    """Runs the full federated learning loop in-process.

    For each round:

    1. the server's global weights are fetched,
    2. every client adopts the global weights,
    3. every client trains locally on its private data,
    4. every client produces an update,
    5. the server receives, validates and aggregates the updates (FedAvg),
    6. the aggregated weights are applied to the global model.

    No networking is required: this all happens in the current process.
    """

    def __init__(
        self,
        server: FederatedServer,
        clients: Sequence[Any],
        rounds: int = 10,
        *,
        on_round: Callable[[int, Weights], Any] | None = None,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be >= 1.")
        if not clients:
            raise ValueError("At least one client is required.")
        self.server = server
        self.clients = list(clients)
        self.rounds = rounds
        self.on_round = on_round

    def fit(self) -> "FederatedTrainer":
        """Run all federated rounds and return ``self``."""
        for round_index in range(self.rounds):
            global_weights = self.server.get_global_weights()

            updates = []
            for client in self.clients:
                if global_weights is not None:
                    client.set_weights(global_weights)
                client.train()
                updates.append(client.get_update())

            self.server.receive_updates(updates)
            new_weights = self.server.aggregate_and_update()

            if self.on_round is not None:
                self.on_round(round_index, new_weights)
        return self

    def get_model(self) -> Any:
        """Return the final global model."""
        return self.server.get_model()

    @property
    def metrics_history(self) -> list[dict]:
        """Per-round metrics recorded by the server.

        Each entry is ``{"round": int, "global": {...}, "clients": {...}}``.
        Client metrics are only present when clients are configured with
        ``metrics``; global metrics only when the server has ``metrics`` and
        ``evaluation_data`` configured.
        """
        return self.server.get_metrics_history()