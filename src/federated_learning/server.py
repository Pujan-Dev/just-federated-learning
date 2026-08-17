"""Federated server: stores the global model and aggregates client updates."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from federated_learning.aggregation import FedAvg
from federated_learning.client import ClientUpdate
from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidWeightsError,
)
from federated_learning.models import ModelAdapter, create_adapter
from federated_learning.models.base import Weights


class FederatedServer:
    """Coordinates federated training.

    Holds the global model and its weights, receives client updates, validates
    them, runs FedAvg and applies the result back to the global model.

    The server is transport-agnostic: it works in-process and the HTTP layer
    is an optional add-on.
    """

    def __init__(
        self,
        model: Any,
        *,
        adapter: ModelAdapter | None = None,
        aggregation: FedAvg | None = None,
    ) -> None:
        self.model = model
        self.adapter = adapter if adapter is not None else create_adapter(model)
        self.aggregation = aggregation if aggregation is not None else FedAvg()
        self._updates: list[ClientUpdate] = []
        self.round = 0
        self._weights: Weights | None = None
        if model is not None:
            try:
                self._weights = self.adapter.get_weights(model)
            except InvalidWeightsError:
                # e.g. an unfitted scikit-learn estimator: global weights will be
                # established once the first round of updates is aggregated.
                self._weights = None

    @property
    def has_global_weights(self) -> bool:
        return self._weights is not None

    def get_global_weights(self) -> Weights | None:
        """Return a copy of the current global weights.

        Returns ``None`` if no global weights are available yet (e.g. the
        global scikit-learn estimator has not been fitted and no round has
        been aggregated). In that case the first round simply establishes the
        global weights from the client updates.
        """
        if self._weights is None:
            try:
                self._weights = self.adapter.get_weights(self.model)
            except InvalidWeightsError:
                return None
        return [np.array(w) for w in self._weights]

    def _validate_update(self, update: ClientUpdate) -> None:
        if not isinstance(update, ClientUpdate):
            raise InvalidClientUpdateError(
                "Updates must be ClientUpdate objects, got "
                f"{type(update).__name__!r}."
            )
        update.validate()
        if self._weights is not None:
            self.adapter.validate_weights(self.model, update.weights)

    def receive_update(self, update: ClientUpdate) -> None:
        """Validate and store a single client update."""
        self._validate_update(update)
        self._updates.append(update)

    def receive_updates(self, updates: Iterable[ClientUpdate]) -> None:
        """Validate and store a batch of client updates."""
        for update in updates:
            self.receive_update(update)

    def aggregate(self, updates: Sequence[ClientUpdate] | None = None) -> Weights:
        """Aggregate the given updates (or the stored ones) via FedAvg."""
        selected = list(updates) if updates is not None else list(self._updates)
        if not selected:
            raise InvalidClientUpdateError(
                "No client updates available to aggregate. Call "
                "receive_update() or pass updates explicitly."
            )
        return self.aggregation.aggregate(selected)

    def update_global_model(self, weights: Weights) -> Weights:
        """Apply aggregated weights to the global model."""
        self.adapter.set_weights(self.model, weights)
        self._weights = [np.array(w) for w in weights]
        self.round += 1
        return self.get_global_weights() or self._weights

    def aggregate_and_update(self) -> Weights:
        """Aggregate stored updates and apply them to the global model."""
        weights = self.aggregate()
        return self.update_global_model(weights)

    def reset(self) -> None:
        """Clear stored updates (round counter is preserved)."""
        self._updates = []

    def get_model(self) -> Any:
        """Return the global model."""
        return self.model

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FederatedServer(model={type(self.model).__name__}, "
            f"round={self.round}, pending_updates={len(self._updates)})"
        )