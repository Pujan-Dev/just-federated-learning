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
from federated_learning.metrics import evaluate_model, resolve_metrics
from federated_learning.models import ModelAdapter, create_adapter
from federated_learning.models.base import Weights


class FederatedServer:
    """Coordinates federated training.

    Holds the global model and its weights, receives client updates, validates
    them, runs FedAvg and applies the result back to the global model.

    The server is transport-agnostic: it works in-process and the HTTP layer
    is an optional add-on.

    Metrics: when ``metrics`` and ``evaluation_data`` are configured, every
    aggregated round records a ``metrics_history`` entry with the global
    model's metrics plus the metrics each client reported in its update.
    """

    def __init__(
        self,
        model: Any,
        *,
        adapter: ModelAdapter | None = None,
        aggregation: FedAvg | None = None,
        metrics: Any = None,
        evaluation_data: Any = None,
    ) -> None:
        self.model = model
        self.adapter = adapter if adapter is not None else create_adapter(model)
        self.aggregation = aggregation if aggregation is not None else FedAvg()
        self._updates: list[ClientUpdate] = []
        self.round = 0
        self._weights: Weights | None = None
        self.metrics = resolve_metrics(metrics)
        self.evaluation_data = evaluation_data
        self.metrics_history: list[dict] = []
        if model is not None:
            try:
                self._weights = self.adapter.get_weights(model)
            except InvalidWeightsError:
                # e.g. an unfitted scikit-learn estimator: global weights will be
                # established once the first round of updates is aggregated.
                self._weights = None

    def evaluate(
        self,
        data: Any = None,
        metrics: Any = None,
    ) -> dict[str, float]:
        """Evaluate the global model and return ``{name: value}`` metrics.

        ``data`` defaults to the server's ``evaluation_data``; ``metrics``
        defaults to the server's configured metrics.
        """
        resolved = resolve_metrics(metrics if metrics is not None else self.metrics)
        if not resolved:
            return {}
        eval_data = data if data is not None else self.evaluation_data
        if eval_data is None:
            raise ValueError(
                "Cannot evaluate the global model: no evaluation_data is "
                "configured on the server."
            )
        return evaluate_model(self.adapter, self.model, eval_data, resolved)

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
        """Aggregate stored updates, apply them, and record round metrics."""
        weights = self.aggregate()
        self.update_global_model(weights)

        global_metrics: dict[str, float] = {}
        if self.metrics and self.evaluation_data is not None:
            global_metrics = evaluate_model(
                self.adapter, self.model, self.evaluation_data, self.metrics
            )
        client_metrics: dict[str, dict[str, float]] = {
            u.client_id: dict(u.metrics) for u in self._updates if u.metrics
        }
        self.metrics_history.append(
            {
                "round": self.round,
                "global": global_metrics,
                "clients": client_metrics,
            }
        )
        self._updates = []
        return weights

    def get_metrics_history(self) -> list[dict]:
        """Per-round metrics: ``{round, global, clients}`` entries."""
        return list(self.metrics_history)

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