"""Federated client: holds a model and local data, and can train locally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidSampleCountError,
)
from federated_learning.models import ModelAdapter, create_adapter
from federated_learning.models.base import Weights


@dataclass
class ClientUpdate:
    """The payload a client sends back to the server.

    Attributes
    ----------
    client_id:
        Identifier of the producing client.
    weights:
        Ordered list of the client's model parameters after local training.
    num_samples:
        Number of samples used for local training. Required for weighted FedAvg.
    """

    client_id: str
    weights: Weights
    num_samples: int

    def validate(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id:
            raise InvalidClientUpdateError("client_id must be a non-empty string.")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise InvalidSampleCountError(
                "num_samples must be a positive integer, "
                f"got {self.num_samples!r}."
            )
        if not isinstance(self.weights, (list, tuple)) or not self.weights:
            raise InvalidClientUpdateError(
                "Client update must contain a non-empty list of weights."
            )
        for i, w in enumerate(self.weights):
            if w is None:
                raise InvalidClientUpdateError(
                    f"Weight at index {i} is None."
                )


class FederatedClient:
    """A single federated client.

    Holds a model plus its private local training data and can:

    * train locally (:meth:`train`),
    * expose its current parameters (:meth:`get_weights`),
    * adopt parameters sent by the server (:meth:`set_weights`),
    * produce a :class:`ClientUpdate` for the server (:meth:`get_update`).

    The client works fully in-process and does not depend on any networking
    layer.
    """

    def __init__(
        self,
        client_id: str,
        model: Any,
        train_data: Any,
        local_epochs: int = 1,
        *,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        device: str | None = None,
        optimizer: Any = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: Any = None,
        train_kwargs: dict[str, Any] | None = None,
        num_samples: int | None = None,
        seed: int | None = None,
        adapter: ModelAdapter | None = None,
    ) -> None:
        if not isinstance(client_id, str) or not client_id:
            raise ValueError("client_id must be a non-empty string.")
        if local_epochs < 1:
            raise ValueError("local_epochs must be >= 1.")

        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.criterion = criterion
        self.train_kwargs = train_kwargs or {}
        self.seed = seed
        self.adapter = adapter if adapter is not None else create_adapter(model)

        if num_samples is not None:
            if not isinstance(num_samples, int) or num_samples <= 0:
                raise ValueError("num_samples must be a positive integer.")
            self._num_samples = num_samples
        else:
            self._num_samples = self.adapter.num_samples(train_data)

    @property
    def num_samples(self) -> int:
        """Number of samples in this client's local dataset."""
        return self._num_samples

    def train(self) -> "FederatedClient":
        """Train locally on the client's own dataset (no server required)."""
        self.adapter.train(
            self.model,
            self.train_data,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            device=self.device,
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            criterion=self.criterion,
            seed=self.seed,
            **self.train_kwargs,
        )
        return self

    def get_weights(self) -> Weights:
        """Return the client's current model parameters."""
        return self.adapter.get_weights(self.model)

    def set_weights(self, weights: Weights) -> "FederatedClient":
        """Adopt model parameters (e.g. global weights from the server)."""
        self.adapter.set_weights(self.model, weights)
        return self

    def get_update(self) -> ClientUpdate:
        """Build the update to send to the server after local training."""
        return ClientUpdate(
            client_id=self.client_id,
            weights=self.get_weights(),
            num_samples=self.num_samples,
        )