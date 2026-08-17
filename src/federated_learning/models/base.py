"""Abstract model adapter interface.

The federated learning core only ever talks to models through the
:class:`ModelAdapter` interface. This keeps the core framework-agnostic:
it does not care whether the underlying model is a PyTorch ``nn.Module`` or
a scikit-learn estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from federated_learning.exceptions import (
    InvalidWeightsError,
    WeightShapeMismatchError,
)

Weights = list[np.ndarray]


class ModelAdapter(ABC):
    """Common interface for framework-specific model adapters.

    The exact internal representation of *weights* differs per framework,
    but every adapter must be able to:

    * extract the model parameters as an ordered list of arrays
      (:meth:`get_weights`),
    * apply a list of arrays back onto the model (:meth:`set_weights`),
    * run local training on a model (:meth:`train`),
    * validate that weights are compatible with a model (:meth:`validate_weights`).

    The order of the weight list is the *contract*: ``get_weights`` and
    ``set_weights`` must agree on ordering for a given model architecture.
    """

    framework: str = "base"

    @classmethod
    @abstractmethod
    def is_supported(cls, model: Any) -> bool:
        """Return True if ``model`` is handled by this adapter."""

    @abstractmethod
    def get_weights(self, model: Any) -> Weights:
        """Extract model parameters as an ordered list of numpy arrays."""

    @abstractmethod
    def set_weights(self, model: Any, weights: Weights) -> Any:
        """Apply an ordered list of numpy arrays to ``model`` and return it."""

    @abstractmethod
    def train(self, model: Any, train_data: Any, **kwargs: Any) -> Any:
        """Perform local training on ``train_data`` and return the model."""

    @abstractmethod
    def validate_weights(self, model: Any, weights: Weights) -> None:
        """Validate that ``weights`` are structurally compatible with ``model``.

        Raises :class:`WeightShapeMismatchError` or
        :class:`InvalidWeightsError` on any mismatch.
        """

    @abstractmethod
    def predict(self, model: Any, x: Any) -> np.ndarray:
        """Return model predictions for feature input ``x`` as numpy arrays."""

    def num_samples(self, train_data: Any) -> int:
        """Best-effort number of samples represented by ``train_data``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not know how to count samples."
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weights_to_arrays(weights: object) -> list[np.ndarray]:
        if weights is None or not isinstance(weights, (list, tuple)):
            raise InvalidWeightsError(
                "Weights must be a non-empty list of arrays."
            )
        if not weights:
            raise InvalidWeightsError("Weights cannot be empty.")
        arrays: list[np.ndarray] = []
        for i, w in enumerate(weights):
            if w is None:
                raise InvalidWeightsError(f"Weight at index {i} is None.")
            arr = np.asarray(w)
            if arr.dtype.kind not in "iufc":
                raise InvalidWeightsError(
                    f"Weight at index {i} must be numeric, got dtype "
                    f"{arr.dtype!r}."
                )
            arrays.append(arr)
        return arrays

    @staticmethod
    def _check_shapes(expected: list[np.ndarray], actual: list[np.ndarray]) -> None:
        if len(actual) != len(expected):
            raise WeightShapeMismatchError(
                "Expected a weight structure with "
                f"{len(expected)} parameters, got {len(actual)}."
            )
        for i, (exp, act) in enumerate(zip(expected, actual)):
            if tuple(act.shape) != tuple(exp.shape):
                raise WeightShapeMismatchError(
                    f"Parameter {i} shape mismatch: expected {tuple(exp.shape)}, "
                    f"got {tuple(act.shape)}."
                )
