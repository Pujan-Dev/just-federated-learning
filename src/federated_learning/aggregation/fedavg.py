"""Federated Averaging (FedAvg) aggregation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidSampleCountError,
    InvalidWeightsError,
    WeightShapeMismatchError,
)

Weights = list[np.ndarray]


def _as_weight_arrays(weights: Sequence[object]) -> list[np.ndarray]:
    """Coerce raw client weights into a validated list of ndarrays."""
    if weights is None or not isinstance(weights, (list, tuple)):
        raise InvalidWeightsError(
            "Client weights must be a non-empty sequence of arrays."
        )
    if len(weights) == 0:
        raise InvalidWeightsError("Client weights cannot be empty.")
    arrays: list[np.ndarray] = []
    for i, w in enumerate(weights):
        if w is None:
            raise InvalidWeightsError(f"Weight at index {i} is None.")
        arr = np.asarray(w)
        if arr.dtype.kind not in "iufc":
            raise InvalidWeightsError(
                f"Weight at index {i} must be numeric, got dtype {arr.dtype!r}."
            )
        arrays.append(arr)
    return arrays


def _validate_num_samples(num_samples: object) -> int:
    if not isinstance(num_samples, (int, np.integer)):
        raise InvalidSampleCountError(
            f"num_samples must be an integer, got {type(num_samples).__name__!r}."
        )
    value = int(num_samples)
    if value <= 0:
        raise InvalidSampleCountError(
            f"num_samples must be a positive integer, got {value}."
        )
    return value


def _weights_from_update(update: object) -> tuple[list[np.ndarray], int]:
    """Extract and validate (weights, num_samples) from a client update.

    Any object exposing ``.weights`` and ``.num_samples`` attributes is
    accepted so that the aggregator stays decoupled from the client code.
    """
    if update is None:
        raise InvalidClientUpdateError("Client update cannot be None.")
    if not hasattr(update, "weights") or not hasattr(update, "num_samples"):
        raise InvalidClientUpdateError(
            "Client update must expose 'weights' and 'num_samples' attributes."
        )
    weights = _as_weight_arrays(getattr(update, "weights"))
    num_samples = _validate_num_samples(getattr(update, "num_samples"))
    return weights, num_samples


class FedAvg:
    """Weighted Federated Averaging.

    The global weights are the sample-count weighted average of the client
    weight vectors:

        W_global = (sum_i n_i * W_i) / (sum_i n_i)

    where ``W_i`` are the weights of client ``i`` and ``n_i`` is the number of
    local samples used to train them.
    """

    def aggregate(self, updates: Sequence[object]) -> list[np.ndarray]:
        """Aggregate a sequence of client updates into one weight vector.

        Parameters
        ----------
        updates:
            Non-empty sequence of client updates. Each update must expose
            ``weights`` (an ordered sequence of ndarray-like parameters) and
            ``num_samples`` (a positive integer).

        Returns
        -------
        list[np.ndarray]
            The aggregated global weights. Computed in float64.
        """
        if not updates:
            raise InvalidClientUpdateError(
                "Cannot aggregate: at least one client update is required."
            )

        parsed = [_weights_from_update(u) for u in updates]

        reference_weights, _ = parsed[0]
        reference_shapes = [w.shape for w in reference_weights]

        for weights, _ in parsed:
            if len(weights) != len(reference_weights):
                raise WeightShapeMismatchError(
                    "All clients must report the same number of parameters "
                    f"(expected {len(reference_weights)}, got {len(weights)})."
                )
            for i, (ref_shape, w_shape) in enumerate(
                zip(reference_shapes, (w.shape for w in weights))
            ):
                if w_shape != ref_shape:
                    raise WeightShapeMismatchError(
                        f"Parameter {i} shape mismatch: expected {ref_shape}, "
                        f"got {w_shape}."
                    )

        total_samples = sum(n for _, n in parsed)

        aggregated: list[np.ndarray] = []
        for i in range(len(reference_weights)):
            weighted = np.zeros(reference_shapes[i], dtype=np.float64)
            for weights, num_samples in parsed:
                weighted = weighted + (num_samples * weights[i].astype(np.float64))
            aggregated.append(weighted / total_samples)

        return aggregated


def fedavg(updates: Sequence[object]) -> list[np.ndarray]:
    """Module-level convenience wrapper around :class:`FedAvg`."""
    return FedAvg().aggregate(updates)


__all__ = ["FedAvg", "fedavg"]
