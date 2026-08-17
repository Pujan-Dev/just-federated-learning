"""Safe, dependency-light weight serialization for client<->server transport.

Weights are converted into an explicit schema (dtype, shape, base64 bytes)
instead of pickling arbitrary Python objects. This is safe to send over HTTP
and sufficient to fully reconstruct the client update payload.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np

from federated_learning.client import ClientUpdate
from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidWeightsError,
)


def serialize_weights(weights: list[np.ndarray]) -> list[dict[str, Any]]:
    """Serialize an ordered weight list into a JSON-safe list of dicts."""
    if weights is None or not isinstance(weights, (list, tuple)) or not weights:
        raise InvalidWeightsError("Weights must be a non-empty list of arrays.")
    payload: list[dict[str, Any]] = []
    for i, w in enumerate(weights):
        arr = np.asarray(w)
        if arr.dtype.kind not in "iufc":
            raise InvalidWeightsError(
                f"Weight at index {i} must be numeric, got dtype {arr.dtype!r}."
            )
        payload.append(
            {
                "dtype": arr.dtype.str,
                "shape": [int(s) for s in arr.shape],
                "data": base64.b64encode(
                    np.ascontiguousarray(arr).tobytes()
                ).decode("ascii"),
            }
        )
    return payload


def deserialize_weights(payload: list[dict[str, Any]]) -> list[np.ndarray]:
    """Rebuild a weight list from the JSON-safe representation."""
    if payload is None or not isinstance(payload, list) or not payload:
        raise InvalidWeightsError("Serialized weights must be a non-empty list.")
    weights: list[np.ndarray] = []
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise InvalidWeightsError(
                f"Serialized weight {i} must be a dict, got {type(item).__name__!r}."
            )
        try:
            dtype = np.dtype(item["dtype"])
            shape = tuple(int(s) for s in item["shape"])
            raw = base64.b64decode(item["data"], validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise InvalidWeightsError(
                f"Serialized weight {i} is malformed: {exc}."
            ) from exc
        expected_items = int(np.prod(shape, dtype=np.int64)) if shape else 1
        expected_bytes = expected_items * dtype.itemsize
        if len(raw) != expected_bytes:
            raise InvalidWeightsError(
                f"Serialized weight {i} size mismatch: expected {shape} "
                f"({expected_bytes} bytes), got {len(raw)} bytes."
            )
        weights.append(np.frombuffer(raw, dtype=dtype).reshape(shape))
    return weights


def update_to_dict(update: ClientUpdate) -> dict[str, Any]:
    """Convert a :class:`ClientUpdate` into a JSON-safe dict."""
    update.validate()
    payload: dict[str, Any] = {
        "client_id": update.client_id,
        "num_samples": update.num_samples,
        "weights": serialize_weights(update.weights),
    }
    if update.metrics:
        payload["metrics"] = {k: float(v) for k, v in update.metrics.items()}
    return payload


def update_from_dict(payload: dict[str, Any]) -> ClientUpdate:
    """Rebuild and validate a :class:`ClientUpdate` from a dict."""
    if not isinstance(payload, dict):
        raise InvalidClientUpdateError(
            f"Update payload must be a dict, got {type(payload).__name__!r}."
        )
    client_id = payload.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise InvalidClientUpdateError("client_id must be a non-empty string.")
    num_samples = payload.get("num_samples")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
        raise InvalidClientUpdateError(
            f"num_samples must be a positive integer, got {num_samples!r}."
        )
    weights = deserialize_weights(payload.get("weights"))
    metrics = payload.get("metrics")
    if metrics is not None:
        if not isinstance(metrics, dict):
            raise InvalidClientUpdateError("metrics must be a dict.")
        metrics = {str(k): float(v) for k, v in metrics.items()}
    update = ClientUpdate(
        client_id=client_id,
        weights=weights,
        num_samples=num_samples,
        metrics=metrics,
    )
    update.validate()
    return update