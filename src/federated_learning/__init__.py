"""federated_learning: a framework-agnostic Federated Learning library.

Only PyTorch and scikit-learn models are supported. TensorFlow/Keras models
are explicitly **not** supported.
"""

from __future__ import annotations

from federated_learning.aggregation import FedAvg, fedavg
from federated_learning.client import ClientUpdate, FederatedClient
from federated_learning.exceptions import (
    FederatedLearningError,
    InvalidClientUpdateError,
    InvalidSampleCountError,
    InvalidWeightsError,
    UnsupportedModelError,
    WeightShapeMismatchError,
)
from federated_learning.metrics import (
    accuracy,
    f1,
    mae,
    mse,
    precision,
    r2,
    recall,
    rmse,
)
from federated_learning.models import ModelAdapter, create_adapter
from federated_learning.server import FederatedServer
from federated_learning.trainer import FederatedTrainer

__version__ = "0.1.0"

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FederatedTrainer",
    "ClientUpdate",
    "FedAvg",
    "fedavg",
    "ModelAdapter",
    "create_adapter",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "mae",
    "rmse",
    "r2",
    "FederatedLearningError",
    "UnsupportedModelError",
    "InvalidWeightsError",
    "WeightShapeMismatchError",
    "InvalidClientUpdateError",
    "InvalidSampleCountError",
]

_LAZY_EXPORTS = {
    "PyTorchAdapter": "federated_learning.models.pytorch",
    "SklearnAdapter": "federated_learning.models.sklearn",
}


def __getattr__(name: str):
    # Lazily import framework-specific adapters so that the core package can
    # be imported without torch/scikit-learn being installed.
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))