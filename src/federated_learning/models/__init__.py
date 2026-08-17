"""Model adapters: a framework-agnostic layer over PyTorch / scikit-learn models."""

from __future__ import annotations

from typing import Any

from federated_learning.exceptions import UnsupportedModelError
from federated_learning.models.base import ModelAdapter

_TENSORFLOW_KERAS_MESSAGE = (
    "Only PyTorch and scikit-learn models are supported. "
    "TensorFlow/Keras models are not supported."
)


def _looks_like_tensorflow_keras(model: Any) -> bool:
    module = type(model).__module__ or ""
    return module.startswith("tensorflow") or module.startswith("keras")


def create_adapter(model: Any) -> ModelAdapter:
    """Return the adapter that handles ``model``.

    Raises
    ------
    UnsupportedModelError
        If the model is a TensorFlow/Keras model or belongs to any other
        unsupported framework.
    """
    from federated_learning.models.pytorch import PyTorchAdapter
    from federated_learning.models.sklearn import SklearnAdapter

    if PyTorchAdapter.is_supported(model):
        return PyTorchAdapter()
    if SklearnAdapter.is_supported(model):
        return SklearnAdapter()
    if _looks_like_tensorflow_keras(model):
        raise UnsupportedModelError(_TENSORFLOW_KERAS_MESSAGE, model=model)
    raise UnsupportedModelError(model=model)


__all__ = ["ModelAdapter", "create_adapter", "UnsupportedModelError"]