"""Custom exceptions used across the federated learning package."""

from __future__ import annotations

from typing import Any


class FederatedLearningError(Exception):
    """Base class for all errors raised by this package."""


class UnsupportedModelError(FederatedLearningError):
    """Raised when a model cannot be used with this package.

    Only PyTorch (``torch.nn.Module``) and scikit-learn estimators are
    supported. TensorFlow and Keras models are explicitly not supported.
    """

    def __init__(self, message: str = "", model: Any = None) -> None:
        if message and model is not None:
            message = f"{message} Got model of type {type(model).__name__!r}."
        elif not message and model is not None:
            message = (
                f"Unsupported model type {type(model).__name__!r}. Only PyTorch "
                "and scikit-learn models are supported."
            )
        super().__init__(message)


class InvalidWeightsError(FederatedLearningError):
    """Raised when a weight structure is invalid."""


class WeightShapeMismatchError(InvalidWeightsError):
    """Raised when weight shapes do not match the expected model structure."""


class InvalidClientUpdateError(FederatedLearningError):
    """Raised when a client update is malformed."""


class InvalidSampleCountError(InvalidClientUpdateError):
    """Raised when a client update carries an invalid number of samples."""
