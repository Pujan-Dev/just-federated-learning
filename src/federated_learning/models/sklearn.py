"""scikit-learn model adapter.

Only a curated set of scikit-learn estimators can be meaningfully aggregated
with Federated Averaging. Those are estimators whose learned parameters are
**numerical model parameters** (e.g. ``coef_``/``intercept_`` or
``cluster_centers_``) that can be averaged in a mathematically sound way.

Estimators that store the training data itself (e.g.
``KNeighborsClassifier``), or that learn a structure that cannot be averaged
(e.g. tree-based models such as ``DecisionTreeClassifier`` or
``RandomForestClassifier``, ``GaussianNB``, etc.) are **not** supported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from federated_learning.exceptions import (
    InvalidSampleCountError,
    InvalidWeightsError,
    WeightShapeMismatchError,
)
from federated_learning.models.base import ModelAdapter, Weights

try:  # pragma: no cover - import guard for optional dependency
    from sklearn import __version__ as _sklearn_version
    from sklearn.base import is_classifier
    from sklearn.cluster import KMeans
    from sklearn.linear_model import (
        ElasticNet,
        Lasso,
        LinearRegression,
        LogisticRegression,
        PassiveAggressiveClassifier,
        PassiveAggressiveRegressor,
        Perceptron,
        Ridge,
        RidgeClassifier,
        SGDClassifier,
        SGDRegressor,
    )
    from sklearn.svm import LinearSVC, LinearSVR
except ImportError:  # pragma: no cover
    _sklearn_version = None
    is_classifier = None
    KMeans = None
    ElasticNet = Lasso = LinearRegression = LogisticRegression = None
    PassiveAggressiveClassifier = PassiveAggressiveRegressor = Perceptron = None
    Ridge = RidgeClassifier = SGDClassifier = SGDRegressor = None
    LinearSVC = LinearSVR = None

# Estimators whose parameters can safely be averaged via FedAvg.
# Any estimator added here must expose only numeric, averageable parameters.
SUPPORTED_ESTIMATORS: tuple[type, ...] = (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
    SGDRegressor,
    SGDClassifier,
    RidgeClassifier,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
    Perceptron,
    LinearSVC,
    LinearSVR,
    KMeans,
)

_DOCUMENTED_ESTIMATORS = (
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "LogisticRegression",
    "SGDRegressor",
    "SGDClassifier",
    "RidgeClassifier",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "LinearSVC",
    "LinearSVR",
    "KMeans",
)


class SklearnAdapter(ModelAdapter):
    """Adapter for the supported scikit-learn estimators.

    Weights are the ordered list of the estimator's numeric parameters:

    * linear models: ``[coef_, intercept_]``
    * ``KMeans``: ``[cluster_centers_]``
    """

    framework = "sklearn"

    @classmethod
    def is_supported(cls, model: Any) -> bool:
        if _sklearn_version is None:
            return False
        return isinstance(model, SUPPORTED_ESTIMATORS)

    def _parameter_names(self, model: Any) -> tuple[str, ...]:
        if isinstance(model, KMeans):
            return ("cluster_centers_",)
        return ("coef_", "intercept_")

    def _require_fitted(self, model: Any) -> None:
        for name in self._parameter_names(model):
            if not hasattr(model, name):
                raise InvalidWeightsError(
                    f"Estimator {type(model).__name__} has not been fitted yet; "
                    f"no parameter {name!r} is available. Fit it locally before "
                    "extracting weights."
                )

    def get_weights(self, model: Any) -> Weights:
        self._require_fitted(model)
        return [
            np.asarray(getattr(model, name)).copy()
            for name in self._parameter_names(model)
        ]

    def set_weights(self, model: Any, weights: Weights) -> Any:
        arrays = self._weights_to_arrays(weights)
        names = self._parameter_names(model)
        if len(arrays) != len(names):
            raise WeightShapeMismatchError(
                f"Expected a weight structure with {len(names)} parameters "
                f"(for {type(model).__name__}), got {len(arrays)}."
            )
        for name, arr in zip(names, arrays):
            existing = getattr(model, name, None)
            if existing is not None and np.asarray(existing).shape != arr.shape:
                raise WeightShapeMismatchError(
                    f"Parameter {name!r} shape mismatch: expected "
                    f"{np.asarray(existing).shape}, got {arr.shape}."
                )
            setattr(model, name, np.asarray(arr).copy())
        self._complete_fit_metadata(model, arrays)
        return model

    def _complete_fit_metadata(self, model: Any, arrays: list[np.ndarray]) -> None:
        """Make a freshly aggregated classifier usable without a full refit.

        scikit-learn classifiers keep ``classes_`` (and friends) from ``fit``.
        When weights are applied to an estimator that was never fitted (the
        common server-side case), those attributes are missing and ``predict``
        would fail. We restore the label space from the coefficient shape,
        assuming integer labels ``0..C-1`` (documented limitation).
        """
        if is_classifier is None or not is_classifier(model):
            return
        if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
            return
        if not hasattr(model, "classes_"):
            n_classes = arrays[0].shape[0]
            if n_classes <= 1:
                model.classes_ = np.array([0, 1])
            else:
                model.classes_ = np.arange(n_classes)
        if not hasattr(model, "n_features_in_") and len(arrays[0].shape) >= 2:
            model.n_features_in_ = arrays[0].shape[1]

    def validate_weights(self, model: Any, weights: Weights) -> None:
        arrays = self._weights_to_arrays(weights)
        names = self._parameter_names(model)
        if len(arrays) != len(names):
            raise WeightShapeMismatchError(
                f"Expected a weight structure with {len(names)} parameters "
                f"(for {type(model).__name__}), got {len(arrays)}."
            )
        for name, arr in zip(names, arrays):
            existing = getattr(model, name, None)
            if existing is not None and np.asarray(existing).shape != arr.shape:
                raise WeightShapeMismatchError(
                    f"Parameter {name!r} shape mismatch: expected "
                    f"{np.asarray(existing).shape}, got {arr.shape}."
                )

    def train(
        self,
        model: Any,
        train_data: Any,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Train locally on ``(X, y)`` data.

        Two strategies are used, depending on the estimator:

        * Estimators exposing ``partial_fit`` (``SGDClassifier``,
          ``SGDRegressor``, ``PassiveAggressiveClassifier``,
          ``PassiveAggressiveRegressor``, ``Perceptron``) are trained
          incrementally with mini-batches starting from the currently loaded
          (global) weights. This is genuine federated learning: every round
          takes small optimization steps from the global model.
        * Other supported estimators (e.g. ``LinearRegression``,
          ``LogisticRegression``) are refit with ``fit``. When ``warm_start``
          is available it is temporarily enabled so training continues from the
          loaded weights; otherwise a fresh fit is used.

        ``epochs`` controls the number of passes over the local data (default 1).
        """
        x, y = self._split_data(train_data)

        if hasattr(model, "partial_fit"):
            return self._train_incremental(
                model, x, y, epochs=epochs, batch_size=batch_size,
                shuffle=shuffle, seed=seed,
            )

        previous_warm_start: bool | None = None
        if hasattr(model, "warm_start"):
            previous_warm_start = bool(model.warm_start)
            try:
                model.set_params(warm_start=True)
            except ValueError:  # pragma: no cover - defensive
                pass
        try:
            model.fit(x, y)
        finally:
            if previous_warm_start is not None:
                try:
                    model.set_params(warm_start=previous_warm_start)
                except ValueError:  # pragma: no cover - defensive
                    pass
        return model

    def _train_incremental(
        self,
        model: Any,
        x: Any,
        y: Any,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool,
        seed: int | None,
    ) -> Any:
        previous_warm_start: bool | None = None
        if hasattr(model, "warm_start"):
            previous_warm_start = bool(model.warm_start)
            try:
                model.set_params(warm_start=True)
            except ValueError:  # pragma: no cover - defensive
                pass

        n = len(x)
        classes = None
        if is_classifier(model):
            classes = getattr(model, "classes_", None)
            if classes is None:
                classes = np.unique(y)

        rng = np.random.default_rng(seed)
        try:
            for _ in range(epochs):
                indices = np.arange(n)
                if shuffle:
                    rng.shuffle(indices)
                for start in range(0, n, batch_size):
                    idx = indices[start : start + batch_size]
                    if is_classifier(model):
                        model.partial_fit(x[idx], y[idx], classes=classes)
                    else:
                        model.partial_fit(x[idx], y[idx])
        finally:
            if previous_warm_start is not None:
                try:
                    model.set_params(warm_start=previous_warm_start)
                except ValueError:  # pragma: no cover - defensive
                    pass
        return model

    def num_samples(self, train_data: Any) -> int:
        x, _ = self._split_data(train_data)
        return len(x)

    @staticmethod
    def _split_data(train_data: Any) -> tuple[Any, Any]:
        if (
            isinstance(train_data, (tuple, list))
            and len(train_data) == 2
            and hasattr(train_data[0], "__len__")
        ):
            return train_data[0], train_data[1]
        raise ValueError(
            "sklearn train_data must be an (X, y) pair of array-like objects."
        )

    @classmethod
    def supported_estimators(cls) -> tuple[str, ...]:
        """Names of the estimators that can be aggregated with FedAvg."""
        return _DOCUMENTED_ESTIMATORS