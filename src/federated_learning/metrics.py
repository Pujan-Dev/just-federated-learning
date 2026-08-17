"""Evaluation metrics for federated learning.

Metrics operate on ``(y_true, y_pred)`` numpy arrays and are framework
agnostic (they work with predictions produced by either adapter).

Classification metrics (``accuracy``, ``precision``, ``recall``, ``f1``)
accept hard labels *or* 2D score matrices (probabilities / logits): the
argmax over the class dimension is applied automatically. Regression metrics
(``mse``, ``mae``, ``rmse``, ``r2``) flatten their inputs.

Custom metrics can be any callable ``(y_true, y_pred) -> float``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def _as_labels(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim == 1 and yp.ndim == 2 and yp.shape[1] > 1:
        yp = yp.argmax(axis=1)
    return yt.ravel(), yp.ravel()


def _class_scores(
    y_true: Any, y_pred: Any, which: str
) -> np.ndarray:
    yt, yp = _as_labels(y_true, y_pred)
    classes = np.unique(yt)
    per_class = []
    for c in classes:
        tp = float(np.sum((yp == c) & (yt == c)))
        fp = float(np.sum((yp == c) & (yt != c)))
        fn = float(np.sum((yp != c) & (yt == c)))
        if which == "precision":
            per_class.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        elif which == "recall":
            per_class.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        else:  # f1
            denom = 2 * tp + fp + fn
            per_class.append(2 * tp / denom if denom > 0 else 0.0)
    return np.asarray(per_class)


def accuracy(y_true: Any, y_pred: Any) -> float:
    """Fraction of correct predictions (macro none, plain accuracy)."""
    yt, yp = _as_labels(y_true, y_pred)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def precision(y_true: Any, y_pred: Any) -> float:
    """Macro-averaged precision over the classes in ``y_true``."""
    scores = _class_scores(y_true, y_pred, "precision")
    return float(np.mean(scores)) if len(scores) else 0.0


def recall(y_true: Any, y_pred: Any) -> float:
    """Macro-averaged recall over the classes in ``y_true``."""
    scores = _class_scores(y_true, y_pred, "recall")
    return float(np.mean(scores)) if len(scores) else 0.0


def f1(y_true: Any, y_pred: Any) -> float:
    """Macro-averaged F1 score over the classes in ``y_true``."""
    scores = _class_scores(y_true, y_pred, "f1")
    return float(np.mean(scores)) if len(scores) else 0.0


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def _as_1d(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()


def mse(y_true: Any, y_pred: Any) -> float:
    """Mean squared error."""
    yt, yp = _as_1d(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2)) if len(yt) else 0.0


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""
    yt, yp = _as_1d(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp))) if len(yt) else 0.0


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination (R^2)."""
    yt, yp = _as_1d(y_true, y_pred)
    if len(yt) == 0:
        return 0.0
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Registry + resolution
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, Metric] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "mse": mse,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}


def resolve_metrics(metrics: Any) -> dict[str, Metric]:
    """Normalize a flexible metric specification into ``{name: callable}``.

    Accepted inputs:

    * ``None`` -> ``{}``
    * a metric name string (e.g. ``"accuracy"``)
    * a callable ``(y_true, y_pred) -> float``
    * a sequence of names/callables
    * a mapping ``{name: callable_or_name}``
    """
    if metrics is None:
        return {}
    if isinstance(metrics, str):
        return {metrics: _lookup(metrics)}
    if callable(metrics):
        return {getattr(metrics, "__name__", "metric"): metrics}
    if isinstance(metrics, Mapping):
        result: dict[str, Metric] = {}
        for name, fn in metrics.items():
            resolved = _lookup(fn) if isinstance(fn, str) else fn
            if not callable(resolved):
                raise ValueError(
                    f"Metric {name!r} must be a callable, got {type(fn).__name__!r}."
                )
            result[str(name)] = resolved
        return result
    if isinstance(metrics, Sequence):
        result = {}
        for item in metrics:
            result.update(resolve_metrics(item))
        return result
    raise ValueError(
        "metrics must be a name, callable, sequence, or mapping; "
        f"got {type(metrics).__name__!r}."
    )


def _lookup(name: str) -> Metric:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric {name!r}. Available: {sorted(METRIC_REGISTRY)}."
        )
    return METRIC_REGISTRY[name]


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _collect_predictions(adapter: Any, model: Any, data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` arrays for a model on ``data``.

    ``data`` may be an ``(X, y)`` pair or a batched iterable (e.g. a PyTorch
    ``DataLoader`` yielding ``(x_batch, y_batch)``).
    """
    if (
        isinstance(data, (tuple, list))
        and len(data) == 2
        and hasattr(data[0], "__len__")
    ):
        x, y = data
        return np.asarray(y), np.asarray(adapter.predict(model, x))
    try:
        y_true_parts: list[np.ndarray] = []
        y_pred_parts: list[np.ndarray] = []
        for x_batch, y_batch in data:
            y_true_parts.append(np.asarray(y_batch).reshape(-1))
            y_pred_parts.append(np.asarray(adapter.predict(model, x_batch)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation data must be an (X, y) pair or a batched iterable "
            "such as a DataLoader."
        ) from exc
    if not y_true_parts:
        raise ValueError("evaluation data contains no batches.")
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts, axis=0)


def evaluate_model(
    adapter: Any,
    model: Any,
    data: Any,
    metrics: Mapping[str, Metric] | Sequence[str] | Any,
) -> dict[str, float]:
    """Compute ``metrics`` for ``model`` on ``data`` and return ``{name: value}``."""
    resolved = resolve_metrics(metrics)
    if not resolved:
        return {}
    y_true, y_pred = _collect_predictions(adapter, model, data)
    return {name: fn(y_true, y_pred) for name, fn in resolved.items()}


__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "mae",
    "rmse",
    "r2",
    "resolve_metrics",
    "evaluate_model",
    "METRIC_REGISTRY",
]