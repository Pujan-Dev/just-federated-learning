"""Tests for the metrics module."""

import numpy as np
import pytest

from federated_learning.metrics import (
    METRIC_REGISTRY,
    accuracy,
    evaluate_model,
    f1,
    mae,
    mse,
    precision,
    r2,
    recall,
    resolve_metrics,
    rmse,
)


def test_accuracy_hard_labels():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_accuracy_with_score_matrix():
    # 2D scores are argmax'ed automatically.
    y_true = np.array([0, 1, 1])
    y_pred = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8]])
    assert accuracy(y_true, y_pred) == 1.0


def test_precision_recall_f1_binary():
    y_true = np.array([1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 1])
    # per-class: class 0 -> tp=1 fp=1 fn=1 ; class 1 -> tp=3 fp=1 fn=1
    assert precision(y_true, y_pred) == pytest.approx(0.625)
    assert recall(y_true, y_pred) == pytest.approx(0.625)
    assert f1(y_true, y_pred) == pytest.approx(0.625)


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 2.0, 2.0, 4.0])
    assert mse(y_true, y_pred) == pytest.approx(0.5)
    assert mae(y_true, y_pred) == pytest.approx(0.5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.5))
    assert r2(y_true, y_pred) == pytest.approx(0.6)


def test_r2_constant_target_is_zero():
    assert r2([3.0, 3.0, 3.0], [1.0, 2.0, 4.0]) == 0.0


def test_registry_complete():
    for name in ("accuracy", "precision", "recall", "f1", "mse", "mae", "rmse", "r2"):
        assert name in METRIC_REGISTRY


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, set()),
        ("accuracy", {"accuracy"}),
        (["accuracy", "f1"], {"accuracy", "f1"}),
        ({"acc": "accuracy", "custom": lambda a, b: 1.0}, {"acc", "custom"}),
        (accuracy, {"accuracy"}),
    ],
)
def test_resolve_metrics(spec, expected):
    resolved = resolve_metrics(spec)
    assert set(resolved) == expected
    assert all(callable(fn) for fn in resolved.values())


def test_resolve_unknown_metric_name():
    with pytest.raises(ValueError, match="Unknown metric"):
        resolve_metrics("nope")


def test_resolve_invalid_type():
    with pytest.raises(ValueError, match="metrics must be"):
        resolve_metrics(42)


def _dummy_model(x):
    """Linear model returning 2x[:, 0] - x[:, 1]."""


def test_evaluate_with_sklearn_adapter():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=60, n_features=2, noise=0.0, random_state=0)
    model = LinearRegression().fit(X, y)

    from federated_learning import SklearnAdapter

    result = evaluate_model(
        SklearnAdapter(), model, (X, y), {"r2": "r2", "mse": "mse"}
    )
    assert result["r2"] == pytest.approx(1.0, abs=1e-6)
    assert result["mse"] < 1e-8


def test_evaluate_with_pytorch_data_loader():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from federated_learning import PyTorchAdapter

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    model = Net()
    # Set weights so predictions are deterministic.
    with torch.no_grad():
        model.fc.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.fc.bias.zero_()

    x = torch.randn(20, 2)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=5)

    result = evaluate_model(PyTorchAdapter(), model, loader, {"accuracy": "accuracy"})
    # logits == [x0, x1] (identity weight, zero bias), so the predicted label
    # is argmax of x along the class axis.
    expected = float(np.mean(np.argmax(x.numpy(), axis=1) == y.numpy()))
    assert result["accuracy"] == pytest.approx(expected)


def test_evaluate_unknown_metric():
    from federated_learning import SklearnAdapter
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    model = LinearRegression().fit(X, y)
    with pytest.raises(ValueError, match="Unknown metric"):
        evaluate_model(SklearnAdapter(), model, (X, y), ["nope"])