"""Tests for the FederatedServer."""

import numpy as np
import pytest
import torch
from torch import nn

from federated_learning import FederatedServer
from federated_learning.client import ClientUpdate
from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidSampleCountError,
    WeightShapeMismatchError,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


def _make_server(**kwargs):
    torch.manual_seed(0)
    defaults = dict(model=TinyModel())
    defaults.update(kwargs)
    return FederatedServer(**defaults)


def _update(cid, weights, num_samples):
    return ClientUpdate(client_id=cid, weights=weights, num_samples=num_samples)


def _random_weights(shape=(3, 1)):
    return [np.random.default_rng(0).standard_normal(shape)]


def _tiny_weights(w=0.0, b=0.0):
    """Weights matching TinyModel: fc.weight (1, 3) and fc.bias (1,)."""
    return [np.full((1, 3), w), np.array([b])]


def test_initial_global_weights_match_model():
    server = _make_server()
    model_weights = server.adapter.get_weights(server.model)
    global_weights = server.get_global_weights()
    assert len(global_weights) == len(model_weights)
    for a, b in zip(global_weights, model_weights):
        np.testing.assert_allclose(a, b)


def test_global_weights_are_copies():
    server = _make_server()
    first = server.get_global_weights()
    first[0][0, 0] = 12345.0
    second = server.get_global_weights()
    assert second[0][0, 0] != 12345.0


def test_receive_and_aggregate_updates():
    server = _make_server()
    server.receive_update(_update("a", _tiny_weights(w=2.0, b=1.0), 3))
    server.receive_update(_update("b", _tiny_weights(w=4.0, b=3.0), 1))
    aggregated = server.aggregate()
    expected = (3 * np.full((1, 3), 2.0) + 1 * np.full((1, 3), 4.0)) / 4
    np.testing.assert_allclose(aggregated[0], expected)
    np.testing.assert_allclose(aggregated[1], np.array((3 * 1.0 + 1 * 3.0) / 4))


def test_aggregate_with_explicit_updates_ignores_stored():
    server = _make_server()
    aggregated = server.aggregate([_update("a", [np.array([10.0, 0.0, 0.0])], 1)])
    np.testing.assert_allclose(aggregated[0], [10.0, 0.0, 0.0])


def test_aggregate_with_no_updates_raises():
    server = _make_server()
    with pytest.raises(InvalidClientUpdateError, match="No client updates"):
        server.aggregate()


def test_rejects_non_client_update():
    server = _make_server()
    with pytest.raises(InvalidClientUpdateError, match="ClientUpdate"):
        server.receive_update({"client_id": "x", "weights": [], "num_samples": 1})


def test_rejects_invalid_sample_count():
    server = _make_server()
    with pytest.raises(InvalidSampleCountError):
        server.receive_update(_update("a", _random_weights(), 0))
    with pytest.raises(InvalidSampleCountError):
        server.receive_update(_update("a", _random_weights(), -5))
    with pytest.raises(InvalidSampleCountError):
        server.receive_update(_update("a", _random_weights(), 2.5))


def test_rejects_missing_client_id():
    server = _make_server()
    with pytest.raises(InvalidClientUpdateError, match="client_id"):
        server.receive_update(_update("", _random_weights(), 5))


def test_rejects_shape_mismatch_against_global():
    server = _make_server()
    bad = [np.zeros((99, 99))]
    with pytest.raises(WeightShapeMismatchError):
        server.receive_update(_update("a", bad, 5))


def test_update_global_model_and_round_increment():
    server = _make_server()
    weights = server.adapter.get_weights(server.model)
    shifted = [w + 1.0 for w in weights]
    server.update_global_model(shifted)
    assert server.round == 1
    applied = server.get_global_weights()
    for a, b in zip(shifted, applied):
        np.testing.assert_allclose(a, b)


def test_aggregate_and_update_flow():
    server = _make_server()
    for i in range(3):
        server.receive_update(_update(f"c{i}", _tiny_weights(w=i + 1.0), 1))
    new_weights = server.aggregate_and_update()
    assert server.round == 1
    np.testing.assert_allclose(new_weights[0], np.full((1, 3), 2.0))


def test_reset_clears_stored_updates():
    server = _make_server()
    server.receive_update(_update("a", _tiny_weights(), 5))
    server.reset()
    with pytest.raises(InvalidClientUpdateError, match="No client updates"):
        server.aggregate()


def test_sklearn_server_initialized_unfitted():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=20, n_features=2, random_state=0)
    model = LinearRegression()
    server = FederatedServer(model=model)
    assert server.has_global_weights is False
    assert server.get_global_weights() is None

    # First round establishes the global weights.
    server.receive_update(_update("a", [np.array([1.0, 2.0]), np.array([0.5])], 10))
    server.receive_update(_update("b", [np.array([2.0, 3.0]), np.array([1.5])], 10))
    new_weights = server.aggregate_and_update()
    assert server.has_global_weights is True
    np.testing.assert_allclose(new_weights[0], [1.5, 2.5])
    np.testing.assert_allclose(new_weights[1], np.array(1.0))
    assert server.get_model() is model
    # Model now carries the aggregated coefficients.
    assert model.coef_.shape == (2,)