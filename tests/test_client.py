"""Tests for the FederatedClient."""

import numpy as np
import pytest
import torch
from torch import nn

from federated_learning import FederatedClient
from federated_learning.client import ClientUpdate


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


def _data(n=24, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3)).astype(np.float32)
    y = rng.standard_normal((n, 1)).astype(np.float32)
    return x, y


def _make_client(**kwargs):
    torch.manual_seed(0)
    model = TinyModel()
    x, y = _data()
    defaults = dict(
        client_id="c1",
        model=model,
        train_data=(x, y),
        local_epochs=1,
        learning_rate=0.1,
        seed=0,
    )
    defaults.update(kwargs)
    return FederatedClient(**defaults)


def test_local_training_changes_weights():
    client = _make_client()
    before = client.get_weights()
    client.train()
    after = client.get_weights()
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_local_training_without_server():
    client = _make_client()
    client.train()
    assert client.model is not None
    # No server involved: nothing to raise, weights simply changed.
    assert client.get_weights()


def test_num_samples_detected_from_data():
    client = _make_client(train_data=_data(n=17))
    assert client.num_samples == 17


def test_num_samples_override():
    client = _make_client(num_samples=999)
    assert client.num_samples == 999


def test_set_weights_applies_global():
    client_a = _make_client(client_id="a")
    client_b = _make_client(client_id="b")
    client_a.train()
    global_weights = client_a.get_weights()
    client_b.set_weights(global_weights)
    for ga, gb in zip(global_weights, client_b.get_weights()):
        np.testing.assert_allclose(ga, gb)


def test_get_update_shape_and_contents():
    client = _make_client()
    client.train()
    update = client.get_update()
    assert isinstance(update, ClientUpdate)
    assert update.client_id == "c1"
    assert update.num_samples == client.num_samples == 24
    assert len(update.weights) == len(client.get_weights())


def test_update_validates():
    good = ClientUpdate(client_id="c", weights=[np.array([1.0])], num_samples=5)
    good.validate()

    with pytest.raises(Exception):
        ClientUpdate(client_id="", weights=[np.array([1.0])], num_samples=5).validate()
    with pytest.raises(Exception):
        ClientUpdate(client_id="c", weights=[], num_samples=5).validate()
    with pytest.raises(Exception):
        ClientUpdate(client_id="c", weights=[np.array([1.0])], num_samples=0).validate()


def test_requires_nonempty_client_id():
    with pytest.raises(ValueError, match="client_id"):
        _make_client(client_id="")


def test_requires_positive_local_epochs():
    with pytest.raises(ValueError, match="local_epochs"):
        _make_client(local_epochs=0)


def test_sklearn_client_flow():
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=30, n_features=2, random_state=0)
    client = FederatedClient(
        client_id="sk",
        model=LinearRegression(),
        train_data=(X, y),
    )
    assert client.num_samples == 30
    client.train()
    update = client.get_update()
    assert len(update.weights) == 2
    scaled = [w * 2 for w in update.weights]
    client.set_weights(scaled)
    restored = client.get_weights()
    for a, b in zip(scaled, restored):
        np.testing.assert_allclose(a, b)