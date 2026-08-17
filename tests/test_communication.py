"""Tests for weight serialization and the optional HTTP communication layer."""

import numpy as np
import pytest

from federated_learning import FederatedClient, FederatedServer
from federated_learning.client import ClientUpdate
from federated_learning.communication.http import (
    FastAPIServer,
    HTTPClient,
    HTTPClientChannel,
)
from federated_learning.communication.serialization import (
    deserialize_weights,
    serialize_weights,
    update_from_dict,
    update_to_dict,
)
from federated_learning.exceptions import InvalidClientUpdateError

pytest.importorskip("httpx")


def _arrays():
    return [
        np.array([[1.5, -2.0], [0.5, 3.0]], dtype=np.float32),
        np.array([0.25, -1.0], dtype=np.float32),
    ]


def test_serialize_round_trip():
    original = _arrays()
    payload = serialize_weights(original)
    restored = deserialize_weights(payload)
    assert len(restored) == len(original)
    for a, b in zip(original, restored):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape


def test_serialize_rejects_empty():
    with pytest.raises(Exception, match="non-empty"):
        serialize_weights([])


def test_serialize_malformed_payload():
    with pytest.raises(Exception, match="malformed"):
        deserialize_weights([{"dtype": "<f4", "shape": [1]}])


def test_serialize_size_mismatch():
    with pytest.raises(Exception, match="size mismatch"):
        deserialize_weights(
            [{"dtype": "<f4", "shape": [5], "data": "AAAAAAAA"}]  # 2 floats -> size 2
        )


def test_update_dict_round_trip():
    update = ClientUpdate(client_id="c1", weights=_arrays(), num_samples=12)
    payload = update_to_dict(update)
    assert set(payload) == {"client_id", "num_samples", "weights"}
    restored = update_from_dict(payload)
    assert restored.client_id == "c1"
    assert restored.num_samples == 12
    assert restored.metrics is None
    for a, b in zip(update.weights, restored.weights):
        np.testing.assert_array_equal(a, b)


def test_update_dict_round_trip_with_metrics():
    update = ClientUpdate(
        client_id="c1",
        weights=_arrays(),
        num_samples=12,
        metrics={"accuracy": 0.9, "loss": 0.25},
    )
    payload = update_to_dict(update)
    assert payload["metrics"] == {"accuracy": 0.9, "loss": 0.25}
    restored = update_from_dict(payload)
    assert restored.metrics == {"accuracy": 0.9, "loss": 0.25}


def test_update_from_dict_validates():
    with pytest.raises(InvalidClientUpdateError, match="client_id"):
        update_from_dict({"client_id": "", "num_samples": 1, "weights": []})
    with pytest.raises(InvalidClientUpdateError, match="num_samples"):
        update_from_dict(
            {"client_id": "c", "num_samples": -1, "weights": serialize_weights(_arrays())}
        )
    with pytest.raises(Exception, match="non-empty"):
        update_from_dict({"client_id": "c", "num_samples": 1, "weights": []})


def _build_fastapi_server():
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 1)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    server = FederatedServer(model=Net())
    return FastAPIServer(server), server


def test_fastapi_endpoints_via_testclient():
    from fastapi.testclient import TestClient

    pytest.importorskip("fastapi")
    fast_api, server = _build_fastapi_server()
    client = TestClient(fast_api.app())

    assert client.get("/health").json() == {"status": "ok"}

    weights = server.get_global_weights()
    payload = client.get("/global-weights").json()
    assert payload["round"] == 0
    restored = deserialize_weights(payload["weights"])
    for a, b in zip(weights, restored):
        np.testing.assert_array_equal(a, b)

    # Bad update (non-positive sample count) -> 400.
    resp = client.post(
        "/updates",
        json={"client_id": "c", "num_samples": 0, "weights": serialize_weights(_arrays())},
    )
    assert resp.status_code == 400

    # Missing weights -> 422 (schema validation).
    resp = client.post("/updates", json={"client_id": "c", "num_samples": 5})
    assert resp.status_code == 422

    # Good updates then aggregate (shapes must match the Net model: weight (1, 2), bias (1,)).
    u1 = update_to_dict(ClientUpdate("a", [np.full((1, 2), 2.0), np.array([0.0])], 3))
    u2 = update_to_dict(ClientUpdate("b", [np.full((1, 2), 6.0), np.array([0.0])], 1))
    assert client.post("/updates", json=u1).json()["status"] == "ok"
    assert client.post("/updates", json=u2).json()["status"] == "ok"

    agg = client.post("/aggregate").json()
    assert agg["round"] == 1
    averaged = deserialize_weights(agg["weights"])
    np.testing.assert_allclose(averaged[0], np.full((1, 2), 3.0))

    # A server without configured metrics records an empty history.
    assert client.get("/metrics").json()["history"] == [
        {"round": 1, "global": {}, "clients": {}}
    ]


def test_remote_client_full_round():
    import torch
    from torch import nn

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 1)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    server = FederatedServer(model=Net())
    fast_api = FastAPIServer(server)
    channel = HTTPClientChannel("http://testserver", client=TestClient(fast_api.app()))

    def make_client(i):
        torch.manual_seed(0)
        x = np.random.default_rng(i).standard_normal((20, 2)).astype(np.float32)
        y = np.random.default_rng(100 + i).standard_normal((20, 1)).astype(np.float32)
        return FederatedClient(
            client_id=f"remote-{i}",
            model=Net(),
            train_data=(x, y),
            local_epochs=2,
            learning_rate=0.05,
            seed=0,
        )

    remotes = [HTTPClient(make_client(i), channel=channel) for i in range(3)]
    for r in remotes:
        update = r.run_round()
        assert update.client_id.startswith("remote-")

    # No aggregation yet, but updates were received and stored.
    assert server.round == 0

    # Trigger aggregation over HTTP and check the round advanced.
    agg = channel._client.post(f"{channel.base_url}/aggregate")
    assert agg.status_code == 200
    assert server.round == 1
    assert server.get_global_weights() is not None

    # A fresh remote client can adopt the global weights for the next round.
    r = HTTPClient(make_client(9), channel=channel)
    r.run_round()
    assert len(server.get_global_weights()) == 2


def test_metrics_over_http():
    import torch
    from torch import nn

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    eval_x = rng.standard_normal((30, 2)).astype(np.float32)
    eval_y = (eval_x[:, 0] > 0).astype(np.int64)

    server = FederatedServer(
        model=Net(),
        metrics=["accuracy"],
        evaluation_data=(eval_x, eval_y),
    )
    fast_api = FastAPIServer(server)
    channel = HTTPClientChannel("http://testserver", client=TestClient(fast_api.app()))
    assert channel.get_health() is True

    def make_client(i):
        x = rng.standard_normal((20, 2)).astype(np.float32)
        y = (x[:, 0] > 0).astype(np.int64)
        return FederatedClient(
            client_id=f"client-{i}",
            model=Net(),
            train_data=(x, y),
            local_epochs=1,
            learning_rate=0.01,
            seed=0,
            criterion=nn.CrossEntropyLoss(),
            metrics=["accuracy"],
        )

    remotes = [HTTPClient(make_client(i), channel=channel) for i in range(2)]
    for r in remotes:
        update = r.run_round()
        assert update.metrics is not None
        assert "accuracy" in update.metrics

    agg = channel.aggregate()
    assert agg["round"] == 1

    history = channel.get_metrics()
    assert len(history) == 1
    entry = history[0]
    assert entry["round"] == 1
    assert "accuracy" in entry["global"]
    assert set(entry["clients"]) == {"client-0", "client-1"}
    for client_id, metrics in entry["clients"].items():
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0