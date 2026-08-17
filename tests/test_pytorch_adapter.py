"""Tests for the PyTorch model adapter."""

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated_learning import PyTorchAdapter, create_adapter
from federated_learning.exceptions import (
    InvalidWeightsError,
    UnsupportedModelError,
    WeightShapeMismatchError,
)


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_model():
    torch.manual_seed(0)
    return SimpleMLP()


def test_is_supported():
    assert PyTorchAdapter.is_supported(_make_model()) is True
    assert PyTorchAdapter.is_supported("not a model") is False


def test_factory_creates_pytorch_adapter():
    assert isinstance(create_adapter(_make_model()), PyTorchAdapter)


def test_get_weights_count_and_order():
    model = _make_model()
    adapter = PyTorchAdapter()
    weights = adapter.get_weights(model)
    state = model.state_dict()
    assert len(weights) == len(state)
    for arr, param in zip(weights, state.values()):
        assert arr.shape == param.shape


def test_set_weights_round_trip():
    model = _make_model()
    adapter = PyTorchAdapter()
    original = adapter.get_weights(model)

    modified = [w * 2.0 for w in original]
    adapter.set_weights(model, modified)

    restored = adapter.get_weights(model)
    for a, b in zip(modified, restored):
        np.testing.assert_allclose(a, b)


def test_weight_shape_mismatch_on_set():
    model = _make_model()
    adapter = PyTorchAdapter()
    weights = [np.zeros((5, 5), dtype=np.float32)] * len(model.state_dict())
    with pytest.raises(WeightShapeMismatchError, match="shape mismatch"):
        adapter.set_weights(model, weights)


def test_parameter_count_mismatch_on_set():
    model = _make_model()
    adapter = PyTorchAdapter()
    weights = [np.zeros((4, 8), dtype=np.float32)]  # too few
    with pytest.raises(WeightShapeMismatchError, match="parameters"):
        adapter.set_weights(model, weights)


def test_validate_weights():
    model = _make_model()
    adapter = PyTorchAdapter()
    adapter.validate_weights(model, adapter.get_weights(model))
    bad = [np.zeros((1, 1), dtype=np.float32)] * len(model.state_dict())
    with pytest.raises(WeightShapeMismatchError):
        adapter.validate_weights(model, bad)


def test_train_changes_weights_with_tensors():
    torch.manual_seed(1)
    model = _make_model()
    adapter = PyTorchAdapter()
    before = adapter.get_weights(model)
    x = torch.randn(64, 4)
    y = torch.randn(64, 2)
    adapter.train(
        model,
        (x, y),
        epochs=5,
        batch_size=16,
        lr=0.1,
    )
    after = adapter.get_weights(model)
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_train_with_numpy_data_and_data_loader():
    torch.manual_seed(1)
    model = _make_model()
    adapter = PyTorchAdapter()

    x = np.random.randn(40, 4).astype(np.float32)
    y = np.random.randn(40, 2).astype(np.float32)

    model_a = _make_model()
    adapter.train(model_a, (x, y), epochs=2, batch_size=8, lr=0.1, seed=0)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
    model_b = _make_model()
    adapter.train(model_b, loader, epochs=2, batch_size=8, lr=0.1, seed=0)

    # Same starting weights + same data -> same result (deterministic).
    wa = adapter.get_weights(model_a)
    wb = adapter.get_weights(model_b)
    for a, b in zip(wa, wb):
        np.testing.assert_allclose(a, b)


def test_configurable_optimizer_and_criterion():
    torch.manual_seed(1)
    model = _make_model()
    adapter = PyTorchAdapter()
    before = adapter.get_weights(model)
    x = torch.randn(32, 4)
    y = torch.randint(0, 2, (32,))
    adapter.train(
        model,
        (x, y),
        epochs=3,
        batch_size=16,
        lr=0.05,
        optimizer=torch.optim.Adam,
        optimizer_kwargs={"weight_decay": 1e-4},
        criterion=nn.CrossEntropyLoss(),
    )
    after = adapter.get_weights(model)
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_num_samples_from_pairs_and_loader():
    adapter = PyTorchAdapter()
    assert adapter.num_samples((np.zeros((17, 4)), np.zeros(17))) == 17
    ds = TensorDataset(torch.zeros(9, 4), torch.zeros(9))
    assert adapter.num_samples(DataLoader(ds, batch_size=4)) == 9
    assert adapter.num_samples(ds) == 9


def test_invalid_train_data_raises():
    adapter = PyTorchAdapter()
    with pytest.raises(ValueError, match="train_data"):
        adapter.train(_make_model(), "not data", epochs=1)


def test_unsupported_non_model_rejected():
    class Dummy:
        pass

    with pytest.raises(UnsupportedModelError, match="Only PyTorch and scikit-learn"):
        create_adapter(Dummy())


def test_tensorflow_keras_model_rejected():
    fake_keras = type("Sequential", (), {"__module__": "tensorflow.keras.models"})
    with pytest.raises(UnsupportedModelError, match="TensorFlow/Keras"):
        create_adapter(fake_keras())

    fake_keras2 = type("Model", (), {"__module__": "keras.models"})
    with pytest.raises(UnsupportedModelError, match="TensorFlow/Keras"):
        create_adapter(fake_keras2())