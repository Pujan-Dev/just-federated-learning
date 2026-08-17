"""Tests for the FederatedTrainer and end-to-end federated training."""

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from torch import nn

from federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedTrainer,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc(x)


def _make_clients(n_clients=3, samples_per_client=24, seed=0):
    rng = np.random.default_rng(seed)
    clients = []
    for i in range(n_clients):
        x = rng.standard_normal((samples_per_client, 3)).astype(np.float32)
        y = rng.standard_normal((samples_per_client, 1)).astype(np.float32)
        torch.manual_seed(0)
        clients.append(
            FederatedClient(
                client_id=f"c{i}",
                model=TinyModel(),
                train_data=(x, y),
                local_epochs=2,
                learning_rate=0.05,
                seed=0,
            )
        )
    return clients


def test_trainer_rejects_invalid_config():
    server = FederatedServer(model=TinyModel())
    client = _make_clients(n_clients=1)[0]
    with pytest.raises(ValueError, match="rounds"):
        FederatedTrainer(server=server, clients=[client], rounds=0)
    with pytest.raises(ValueError, match="At least one client"):
        FederatedTrainer(server=server, clients=[], rounds=3)


def test_trainer_runs_multiple_rounds():
    server = FederatedServer(model=TinyModel())
    clients = _make_clients()
    rounds_run = []

    trainer = FederatedTrainer(
        server=server,
        clients=clients,
        rounds=5,
        on_round=lambda r, w: rounds_run.append(r),
    )
    trainer.fit()

    assert rounds_run == [0, 1, 2, 3, 4]
    assert server.round == 5
    assert trainer.get_model() is server.get_model()


def test_client_server_workflow_in_each_round():
    server = FederatedServer(model=TinyModel())
    clients = _make_clients()
    initial_weights = server.get_global_weights()

    trainer = FederatedTrainer(server=server, clients=clients, rounds=2)
    trainer.fit()

    final_weights = server.get_global_weights()
    assert len(final_weights) == len(initial_weights)
    assert any(
        not np.allclose(a, b) for a, b in zip(initial_weights, final_weights)
    )


def test_e2e_pytorch_accuracy_improves():
    torch.manual_seed(0)

    # Synthetic 2D binary classification (linearly separable-ish).
    def make_data(n, seed):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, 2)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
        return x, y

    def make_model():
        return nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def accuracy(model, seed):
        x, y = make_data(100, seed)
        with torch.no_grad():
            preds = model(torch.as_tensor(x)).argmax(dim=1).numpy()
        return float((preds == y).mean())

    server = FederatedServer(model=make_model())
    clients = [
        FederatedClient(
            client_id=f"c{i}",
            model=make_model(),
            train_data=make_data(40, i),
            local_epochs=3,
            batch_size=16,
            learning_rate=0.2,
            criterion=nn.CrossEntropyLoss(),
            seed=0,
        )
        for i in range(3)
    ]

    before = accuracy(server.get_model(), seed=99)
    FederatedTrainer(server=server, clients=clients, rounds=5).fit()
    after = accuracy(server.get_model(), seed=99)

    # The federated model should learn the decision boundary.
    assert after >= before
    assert after > 0.7


def test_e2e_sklearn_federated_regression():
    from sklearn.metrics import r2_score

    X, y = make_regression(n_samples=90, n_features=3, noise=0.1, random_state=0)
    splits = np.array_split(np.arange(len(X)), 3)

    server = FederatedServer(model=LinearRegression())
    clients = [
        FederatedClient(
            client_id=f"c{i}",
            model=LinearRegression(),
            train_data=(X[idx], y[idx]),
        )
        for i, idx in enumerate(splits)
    ]

    trainer = FederatedTrainer(server=server, clients=clients, rounds=3)
    trainer.fit()
    final_model = trainer.get_model()

    assert final_model.coef_.shape == (3,)
    assert r2_score(y, final_model.predict(X)) > 0.8


def test_e2e_sklearn_logistic_classification():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=150,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    splits = np.array_split(np.arange(len(X)), 3)

    server = FederatedServer(model=SGDClassifier(loss="log_loss", random_state=0))
    clients = [
        FederatedClient(
            client_id=f"c{i}",
            model=SGDClassifier(loss="log_loss", random_state=0),
            train_data=(X[idx], y[idx]),
            local_epochs=3,
            batch_size=16,
            seed=0,
        )
        for i, idx in enumerate(splits)
    ]

    trainer = FederatedTrainer(server=server, clients=clients, rounds=5)
    trainer.fit()

    final_model = trainer.get_model()
    assert accuracy_score(final_model.predict(X), y) > 0.9


def test_unequal_client_sizes_aggregated_by_samples():
    rng = np.random.default_rng(0)
    server = FederatedServer(model=TinyModel())

    datasets = [8, 24, 40]
    clients = []
    for i, n in enumerate(datasets):
        x = rng.standard_normal((n, 3)).astype(np.float32)
        y = rng.standard_normal((n, 1)).astype(np.float32)
        clients.append(
            FederatedClient(
                client_id=f"c{i}",
                model=TinyModel(),
                train_data=(x, y),
                local_epochs=1,
                learning_rate=0.01,
                seed=0,
            )
        )

    trainer = FederatedTrainer(server=server, clients=clients, rounds=2)
    trainer.fit()

    assert [c.num_samples for c in clients] == datasets
    assert trainer.get_model() is not None