"""Federated Learning with scikit-learn models using FedAvg.

Supported estimators are those whose learned parameters (``coef_``/
``intercept_`` or ``cluster_centers_``) can be meaningfully averaged. This
example uses a linear regression over three clients with disjoint data.

Linear models whose training is closed-form (e.g. LinearRegression) refit on
their local data every round. SGD-based estimators (SGDClassifier,
SGDRegressor, ...) train incrementally with mini-batches starting from the
global weights, which is genuine federated learning.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedTrainer,
)


def main() -> None:
    n_clients = 3
    rounds = 3

    X, y = make_regression(
        n_samples=300, n_features=5, noise=0.5, random_state=0
    )
    splits = np.array_split(np.arange(len(X)), n_clients)

    server = FederatedServer(model=LinearRegression())

    clients = [
        FederatedClient(
            client_id=f"client_{i}",
            model=LinearRegression(),
            train_data=(X[idx], y[idx]),
        )
        for i, idx in enumerate(splits)
    ]

    print(f"n_clients={n_clients}  rounds={rounds}")
    for i, c in enumerate(clients):
        print(f"  client_{i}: {c.num_samples} local samples")

    def log_round(round_index: int, weights: list[np.ndarray]) -> None:
        # LinearRegression is closed-form: every client refits the exact same
        # local optimum each round, so the global average converges after the
        # first round. SGD-based estimators train incrementally instead and
        # improve round over round.
        print(f"round {round_index + 1}: global coef_={np.round(weights[0], 3)} "
              f"intercept_={np.round(weights[1], 3)}")

    trainer = FederatedTrainer(
        server=server,
        clients=clients,
        rounds=rounds,
        on_round=log_round,
    )
    trainer.fit()

    final_model = trainer.get_model()
    score = r2_score(y, final_model.predict(X))
    print(f"final global model R^2 on the full dataset: {score:.3f}")
    print(f"coef_={np.round(final_model.coef_, 3)}  "
          f"intercept_={np.round(final_model.intercept_, 3)}")


if __name__ == "__main__":
    main()