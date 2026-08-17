# federated-learning

A production-ready, framework-agnostic **Federated Learning** package built on
Federated Averaging (FedAvg). Bring your own model and data — the package
handles the client/server coordination, weighted aggregation and the training
loop.

It is a **reusable Python library** (not an application). The core runs fully
in-process; networking (FastAPI/HTTP) is an optional extra.

## Supported frameworks

* **PyTorch** — any `torch.nn.Module`.
* **scikit-learn** — a curated set of estimators whose parameters can be
  meaningfully averaged (see [scikit-learn support](#scikit-learn-support)).

### TensorFlow and Keras are NOT supported

TensorFlow and Keras are **explicitly not supported**. They are not
dependencies, there are no adapters, no examples and no automatic detection.
If a TensorFlow/Keras model is passed, a `UnsupportedModelError` is raised:

```text
UnsupportedModelError: Only PyTorch and scikit-learn models are supported.
TensorFlow/Keras models are not supported.
```

## Installation

This project is managed with [uv](https://docs.astral.sh/uv/).

Add it to another Python project:

```bash
uv add federated-learning
```

Framework extras:

```bash
uv add "federated-learning[pytorch]"   # PyTorch models
uv add "federated-learning[sklearn]"   # scikit-learn models
uv add "federated-learning[server]"    # optional FastAPI/HTTP communication
```

For development of this repository:

```bash
uv sync
```

Run the tests:

```bash
uv run pytest
```

Run the examples:

```bash
uv run python examples/pytorch_federated.py
uv run python examples/sklearn_federated.py
uv run python examples/http/run.py    # real-world client/server over HTTP
```

## How it works

```
                Global Model
                     │
                     ▼
              ┌─────────────┐
              │    Server   │
              └──────┬──────┘
                     │
              Global Weights
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Client 1       Client 2      Client 3
       │             │             │
 Local Training  Local Training  Local Training
       │             │             │
 Updated Weights Updated Weights Updated Weights
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                   FedAvg
                     │
                     ▼
              New Global Model
```

1. The server holds the global model.
2. Global weights are distributed to every client.
3. Each client applies the global weights to its local model.
4. Each client trains locally on its **private** dataset.
5. Each client extracts its updated weights.
6. Each client sends `(client_id, weights, num_samples)` to the server.
7. The server performs sample-count weighted **FedAvg**.
8. The server applies the aggregated weights to the global model.
9. The new global weights are distributed again.
10. Repeat for the configured number of rounds.

## Metrics

Track model quality per round and per client without extra plumbing. Metrics
are framework agnostic: they operate on `(y_true, y_pred)` arrays produced by
either adapter.

Built-in metrics:

| Name | Type | Notes |
|------|------|-------|
| `accuracy` | classification | auto-argmax on 2D score/logit matrices |
| `precision`, `recall`, `f1` | classification | macro-averaged |
| `mse`, `mae`, `rmse`, `r2` | regression | |

Custom metrics are just callables `(y_true, y_pred) -> float`.

Client-side: configure a client with `metrics` (and optionally
`evaluation_data`) and its `get_update()` will include a `metrics` dict. The
server records those per-client metrics each round:

```python
client = FederatedClient(
    client_id="c1",
    model=model,
    train_data=(x, y),
    metrics=["accuracy", "f1"],   # evaluated on the client's local data
)
```

Server-side: configure the server with `metrics` and `evaluation_data` to
evaluate the **global** model after every aggregation:

```python
server = FederatedServer(
    model=model,
    metrics=["accuracy"],
    evaluation_data=(eval_x, eval_y),
)
```

After each `aggregate_and_update()` the server appends an entry to
`server.metrics_history` (also exposed as `trainer.metrics_history`):

```python
[
    {
        "round": 1,
        "global": {"accuracy": 0.955},
        "clients": {"client_0": {"accuracy": 0.975}, "client_1": {"accuracy": 0.85}},
    },
    ...
]
```

`FederatedClient.evaluate()` and `FederatedServer.evaluate()` return the raw
`{name: value}` metrics for ad-hoc checks.

## FedAvg

Aggregation is **weighted by the number of samples** per client:

```
        Σ(n_i × W_i)
W_global = ------------
           Σ(n_i)
```

where `W_i` are the weights of client `i` and `n_i` is its number of samples.
Simple averaging is only equivalent when all clients have equal sample counts.
All updates are validated before aggregation (weight structure, shapes,
dtypes, sample counts, non-empty updates).

```python
from federated_learning import FedAvg

global_weights = FedAvg().aggregate(updates)
```

## Basic PyTorch usage

```python
import torch
from torch import nn
from federated_learning import FederatedClient, FederatedServer, FederatedTrainer

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

# Each client owns a private dataset.
clients = [
    FederatedClient(
        client_id=f"client_{i}",
        model=MyModel(),
        train_data=(x_local[i], y_local[i]),   # your data
        local_epochs=3,
        batch_size=32,
        learning_rate=0.1,
        criterion=nn.MSELoss(),
        optimizer=torch.optim.SGD,
    )
    for i in range(3)
]

server = FederatedServer(model=MyModel())
trainer = FederatedTrainer(server=server, clients=clients, rounds=10)
trainer.fit()

final_model = trainer.get_model()
```

`train_data` may be a `DataLoader`, a `Dataset`, or an `(X, y)` pair of
array-like objects. For GPU training pass `device="cuda"` to each client.

## Basic scikit-learn usage

```python
from sklearn.linear_model import LinearRegression
from federated_learning import FederatedClient, FederatedServer, FederatedTrainer

clients = [
    FederatedClient(
        client_id=f"client_{i}",
        model=LinearRegression(),
        train_data=(X_local[i], y_local[i]),
    )
    for i in range(3)
]

server = FederatedServer(model=LinearRegression())
trainer = FederatedTrainer(server=server, clients=clients, rounds=3)
trainer.fit()

final_model = trainer.get_model()   # a fitted LinearRegression
```

## Local training (no server)

A `FederatedClient` works standalone — training locally does not require a
server:

```python
client = FederatedClient(
    client_id="c1",
    model=MyModel(),
    train_data=(x, y),
    local_epochs=5,
)
client.train()                 # train on the local dataset
update = client.get_update()   # ClientUpdate(client_id, weights, num_samples)
```

## Client and server API

```python
from federated_learning import FederatedClient, FederatedServer

client = FederatedClient(client_id="c1", model=model, train_data=data)
client.train()
client.set_weights(global_weights)
update = client.get_update()   # -> ClientUpdate(client_id, weights, num_samples)

server = FederatedServer(model=model)
server.get_global_weights()     # weights to send out
server.receive_update(update)   # validate + store
server.aggregate_and_update()   # FedAvg -> apply to global model, round += 1
```

## Optional HTTP communication

The core is transport-agnostic. An optional FastAPI/HTTP layer is available in
the `server` extra. It is fully decoupled from the core.

Server process:

```python
from federated_learning import FederatedServer
from federated_learning.communication.http import FastAPIServer

server = FederatedServer(model=model)
FastAPIServer(server).run(host="127.0.0.1", port=8000)
```

Endpoints: `GET /health`, `GET /global-weights`, `GET /metrics`,
`POST /updates`, `POST /aggregate`.

Remote client (e.g. on a different machine):

```python
from federated_learning import FederatedClient
from federated_learning.communication.http import HTTPClient, HTTPClientChannel

channel = HTTPClientChannel("http://127.0.0.1:8000")
remote = HTTPClient(client, channel=channel)
remote.run_round()   # fetch global weights, train locally, send update
```

`GET /metrics` returns the server's per-round metrics history (global +
per-client metrics), so an operator can monitor training remotely.

Weights are transferred as an explicit JSON-safe schema (dtype, shape,
base64-encoded bytes) — **not** pickled Python objects. Incoming updates are
validated before aggregation.

### Real-world example over HTTP

`examples/http/` shows the full deployment topology over real HTTP:

```bash
uv run python examples/http/run.py   # server + remote clients in one command
```

or run each piece as a genuinely separate process:

```bash
uv run python examples/http/server.py --port 8765            # terminal 1
uv run python examples/http/client.py --client-id client_0   # terminals 2..4
curl -X POST http://127.0.0.1:8765/aggregate                 # each round
curl http://127.0.0.1:8765/metrics                           # monitor
```

Each round prints the global model accuracy (server-side, on its evaluation
data) and every client's local accuracy (sent along with its update).

## Running tests

```bash
uv run pytest
```

The suite covers FedAvg mathematics, both adapters, the client, the server,
the trainer, serialization, the HTTP layer, and full end-to-end federated
rounds.

## Architecture

```
ModelAdapter
├── PyTorchAdapter        # torch.nn.Module via state_dict
└── SklearnAdapter        # supported sklearn estimators

FederatedClient  ──┐
FederatedServer  ──┼──> FedAvg (weighted aggregation)
FederatedTrainer ──┘
        │
        └── optional: FastAPIServer / HTTPClient (communication layer)
```

The core never imports `torch`, `sklearn` or `fastapi`; framework-specific
adapters are resolved lazily. Install only the extras you need.

## scikit-learn support

Not every scikit-learn estimator can be safely aggregated with FedAvg.
Support is limited to estimators whose learned state is numeric model
**parameters** that can be averaged in a mathematically sound way.

Supported:

| Estimator | Parameters averaged |
|-----------|---------------------|
| `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` | `coef_`, `intercept_` |
| `LogisticRegression`, `RidgeClassifier` | `coef_`, `intercept_` |
| `SGDRegressor`, `SGDClassifier` | `coef_`, `intercept_` (trained incrementally via `partial_fit`) |
| `PassiveAggressiveClassifier/Regressor`, `Perceptron` | `coef_`, `intercept_` |
| `LinearSVC`, `LinearSVR` | `coef_`, `intercept_` |
| `KMeans` | `cluster_centers_` |

Any other estimator raises `UnsupportedModelError`. Notable exclusions:

* **Tree-based models** (`DecisionTreeClassifier`, `RandomForestClassifier`,
  `GradientBoosting*`, ...) — structure cannot be averaged.
* **Nearest neighbours** (`KNeighborsClassifier`, ...) — store training data,
  not averageable parameters.
* **Naive Bayes / other probabilistic estimators** (`GaussianNB`, ...).

Limitations:

* Estimators that store non-averageable state (trees, neighbours, ...) are
  rejected rather than naively serialized.
* Closed-form linear models (e.g. `LinearRegression`) fully refit on local
  data each round, so FedAvg converges in a single round.
* SGD-based estimators are the recommended choice for genuine multi-round
  federated convergence: they train incrementally (mini-batch `partial_fit`)
  from the loaded global weights.
* For classifiers whose parameters are applied without a full refit (the
  server-side global model), the label space is reconstructed as integer
  classes `0..C-1` from the coefficient shape.
* All clients must use the same estimator architecture.

## Public API

```python
from federated_learning import (
    FederatedClient,
    FederatedServer,
    FederatedTrainer,
    ClientUpdate,
    FedAvg,
    ModelAdapter,
    PyTorchAdapter,
    SklearnAdapter,
    accuracy,
    precision,
    recall,
    f1,
    mse,
    mae,
    rmse,
    r2,
    UnsupportedModelError,
    InvalidWeightsError,
    WeightShapeMismatchError,
    InvalidClientUpdateError,
    InvalidSampleCountError,
)
```

## Design principles

* Simple, modular, reusable and type-safe.
* Framework-independent core (adapter-based).
* Transport-independent core (communication is optional).
* No database, Docker, Kubernetes, auth, dashboard, frontend or cloud tooling.
* No TensorFlow, Keras, XGBoost or other ML frameworks.

## License

MIT