"""End-to-end federated learning over real HTTP.

This demonstrates the full "real world" round trip in a single command:

* a FastAPI/uvicorn server is started in a background thread,
* several remote clients connect to it over HTTP (the same transport a
  production deployment would use between separate machines),
* each round every client fetches the global weights, trains locally on its
  private data, and pushes its update back,
* the server aggregates with sample-count weighted FedAvg,
* the server records the global model accuracy (on its own evaluation data)
  plus each client's local accuracy, and we fetch those metrics back over HTTP.

Usage::

    uv run python examples/http/run.py --rounds 5 --port 8000

The same setup can be run as genuinely separate processes::

    uv run python examples/http/server.py --port 8000            # terminal 1
    uv run python examples/http/client.py --client-id client_0   # terminals 2-4
    curl -X POST http://127.0.0.1:8000/aggregate                 # each round
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import torch
from torch import nn

from federated_learning import FederatedClient, FederatedServer
from federated_learning.communication import (
    FastAPIServer,
    HTTPClient,
    HTTPClientChannel,
)

from common import ClassificationMLP, make_data


def _wait_until_ready(channel: HTTPClientChannel, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if channel.get_health():
            return
        time.sleep(0.1)
    raise TimeoutError("server did not become ready in time")


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated learning over HTTP")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n-clients", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(0)

    # --- server process ------------------------------------------------------
    eval_x, eval_y = make_data(200, seed=99)
    server = FederatedServer(
        model=ClassificationMLP(),
        metrics=["accuracy"],
        evaluation_data=(eval_x, eval_y),
    )
    api = FastAPIServer(server)
    thread = threading.Thread(
        target=lambda: __import__("uvicorn").run(
            api.app(), host=args.host, port=args.port, log_level="warning"
        ),
        daemon=True,
    )
    thread.start()

    channel = HTTPClientChannel(f"http://{args.host}:{args.port}")
    _wait_until_ready(channel)
    print(f"server ready on http://{args.host}:{args.port}")

    # --- client processes ----------------------------------------------------
    clients = [
        HTTPClient(
            FederatedClient(
                client_id=f"client_{i}",
                model=ClassificationMLP(),
                train_data=make_data(args.samples, seed=i),
                local_epochs=args.local_epochs,
                batch_size=16,
                learning_rate=0.2,
                criterion=nn.CrossEntropyLoss(),
                seed=0,
                metrics=["accuracy"],
            ),
            channel=channel,
        )
        for i in range(args.n_clients)
    ]
    print(f"{args.n_clients} remote clients connected over HTTP")

    # --- federated loop ------------------------------------------------------
    for round_index in range(1, args.rounds + 1):
        for client in clients:
            client.run_round()

        channel.aggregate()
        entry = channel.get_metrics()[-1]

        global_acc = entry["global"].get("accuracy")
        client_accs = {
            client_id: values.get("accuracy")
            for client_id, values in entry["clients"].items()
        }
        client_accs_str = ", ".join(
            f"{cid}={acc:.3f}" for cid, acc in client_accs.items()
        )
        print(
            f"round {round_index:>2}: global accuracy={global_acc:.3f}  "
            f"per-client accuracy=({client_accs_str})"
        )

    channel.close()


if __name__ == "__main__":
    main()