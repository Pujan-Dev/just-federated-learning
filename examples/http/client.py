"""Standalone federated client that talks to a server over HTTP.

This is the other half of the real-world deployment. Run one instance per
client, each with its own private local dataset::

    uv run python examples/http/client.py --client-id client_0
    uv run python examples/http/client.py --client-id client_1
    uv run python examples/http/client.py --client-id client_2

Each round the client fetches the current global weights from the server,
trains locally on its own data, and pushes the resulting update back over
HTTP. Its local accuracy is reported and sent along with the update. Between
rounds, trigger the server-side aggregation (see ``server.py``), then the
client will pick up the fresh global model on its next round.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch import nn

from federated_learning import FederatedClient
from federated_learning.communication import HTTPClient, HTTPClientChannel

from common import ClassificationMLP, make_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone federated client")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    seed = int(args.client_id.rsplit("_", 1)[-1]) if "_" in args.client_id else 0
    torch.manual_seed(0)

    client = FederatedClient(
        client_id=args.client_id,
        model=ClassificationMLP(),
        train_data=make_data(args.samples, seed=seed),
        local_epochs=args.local_epochs,
        batch_size=16,
        learning_rate=0.2,
        criterion=nn.CrossEntropyLoss(),
        seed=0,
        metrics=["accuracy"],
    )

    channel = HTTPClientChannel(args.server_url)
    remote = HTTPClient(client, channel=channel)

    for round_index in range(1, args.rounds + 1):
        update = remote.run_round()
        local_acc = update.metrics["accuracy"] if update.metrics else float("nan")
        print(
            f"[{args.client_id}] round {round_index}: pushed update "
            f"(n={update.num_samples}, local accuracy={local_acc:.3f})"
        )
        if round_index < args.rounds:
            time.sleep(args.sleep)

    channel.close()


if __name__ == "__main__":
    main()