"""Standalone federated server, exposed over HTTP.

This is one half of the real-world deployment: the server process that owns
the global model and aggregates client updates. The other half is
``client.py`` (run one instance per client).

Usage::

    uv run python examples/http/server.py --port 8000

The server starts a FastAPI + uvicorn service and blocks. Clients connect to
it over HTTP; aggregation is triggered with::

    curl -X POST http://127.0.0.1:8000/aggregate

and per-round metrics can be inspected with::

    curl http://127.0.0.1:8000/metrics
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from federated_learning import FederatedServer
from federated_learning.communication import FastAPIServer

from common import ClassificationMLP, make_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone federated server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    torch.manual_seed(0)

    eval_x, eval_y = make_data(200, seed=99)
    server = FederatedServer(
        model=ClassificationMLP(),
        metrics=["accuracy"],
        evaluation_data=(eval_x, eval_y),
    )

    before = server.evaluate().get("accuracy", float("nan"))
    print(f"server up: global accuracy before federated training = {before:.3f}")
    print(f"listening on http://{args.host}:{args.port}")
    print("press Ctrl+C to stop")

    api = FastAPIServer(server)
    api.run(host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()