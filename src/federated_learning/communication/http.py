"""Optional HTTP communication layer (FastAPI server + HTTP client).

Everything in this module is optional: the federated learning core works
fully in-process without it. Install the extra with::

    uv add federated-learning[server]

The client-side pieces are thin adapters around :class:`FederatedClient`; the
server-side piece is a thin adapter around :class:`FederatedServer`.
"""

from __future__ import annotations

import importlib
from typing import Any

from federated_learning.client import ClientUpdate, FederatedClient
from federated_learning.communication.serialization import (
    deserialize_weights,
    serialize_weights,
    update_from_dict,
    update_to_dict,
)
from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidWeightsError,
)
from federated_learning.server import FederatedServer

try:  # pydantic ships with fastapi
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    BaseModel = None  # type: ignore[assignment]


if BaseModel is not None:  # pragma: no branch

    class WeightSpec(BaseModel):
        dtype: str
        shape: list[int]
        data: str

    class UpdateIn(BaseModel):
        client_id: str
        num_samples: int
        weights: list[WeightSpec]
        metrics: dict[str, float] | None = None


def _require(package: str) -> Any:
    try:
        return importlib.import_module(package)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The HTTP communication layer requires the 'server' extra. "
            "Install it with: uv add federated-learning[server]"
        ) from exc


class HTTPClientChannel:
    """Client-side HTTP channel implementing :class:`ServerChannel`.

    Talks to a :class:`FastAPIServer` using ``httpx``. For testing, an
    arbitrary client-like object exposing ``get``/``post``/``close`` (e.g. a
    FastAPI ``TestClient``) may be injected instead.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        headers: dict[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if client is None:
            httpx = _require("httpx")
            self._client = httpx.Client(
                base_url=self.base_url, timeout=timeout, headers=headers or {}
            )
        else:
            self._client = client

    def get_global_weights(self) -> dict[str, Any]:
        response = self._client.get(f"{self.base_url}/global-weights")
        response.raise_for_status()
        return response.json()

    def get_health(self) -> bool:
        """Probe ``GET /health``; ``True`` once the server is reachable."""
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:  # connection refused while the server boots
            return False

    def send_update(self, update_payload: dict[str, Any]) -> None:
        response = self._client.post(f"{self.base_url}/updates", json=update_payload)
        response.raise_for_status()

    def aggregate(self) -> dict[str, Any]:
        """Tell the server to aggregate stored updates (POST /aggregate)."""
        response = self._client.post(f"{self.base_url}/aggregate")
        response.raise_for_status()
        return response.json()

    def get_metrics(self) -> list[dict]:
        """Fetch the server's per-round metrics history (GET /metrics)."""
        response = self._client.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.json()["history"]

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "HTTPClientChannel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class HTTPClient:
    """A remote federated client.

    Wraps a local :class:`FederatedClient` and performs one federated round
    over HTTP: fetch global weights, adopt them, train locally, send the
    update back.
    """

    def __init__(
        self,
        client: FederatedClient,
        base_url: str | None = None,
        *,
        channel: HTTPClientChannel | None = None,
    ) -> None:
        if channel is None:
            if base_url is None:
                raise ValueError("Either base_url or channel must be provided.")
            channel = HTTPClientChannel(base_url)
        self._client = client
        self._channel = channel

    @property
    def num_samples(self) -> int:
        return self._client.num_samples

    @property
    def client_id(self) -> str:
        return self._client.client_id

    def run_round(self) -> ClientUpdate:
        """Fetch global weights, train locally, and push the update."""
        payload = self._channel.get_global_weights()
        weights_payload = payload.get("weights")
        if weights_payload is not None:
            self._client.set_weights(deserialize_weights(weights_payload))
        self._client.train()
        update = self._client.get_update()
        self._channel.send_update(update_to_dict(update))
        return update


class FastAPIServer:
    """Expose a :class:`FederatedServer` over HTTP using FastAPI + uvicorn.

    Endpoints
    ---------
    ``GET /health``
        Health check.
    ``GET /global-weights``
        Return the current global weights (serialized).
    ``POST /updates``
        Receive a client update ``{client_id, num_samples, weights}``.
    ``POST /aggregate``
        Aggregate received updates and update the global model.
    """

    def __init__(self, server: FederatedServer) -> None:
        self._server = server
        self._app: Any = None

    def app(self) -> Any:
        """Build (once) and return the FastAPI application."""
        if self._app is not None:
            return self._app

        fastapi = _require("fastapi")
        FastAPI = fastapi.FastAPI
        HTTPException = fastapi.HTTPException

        server = self._server
        app = FastAPI(title="federated-learning")

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/global-weights")
        def global_weights() -> dict[str, Any]:
            weights = server.get_global_weights()
            return {
                "round": server.round,
                "weights": serialize_weights(weights) if weights is not None else None,
            }

        @app.get("/metrics")
        def metrics() -> dict[str, Any]:
            """Per-round metrics: global model + per-client metrics."""
            return {"history": server.get_metrics_history()}

        @app.post("/updates")
        def receive_update(payload: UpdateIn) -> dict[str, Any]:
            try:
                update = update_from_dict(payload.model_dump())
                server.receive_update(update)
            except (InvalidClientUpdateError, InvalidWeightsError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "status": "ok",
                "client_id": update.client_id,
                "round": server.round,
            }

        @app.post("/aggregate")
        def aggregate() -> dict[str, Any]:
            try:
                weights = server.aggregate_and_update()
            except (InvalidClientUpdateError, InvalidWeightsError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "round": server.round,
                "weights": serialize_weights(weights),
            }

        self._app = app
        return app

    def run(self, host: str = "127.0.0.1", port: int = 8000, **kwargs: Any) -> None:
        """Serve the FastAPI app with uvicorn (blocking)."""
        uvicorn = _require("uvicorn")
        uvicorn.run(self.app(), host=host, port=port, **kwargs)