"""Transport-agnostic communication abstractions.

The federated learning core never touches networking. This module defines the
interface a client-side channel must implement so that alternative transports
(e.g. HTTP via FastAPI, or a future gRPC layer) can be plugged in without
touching the core.
"""

from __future__ import annotations

from typing import Any, Protocol


class ServerChannel(Protocol):
    """Client-side view of a federated server."""

    def get_global_weights(self) -> dict[str, Any]:
        """Return the server payload containing the global weights."""
        ...

    def send_update(self, update_payload: dict[str, Any]) -> None:
        """Send a serialized client update to the server."""
        ...