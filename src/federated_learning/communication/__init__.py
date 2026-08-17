"""Optional communication layer between clients and server.

The HTTP transport (``FastAPIServer``, ``HTTPClient``, ``HTTPClientChannel``)
requires the optional ``server`` extra and is exposed lazily so that the core
package works without it.
"""

from __future__ import annotations

from typing import Any

from federated_learning.communication.base import ServerChannel
from federated_learning.communication.serialization import (
    deserialize_weights,
    serialize_weights,
    update_from_dict,
    update_to_dict,
)

_HTTP_NAMES = ("FastAPIServer", "HTTPClient", "HTTPClientChannel")

__all__ = [
    "ServerChannel",
    "serialize_weights",
    "deserialize_weights",
    "update_to_dict",
    "update_from_dict",
    *_HTTP_NAMES,
]


def __getattr__(name: str) -> Any:
    if name in _HTTP_NAMES:
        from federated_learning.communication.http import (  # noqa: PLC0415
            FastAPIServer,
            HTTPClient,
            HTTPClientChannel,
        )

        return {
            "FastAPIServer": FastAPIServer,
            "HTTPClient": HTTPClient,
            "HTTPClientChannel": HTTPClientChannel,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")