"""Optional communication layer between clients and server."""

from federated_learning.communication.base import ServerChannel
from federated_learning.communication.serialization import (
    deserialize_weights,
    serialize_weights,
    update_from_dict,
    update_to_dict,
)

__all__ = [
    "ServerChannel",
    "serialize_weights",
    "deserialize_weights",
    "update_to_dict",
    "update_from_dict",
]