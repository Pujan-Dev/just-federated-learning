"""PyTorch model adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from federated_learning.exceptions import InvalidSampleCountError
from federated_learning.models.base import ModelAdapter, Weights

try:  # pragma: no cover - import guard for optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = Dataset = TensorDataset = None  # type: ignore[assignment]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch ``nn.Module`` models.

    Weights are represented as an ordered list of numpy arrays obtained from
    ``model.state_dict()`` (in registration order). The order is preserved so
    that :meth:`set_weights` can re-apply the exact same structure, including
    original dtypes and the model's current device.
    """

    framework = "pytorch"

    @classmethod
    def is_supported(cls, model: Any) -> bool:
        return torch is not None and isinstance(model, nn.Module)

    def get_weights(self, model: nn.Module) -> Weights:
        # Copy: the returned numpy arrays must not alias parameter memory,
        # otherwise in-place optimizer steps would silently mutate previously
        # captured "snapshots" of the weights.
        return [v.detach().cpu().numpy().copy() for v in model.state_dict().values()]

    def set_weights(self, model: nn.Module, weights: Weights) -> nn.Module:
        arrays = self._weights_to_arrays(weights)
        current = model.state_dict()
        self._check_shapes(list(current.values()), arrays)
        device = next(model.parameters()).device
        new_state: dict[str, Any] = {}
        for (name, param), arr in zip(current.items(), arrays):
            # torch.tensor always copies, decoupling the parameter from the
            # input numpy buffer.
            new_state[name] = torch.tensor(
                arr, dtype=param.dtype, device=device
            )
        model.load_state_dict(new_state)
        return model

    def validate_weights(self, model: nn.Module, weights: Weights) -> None:
        arrays = self._weights_to_arrays(weights)
        current = model.state_dict()
        self._check_shapes(list(current.values()), arrays)

    def predict(self, model: nn.Module, x: Any) -> np.ndarray:
        """Run the model in evaluation mode and return predictions as numpy."""
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                x_t = x if isinstance(x, torch.Tensor) else torch.as_tensor(
                    x, dtype=torch.float32
                )
                out = model(x_t.to(device))
            return out.detach().cpu().numpy()
        finally:
            model.train(was_training)

    def train(
        self,
        model: nn.Module,
        train_data: Any,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.01,
        device: str | torch.device | None = None,
        optimizer: type[torch.optim.Optimizer] | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        criterion: nn.Module | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> nn.Module:
        """Run local (SGD-style) training on ``train_data``.

        ``train_data`` may be a ``DataLoader``, a ``Dataset``, or an ``(X, y)``
        pair of array-like objects.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("PyTorch is required for the PyTorchAdapter.")

        if seed is not None:
            torch.manual_seed(seed)

        loader = self._build_loader(train_data, batch_size, shuffle, seed)
        target_device = _resolve_device(device)
        model = model.to(target_device)

        opt_cls = optimizer if optimizer is not None else torch.optim.SGD
        opt_kwargs = dict(optimizer_kwargs or {})
        opt_kwargs.setdefault("lr", lr)
        opt = opt_cls(model.parameters(), **opt_kwargs)

        loss_fn = criterion if criterion is not None else nn.MSELoss()

        model.train()
        for _ in range(epochs):
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(target_device)
                y_batch = y_batch.to(target_device)
                opt.zero_grad()
                loss = loss_fn(model(x_batch), y_batch)
                loss.backward()
                opt.step()
        return model

    def num_samples(self, train_data: Any) -> int:
        if isinstance(train_data, DataLoader):
            dataset = getattr(train_data, "dataset", None)
            if dataset is not None and hasattr(dataset, "__len__"):
                return len(dataset)
            raise InvalidSampleCountError(
                "Cannot determine the sample count for this DataLoader; "
                "pass an explicit num_samples."
            )
        if isinstance(train_data, Dataset):
            return len(train_data)
        if isinstance(train_data, (tuple, list)) and len(train_data) == 2:
            return len(train_data[0])
        raise InvalidSampleCountError(
            "Cannot determine the sample count for the provided train_data; "
            "pass an explicit num_samples."
        )

    @staticmethod
    def _build_loader(
        train_data: Any,
        batch_size: int,
        shuffle: bool,
        seed: int | None,
    ) -> DataLoader:
        if isinstance(train_data, DataLoader):
            return train_data
        if isinstance(train_data, Dataset):
            dataset = train_data
        elif isinstance(train_data, (tuple, list)) and len(train_data) == 2:
            x, y = train_data
            if isinstance(x, torch.Tensor):
                x_t = x
            else:
                x_t = torch.as_tensor(x, dtype=torch.float32)
            if isinstance(y, torch.Tensor):
                y_t = y
            elif np.asarray(y).dtype.kind in "iu":
                y_t = torch.as_tensor(y, dtype=torch.long)
            else:
                y_t = torch.as_tensor(y, dtype=torch.float32)
            dataset = TensorDataset(x_t, y_t)
        else:
            raise ValueError(
                "train_data must be a DataLoader, a Dataset, or an (X, y) pair."
            )
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
        )