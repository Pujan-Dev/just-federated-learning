"""Tests for the FedAvg aggregation."""

import numpy as np
import pytest

from federated_learning import FedAvg, fedavg
from federated_learning.exceptions import (
    InvalidClientUpdateError,
    InvalidSampleCountError,
    InvalidWeightsError,
    WeightShapeMismatchError,
)


class FakeUpdate:
    """Duck-typed update: only needs weights + num_samples."""

    def __init__(self, weights, num_samples):
        self.weights = weights
        self.num_samples = num_samples


def test_weighted_averaging_simple():
    a = np.array([2.0, 4.0])
    b = np.array([6.0, 8.0])
    result = fedavg([FakeUpdate([a], 1), FakeUpdate([b], 1)])
    np.testing.assert_allclose(result[0], [4.0, 6.0])


def test_weighted_averaging_unequal_sample_counts():
    # W = (n1*W1 + n2*W2) / (n1 + n2)
    a = np.array([0.0, 10.0])
    b = np.array([10.0, 0.0])
    result = FedAvg().aggregate([FakeUpdate([a], 1), FakeUpdate([b], 3)])
    expected = (1 * a + 3 * b) / 4
    np.testing.assert_allclose(result[0], expected)


def test_multiple_clients_multiple_parameters():
    updates = [
        FakeUpdate([np.array([1.0, 1.0]), np.array([[2.0], [3.0]])], 5),
        FakeUpdate([np.array([2.0, 2.0]), np.array([[4.0], [6.0]])], 5),
        FakeUpdate([np.array([3.0, 3.0]), np.array([[6.0], [9.0]])], 10),
    ]
    result = fedavg(updates)
    expected_first = (5 * 1 + 5 * 2 + 10 * 3) / 20
    np.testing.assert_allclose(result[0], [expected_first, expected_first])
    np.testing.assert_allclose(result[1], [[4.5], [6.75]])


def test_equal_samples_reduces_to_simple_average():
    updates = [FakeUpdate([np.array([2.0])], 4) for _ in range(3)]
    result = fedavg(updates)
    np.testing.assert_allclose(result[0], [2.0])


def test_returns_float64():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([3.0], dtype=np.float32)
    result = fedavg([FakeUpdate([a], 1), FakeUpdate([b], 1)])
    assert result[0].dtype == np.float64


def test_rejects_empty_updates():
    with pytest.raises(InvalidClientUpdateError, match="at least one"):
        fedavg([])


def test_rejects_none_update():
    with pytest.raises(InvalidClientUpdateError):
        fedavg([None])


def test_rejects_object_without_attributes():
    with pytest.raises(InvalidClientUpdateError, match="weights"):
        fedavg([object()])


def test_rejects_empty_weights():
    with pytest.raises(InvalidWeightsError, match="empty"):
        fedavg([FakeUpdate([], 5)])


def test_rejects_non_numeric_weight():
    with pytest.raises(InvalidWeightsError, match="numeric"):
        fedavg([FakeUpdate([np.array("abc")], 5)])


def test_accepts_zerod_array_weights():
    # e.g. scikit-learn's scalar intercept_
    result = fedavg([FakeUpdate([np.array(2.0)], 1), FakeUpdate([np.array(6.0)], 1)])
    np.testing.assert_allclose(result[0], np.array(4.0))


def test_rejects_none_weight():
    with pytest.raises(InvalidWeightsError):
        fedavg([FakeUpdate([None], 5)])


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_rejects_non_positive_sample_counts(bad):
    with pytest.raises(InvalidSampleCountError, match="positive"):
        fedavg([FakeUpdate([np.array([1.0])], bad)])


def test_rejects_non_integer_sample_count():
    with pytest.raises(InvalidSampleCountError):
        fedavg([FakeUpdate([np.array([1.0])], 3.5)])


def test_rejects_shape_mismatch():
    with pytest.raises(WeightShapeMismatchError, match="shape mismatch"):
        fedavg(
            [
                FakeUpdate([np.array([1.0, 2.0])], 1),
                FakeUpdate([np.array([1.0])], 1),
            ]
        )


def test_rejects_parameter_count_mismatch():
    with pytest.raises(WeightShapeMismatchError, match="number of parameters"):
        fedavg(
            [
                FakeUpdate([np.array([1.0]), np.array([2.0])], 1),
                FakeUpdate([np.array([1.0])], 1),
            ]
        )


def test_rejects_mixed_dtype_objects():
    with pytest.raises((InvalidWeightsError, TypeError)):
        fedavg([FakeUpdate(["not-an-array"], 1)])