"""Tests for the scikit-learn model adapter."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from federated_learning import SklearnAdapter, create_adapter
from federated_learning.exceptions import (
    InvalidWeightsError,
    UnsupportedModelError,
    WeightShapeMismatchError,
)


X_REG, y_REG = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=0)


def test_supported_estimator_detection():
    assert SklearnAdapter.is_supported(LinearRegression()) is True
    assert SklearnAdapter.is_supported(LogisticRegression()) is True
    assert SklearnAdapter.is_supported(KMeans(n_clusters=3)) is True
    assert SklearnAdapter.is_supported("not a model") is False


def test_unsupported_estimator_detection():
    for estimator in (
        RandomForestClassifier(),
        KNeighborsClassifier(),
        GaussianNB(),
    ):
        assert SklearnAdapter.is_supported(estimator) is False
        with pytest.raises(UnsupportedModelError, match="Only PyTorch and scikit-learn"):
            create_adapter(estimator)


def test_factory_creates_sklearn_adapter():
    assert isinstance(create_adapter(LinearRegression()), SklearnAdapter)


def test_get_weights_requires_fitted_estimator():
    with pytest.raises(InvalidWeightsError, match="not been fitted"):
        SklearnAdapter().get_weights(LinearRegression())


def test_weight_extraction_linear_model():
    model = LinearRegression().fit(X_REG, y_REG)
    weights = SklearnAdapter().get_weights(model)
    assert len(weights) == 2
    assert weights[0].shape == (3,)  # coef_
    assert weights[1].shape == ()  # intercept_ (scalar)


def test_weight_set_round_trip():
    model = LinearRegression().fit(X_REG, y_REG)
    adapter = SklearnAdapter()
    original = adapter.get_weights(model)
    scaled = [w * 2.0 for w in original]
    adapter.set_weights(model, scaled)
    restored = adapter.get_weights(model)
    for a, b in zip(scaled, restored):
        np.testing.assert_allclose(a, b)


def test_weight_shape_mismatch_on_set():
    model = LinearRegression().fit(X_REG, y_REG)
    adapter = SklearnAdapter()
    with pytest.raises(WeightShapeMismatchError, match="shape mismatch"):
        adapter.set_weights(model, [np.zeros((99,)), np.zeros(())])
    with pytest.raises(WeightShapeMismatchError, match="parameters"):
        adapter.set_weights(model, [np.zeros((3,))])


def test_local_training_changes_weights():
    model = SGDRegressor(random_state=0)
    adapter = SklearnAdapter()
    model.fit(X_REG[:30], y_REG[:30])
    before = adapter.get_weights(model)
    adapter.train(model, (X_REG[30:], y_REG[30:]))
    after = adapter.get_weights(model)
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_warm_start_continues_from_set_weights():
    model = SGDRegressor(random_state=0)
    adapter = SklearnAdapter()
    model.fit(X_REG, y_REG)
    baseline = adapter.get_weights(model)

    model2 = SGDRegressor(random_state=0)
    adapter.set_weights(model2, baseline)
    adapter.train(model2, (X_REG, y_REG))
    after = adapter.get_weights(model2)
    # Training continued from the loaded global weights; results must change.
    assert any(not np.allclose(b, a) for b, a in zip(baseline, after))


def test_kmeans_extraction_and_apply():
    X, _ = make_regression(n_samples=30, n_features=2, noise=0.1, random_state=0)
    model = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    adapter = SklearnAdapter()
    weights = adapter.get_weights(model)
    assert len(weights) == 1
    assert weights[0].shape == (3, 2)

    adapter.set_weights(model, [weights[0] * 1.0])
    restored = adapter.get_weights(model)
    np.testing.assert_allclose(restored[0], weights[0])


def test_num_samples():
    assert SklearnAdapter().num_samples((X_REG, y_REG)) == len(X_REG)


def test_invalid_train_data():
    with pytest.raises(ValueError, match="train_data"):
        SklearnAdapter().train(LinearRegression(), "nope")


def test_documented_supported_estimators():
    names = SklearnAdapter.supported_estimators()
    assert "LinearRegression" in names
    assert "LogisticRegression" in names
    assert "KMeans" in names