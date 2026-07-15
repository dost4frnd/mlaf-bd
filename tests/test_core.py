"""Torch-free unit checks for the analysis core. Run: pytest -q tests/"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mlafbd.features import compute_layer_features, NUM_FEATURES, FEATURE_NAMES
from mlafbd.wla import WeightedLayerAggregation


def test_feature_shape_and_names():
    vecs = [np.random.default_rng(i).dirichlet(np.ones(50)) for i in range(12)]
    F = compute_layer_features(vecs)
    assert F.shape == (12, NUM_FEATURES)
    assert FEATURE_NAMES[6] == "drift"


def test_drift_zero_for_identical_layers():
    a = np.random.default_rng(0).dirichlet(np.ones(64))
    F = compute_layer_features([a, a, a])
    assert F[0, 6] == 0.0                      # first layer
    assert np.allclose(F[1:, 6], 0.0)          # identical layers -> no drift


def test_concentration_is_max_and_entropy_positive():
    a = np.zeros(10); a[3] = 1.0
    F = compute_layer_features([a])
    assert abs(F[0, 2] - 1.0) < 1e-6           # concentration == max
    unif = np.full(10, 0.1)
    Fu = compute_layer_features([unif])
    assert Fu[0, 0] > F[0, 0]                   # uniform has higher entropy than a spike


def test_wla_weights_are_a_simplex():
    rng = np.random.default_rng(1)
    X = rng.random((200, 12 * 7)); y = (rng.random(200) > 0.5).astype(int)
    wla = WeightedLayerAggregation(12, 7, 0.5).fit(X, y)
    w = wla.get_weights()
    assert w.shape == (12,)
    assert abs(w.sum() - 1.0) < 1e-5 and (w >= 0).all()
    assert wla.transform(X).shape == (200, 7)
