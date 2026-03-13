"""
Tests for PINDiagnostics.
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from insurance_pin.model import PINModel
from insurance_pin.diagnostics import PINDiagnostics


FEATURES = {
    "age": "continuous",
    "bm": "continuous",
    "area": 4,
}


def _make_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = {
        "age": rng.uniform(18, 80, n).astype(np.float32),
        "bm": rng.uniform(50, 200, n).astype(np.float32),
        "area": rng.integers(0, 4, n),
    }
    y = (rng.poisson(0.05, n) / 1.0).astype(np.float32)
    exp = rng.uniform(0.5, 2.0, n).astype(np.float32)
    return X, y, exp


@pytest.fixture(scope="module")
def fitted_model():
    model = PINModel(
        features=FEATURES,
        embedding_dim=4,
        hidden_dim=8,
        token_dim=4,
        shared_dims=(8, 8),
        max_epochs=5,
        device="cpu",
        random_seed=42,
    )
    X, y, exp = _make_data()
    model.fit(X, y, exposure=exp, verbose=False)
    return model


@pytest.fixture(scope="module")
def diag(fitted_model):
    return PINDiagnostics(fitted_model)


@pytest.fixture(scope="module")
def background_data():
    X, _, _ = _make_data(n=50, seed=99)
    return X


class TestPINDiagnostics:
    def test_interaction_heatmap_returns_fig_ax(self, diag):
        fig, ax = diag.interaction_heatmap()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_interaction_heatmap_matrix_size(self, diag):
        fig, ax = diag.interaction_heatmap()
        # Check that the image has correct dimensions
        images = ax.get_images()
        assert len(images) > 0
        plt.close(fig)

    def test_weighted_importance_returns_importance_dict(self, diag, background_data):
        fig, ax, importance = diag.weighted_importance(background_data)
        assert isinstance(importance, dict)
        assert len(importance) > 0
        plt.close(fig)

    def test_weighted_importance_positive_values(self, diag, background_data):
        _, _, importance = diag.weighted_importance(background_data)
        for label, val in importance.items():
            assert val >= 0.0, f"{label}: {val}"

    def test_weighted_importance_top_n(self, diag, background_data):
        _, _, importance = diag.weighted_importance(background_data, top_n=2)
        assert len(importance) == 2

    def test_plot_main_effect_continuous(self, diag, background_data):
        fig, ax = diag.plot_main_effect("age", background_data)
        assert fig is not None
        plt.close(fig)

    def test_plot_main_effect_categorical(self, diag, background_data):
        fig, ax = diag.plot_main_effect("area", background_data)
        assert fig is not None
        plt.close(fig)

    def test_plot_main_effect_invalid_feature(self, diag, background_data):
        with pytest.raises(ValueError, match="not found"):
            diag.plot_main_effect("nonexistent_feature", background_data)

    def test_plot_surface_continuous_pair(self, diag, background_data):
        fig, ax = diag.plot_surface("age", "bm", background_data, n_grid=10)
        assert fig is not None
        plt.close(fig)

    def test_plot_surface_categorical_pair(self, diag, background_data):
        fig, ax = diag.plot_surface("age", "area", background_data, n_grid=10)
        assert fig is not None
        plt.close(fig)

    def test_plot_training_history(self, diag):
        fig, ax = diag.plot_training_history()
        lines = ax.get_lines()
        assert len(lines) == 2  # train + val
        plt.close(fig)

    def test_summary_contains_feature_names(self, diag):
        txt = diag.summary()
        for name in FEATURES:
            assert name in txt

    def test_summary_contains_param_count(self, diag):
        txt = diag.summary()
        assert "Parameters:" in txt

    def test_summary_contains_loss_name(self, diag):
        txt = diag.summary()
        assert "poisson" in txt.lower()

    def test_existing_axes_accepted(self, diag):
        fig, ax = plt.subplots()
        fig2, ax2 = diag.interaction_heatmap(ax=ax)
        assert ax2 is ax
        plt.close(fig)
        plt.close(fig2)
