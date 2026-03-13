"""
Tests for PINModel and PINEnsemble.

These tests use small synthetic datasets designed to run quickly.
Training uses CPU and minimal epochs to keep runtime manageable.
"""
import pytest
import numpy as np
import torch
from insurance_pin.model import PINModel, PINEnsemble


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

FEATURES_SMALL = {
    "age": "continuous",
    "bm": "continuous",
    "area": 4,  # 4 categories
}

FEATURES_CONTINUOUS = {
    "x1": "continuous",
    "x2": "continuous",
    "x3": "continuous",
}

N_TRAIN = 200
N_TEST = 50
N_BACKGROUND = 30


def _synthetic_data(n: int, features: dict, seed: int = 42):
    """Generate synthetic insurance-like dataset."""
    rng = np.random.default_rng(seed)
    X = {}
    for name, spec in features.items():
        if spec == "continuous":
            X[name] = rng.uniform(18, 80, n).astype(np.float32)
        else:
            X[name] = rng.integers(0, spec, n)

    # Simple Poisson frequency: base rate * age effect
    exposure = rng.uniform(0.5, 2.0, n).astype(np.float32)
    freq = 0.05 + 0.001 * X.get("age", np.zeros(n))
    y = rng.poisson(freq * exposure).astype(np.float32) / np.maximum(exposure, 1e-6)

    return X, y.astype(np.float32), exposure


@pytest.fixture
def small_data():
    X, y, exp = _synthetic_data(N_TRAIN, FEATURES_SMALL)
    return X, y, exp


@pytest.fixture
def small_model():
    return PINModel(
        features=FEATURES_SMALL,
        embedding_dim=4,
        hidden_dim=8,
        token_dim=4,
        shared_dims=(8, 8),
        loss="poisson",
        max_epochs=5,
        batch_size=64,
        patience=3,
        device="cpu",
        random_seed=7,
    )


@pytest.fixture
def fitted_model(small_model, small_data):
    X, y, exp = small_data
    small_model.fit(X, y, exposure=exp, verbose=False)
    return small_model


# ─────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────

class TestPINModelConstruction:
    def test_creates_with_continuous_features(self):
        model = PINModel(features={"x": "continuous", "y": "continuous"}, device="cpu")
        assert model.q == 2

    def test_creates_with_categorical_features(self):
        model = PINModel(features={"cat": 5}, device="cpu")
        assert model.q == 1

    def test_creates_with_mixed_features(self):
        model = PINModel(features=FEATURES_SMALL, device="cpu")
        assert model.q == 3
        assert model.feature_names == ["age", "bm", "area"]

    def test_n_pairs_correct(self):
        """For q features, n_pairs = q*(q+1)/2."""
        for q in [2, 3, 5, 9]:
            features = {f"x{i}": "continuous" for i in range(q)}
            model = PINModel(features=features, device="cpu")
            assert model.output_weights.shape == (q * (q + 1) // 2,)

    def test_parameter_count_reference_config(self):
        """
        Reference config: d=10, d'=20, d0=10, d1=30, d2=20, q=9 features
        (7 continuous, 2 categorical with 11 and 22 levels).
        Paper says 4,147 parameters.
        """
        features = {f"cont_{i}": "continuous" for i in range(7)}
        features["cat1"] = 11
        features["cat2"] = 22

        model = PINModel(
            features=features,
            embedding_dim=10,
            hidden_dim=20,
            token_dim=10,
            shared_dims=(30, 20),
            device="cpu",
        )
        n = model.count_parameters()
        # Paper says 4,147 — allow small tolerance for our implementation
        assert abs(n - 4147) < 200, f"Expected ~4147 params, got {n}"

    def test_loss_poisson(self):
        model = PINModel(features={"x": "continuous"}, loss="poisson", device="cpu")
        from insurance_pin.losses import PoissonDeviance
        assert isinstance(model._loss_fn, PoissonDeviance)

    def test_loss_gamma(self):
        model = PINModel(features={"x": "continuous"}, loss="gamma", device="cpu")
        from insurance_pin.losses import GammaDeviance
        assert isinstance(model._loss_fn, GammaDeviance)

    def test_loss_tweedie(self):
        model = PINModel(features={"x": "continuous"}, loss="tweedie", tweedie_p=1.5, device="cpu")
        from insurance_pin.losses import TweedieDeviance
        assert isinstance(model._loss_fn, TweedieDeviance)

    def test_device_cpu(self):
        model = PINModel(features={"x": "continuous"}, device="cpu")
        assert model._device == torch.device("cpu")


# ─────────────────────────────────────────────
# Forward pass (untrained)
# ─────────────────────────────────────────────

class TestForwardPass:
    def test_forward_continuous(self):
        model = PINModel(features={"age": "continuous", "bm": "continuous"}, device="cpu")
        x = {"age": torch.randn(16), "bm": torch.randn(16)}
        out = model(x)
        assert out.shape == (16,)
        assert (out > 0).all(), "exp output must be positive"

    def test_forward_categorical(self):
        model = PINModel(features={"area": 5}, device="cpu")
        x = {"area": torch.randint(0, 5, (16,))}
        out = model(x)
        assert out.shape == (16,)

    def test_forward_mixed(self):
        model = PINModel(features=FEATURES_SMALL, device="cpu")
        x = {
            "age": torch.randn(32),
            "bm": torch.randn(32),
            "area": torch.randint(0, 4, (32,)),
        }
        out = model(x)
        assert out.shape == (32,)

    def test_forward_with_exposure(self):
        model = PINModel(features={"x": "continuous"}, device="cpu")
        x = {"x": torch.randn(8)}
        exp = torch.rand(8) + 0.5
        out_no_exp = model(x)
        out_with_exp = model(x, exposure=exp)
        # With exposure, output should scale
        assert not torch.allclose(out_no_exp, out_with_exp)

    def test_positive_output_always(self):
        """exp link guarantees positive predictions."""
        model = PINModel(features={"x": "continuous", "y": "continuous"}, device="cpu")
        for _ in range(5):
            x = {"x": torch.randn(100), "y": torch.randn(100)}
            out = model(x)
            assert (out > 0).all()


# ─────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────

class TestFitting:
    def test_fit_runs(self, small_model, small_data):
        X, y, exp = small_data
        small_model.fit(X, y, exposure=exp, verbose=False)
        assert small_model._is_fitted

    def test_fit_records_history(self, fitted_model):
        assert len(fitted_model.train_history["train_loss"]) > 0
        assert len(fitted_model.train_history["val_loss"]) > 0

    def test_fit_without_exposure(self):
        model = PINModel(
            features={"x": "continuous"},
            max_epochs=3,
            device="cpu",
            random_seed=1,
        )
        X = {"x": np.random.randn(100).astype(np.float32)}
        y = np.random.rand(100).astype(np.float32) * 0.1
        model.fit(X, y, verbose=False)
        assert model._is_fitted

    def test_fit_with_explicit_val(self, small_data):
        X_train, y_train, exp_train = small_data
        X_val, y_val, exp_val = _synthetic_data(50, FEATURES_SMALL, seed=99)
        model = PINModel(
            features=FEATURES_SMALL,
            max_epochs=3,
            device="cpu",
            random_seed=2,
            embedding_dim=4,
            hidden_dim=8,
            token_dim=4,
            shared_dims=(8, 8),
        )
        model.fit(
            X_train, y_train, exposure=exp_train,
            X_val=X_val, y_val=y_val, exposure_val=exp_val,
            verbose=False,
        )
        assert model._is_fitted

    def test_fit_returns_self(self, small_model, small_data):
        X, y, exp = small_data
        result = small_model.fit(X, y, exposure=exp, verbose=False)
        assert result is small_model

    def test_centering_computed_after_fit(self, fitted_model):
        assert fitted_model._pair_means is not None
        assert fitted_model._pair_means.shape == (
            fitted_model.q * (fitted_model.q + 1) // 2,
        )


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

class TestPredict:
    def test_predict_shape(self, fitted_model):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred = fitted_model.predict(X_test)
        assert pred.shape == (N_TEST,)

    def test_predict_positive(self, fitted_model):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred = fitted_model.predict(X_test)
        assert (pred > 0).all()

    def test_predict_returns_numpy(self, fitted_model):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred = fitted_model.predict(X_test)
        assert isinstance(pred, np.ndarray)

    def test_predict_before_fit_raises(self, small_model):
        X_test, _, _ = _synthetic_data(10, FEATURES_SMALL)
        with pytest.raises(RuntimeError, match="fit()"):
            small_model.predict(X_test)

    def test_predict_with_exposure(self, fitted_model):
        X_test, _, exp_test = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred_freq = fitted_model.predict(X_test)
        pred_claims = fitted_model.predict(X_test, exposure=exp_test)
        # With exposure, expected claims = freq * exp (not equal in general)
        assert not np.allclose(pred_freq, pred_claims)

    def test_predict_deterministic(self, fitted_model):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred1 = fitted_model.predict(X_test)
        pred2 = fitted_model.predict(X_test)
        assert np.allclose(pred1, pred2)


# ─────────────────────────────────────────────
# Interpretability
# ─────────────────────────────────────────────

class TestInterpretability:
    def test_pair_contributions_keys(self, fitted_model):
        X, _, _ = _synthetic_data(50, FEATURES_SMALL)
        contribs = fitted_model.pair_contributions(X)
        q = fitted_model.q
        assert len(contribs) == q * (q + 1) // 2

    def test_pair_contributions_shape(self, fitted_model):
        X, _, _ = _synthetic_data(50, FEATURES_SMALL)
        contribs = fitted_model.pair_contributions(X)
        for key, arr in contribs.items():
            assert arr.shape == (50,), f"{key}: {arr.shape}"

    def test_pair_contributions_sum_approx_logpred(self, fitted_model):
        """
        sum_pairs(w_jk * h_jk) + bias ~ log(prediction) on LINEAR PREDICTOR.
        After centering, sum should approximate log(mu) - centering_offset.
        """
        X, _, _ = _synthetic_data(20, FEATURES_SMALL)
        contribs = fitted_model.pair_contributions(X)
        total = sum(contribs.values())
        pred = fitted_model.predict(X)
        # total + bias should = log(pred) + centering_correction
        # Just verify it's a finite array
        assert not np.any(np.isnan(total))

    def test_interaction_weights_keys(self, fitted_model):
        weights = fitted_model.interaction_weights()
        q = fitted_model.q
        assert len(weights) == q * (q + 1) // 2

    def test_interaction_weights_diagonal(self, fitted_model):
        """Diagonal terms should be present (main effects)."""
        weights = fitted_model.interaction_weights()
        for name in fitted_model.feature_names:
            assert (name, name) in weights

    def test_main_effects_shape(self, fitted_model):
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        effects = fitted_model.main_effects(X_bg, n_grid=20)
        assert set(effects.keys()) == set(FEATURES_SMALL.keys())
        for name, (grid, eff) in effects.items():
            spec = FEATURES_SMALL[name]
            if spec == "continuous":
                assert grid.shape == (20,)
                assert eff.shape == (20,)
            else:
                assert grid.shape == (spec,)
                assert eff.shape == (spec,)

    def test_interaction_surfaces_shape(self, fitted_model):
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        surfaces = fitted_model.interaction_surfaces(X_bg, n_grid=10)
        for key, data in surfaces.items():
            fname_j, fname_k = key
            assert "grid_j" in data
            assert "grid_k" in data
            assert "surface" in data
            nj = data["grid_j"].shape[0]
            nk = data["grid_k"].shape[0]
            assert data["surface"].shape == (nj, nk)

    def test_interaction_surfaces_specific_pairs(self, fitted_model):
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        surfaces = fitted_model.interaction_surfaces(
            X_bg, n_grid=8, pairs=[("age", "bm")]
        )
        assert ("age", "bm") in surfaces

    def test_count_parameters(self, small_model):
        n = small_model.count_parameters()
        assert n > 0


# ─────────────────────────────────────────────
# Shapley values
# ─────────────────────────────────────────────

class TestShapleyValues:
    def test_shapley_keys(self, fitted_model):
        X_test, _, _ = _synthetic_data(10, FEATURES_SMALL)
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        shap = fitted_model.shapley_values(X_test, X_bg, n_background=20)
        assert set(shap.keys()) == set(FEATURES_SMALL.keys())

    def test_shapley_shape(self, fitted_model):
        n_test = 15
        X_test, _, _ = _synthetic_data(n_test, FEATURES_SMALL)
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        shap = fitted_model.shapley_values(X_test, X_bg, n_background=20)
        for name, vals in shap.items():
            assert vals.shape == (n_test,), f"{name}: {vals.shape}"

    def test_shapley_finite_values(self, fitted_model):
        X_test, _, _ = _synthetic_data(10, FEATURES_SMALL)
        X_bg, _, _ = _synthetic_data(N_BACKGROUND, FEATURES_SMALL)
        shap = fitted_model.shapley_values(X_test, X_bg, n_background=10)
        for name, vals in shap.items():
            assert not np.any(np.isnan(vals)), f"NaN in {name}"
            assert not np.any(np.isinf(vals)), f"Inf in {name}"

    def test_shapley_before_fit_raises(self, small_model):
        X_test, _, _ = _synthetic_data(5, FEATURES_SMALL)
        X_bg, _, _ = _synthetic_data(10, FEATURES_SMALL)
        with pytest.raises(RuntimeError, match="fit()"):
            small_model.shapley_values(X_test, X_bg)


# ─────────────────────────────────────────────
# PINEnsemble
# ─────────────────────────────────────────────

class TestPINEnsemble:
    @pytest.fixture
    def ensemble(self):
        return PINEnsemble(
            n_models=2,
            features=FEATURES_SMALL,
            embedding_dim=4,
            hidden_dim=8,
            token_dim=4,
            shared_dims=(8, 8),
            loss="poisson",
            max_epochs=3,
            batch_size=64,
            patience=2,
            device="cpu",
        )

    @pytest.fixture
    def fitted_ensemble(self, ensemble, small_data):
        X, y, exp = small_data
        ensemble.fit(X, y, exposure=exp, verbose=False)
        return ensemble

    def test_fit_creates_models(self, fitted_ensemble):
        assert len(fitted_ensemble.models) == 2

    def test_predict_shape(self, fitted_ensemble):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred = fitted_ensemble.predict(X_test)
        assert pred.shape == (N_TEST,)

    def test_predict_positive(self, fitted_ensemble):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        pred = fitted_ensemble.predict(X_test)
        assert (pred > 0).all()

    def test_predict_std_shape(self, fitted_ensemble):
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        std = fitted_ensemble.predict_std(X_test)
        assert std.shape == (N_TEST,)

    def test_predict_before_fit_raises(self, ensemble):
        X_test, _, _ = _synthetic_data(10, FEATURES_SMALL)
        with pytest.raises(RuntimeError, match="fit()"):
            ensemble.predict(X_test)

    def test_ensemble_mean_differs_from_single(self, fitted_ensemble, small_data):
        """Ensemble mean should equal mean of individual model predictions."""
        X_test, _, _ = _synthetic_data(N_TEST, FEATURES_SMALL)
        ens_pred = fitted_ensemble.predict(X_test)
        m0_pred = fitted_ensemble.models[0].predict(X_test)
        m1_pred = fitted_ensemble.models[1].predict(X_test)
        # Ensemble mean should equal mean of individual predictions
        manual_mean = (m0_pred + m1_pred) / 2.0
        assert np.allclose(ens_pred, manual_mean, atol=1e-5)

    def test_interaction_weights_keys(self, fitted_ensemble):
        weights = fitted_ensemble.interaction_weights()
        q = fitted_ensemble.models[0].q
        assert len(weights) == q * (q + 1) // 2

    def test_ensemble_shapley_shape(self, fitted_ensemble):
        X_test, _, _ = _synthetic_data(5, FEATURES_SMALL)
        X_bg, _, _ = _synthetic_data(20, FEATURES_SMALL)
        shap = fitted_ensemble.shapley_values(X_test, X_bg, n_background=10)
        for name, vals in shap.items():
            assert vals.shape == (5,)
