"""
Tests for insurance_pin.losses module.
"""
import pytest
import torch
import numpy as np
from insurance_pin.losses import PoissonDeviance, GammaDeviance, TweedieDeviance, get_loss


def _make_safe_inputs(n=100, y_min=0.01, mu_min=0.01):
    """Random mu and y values that are positive and sensible."""
    torch.manual_seed(123)
    mu = torch.rand(n) + mu_min
    y = torch.rand(n) + y_min
    exposure = torch.rand(n) * 2 + 0.1
    return mu, y, exposure


# ─────────────────────────────────────────────
# PoissonDeviance
# ─────────────────────────────────────────────

class TestPoissonDeviance:
    def test_returns_scalar(self):
        loss = PoissonDeviance()
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.shape == ()

    def test_non_negative(self):
        loss = PoissonDeviance()
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.item() >= 0.0

    def test_zero_when_mu_equals_y(self):
        """Deviance is minimised (zero) when mu == y."""
        loss = PoissonDeviance()
        n = 50
        y = torch.rand(n) + 0.1
        mu = y.clone()
        exp = torch.ones(n)
        out = loss(mu, y, exp)
        assert out.item() < 1e-5, f"Expected ~0 but got {out.item()}"

    def test_handles_zero_y(self):
        """When Y=0, bracket should be 2*v*mu (log term vanishes)."""
        loss = PoissonDeviance()
        mu = torch.tensor([1.0, 2.0, 0.5])
        y = torch.tensor([0.0, 0.0, 0.0])
        exp = torch.tensor([1.0, 1.0, 1.0])
        # Should not raise or produce NaN
        out = loss(mu, y, exp)
        assert not torch.isnan(out)
        assert out.item() > 0

    def test_exposure_weighting(self):
        """Higher exposure -> higher loss for same deviation."""
        loss = PoissonDeviance()
        mu = torch.tensor([2.0])
        y = torch.tensor([1.0])
        exp_low = torch.tensor([0.5])
        exp_high = torch.tensor([5.0])
        out_low = loss(mu, y, exp_low)
        out_high = loss(mu, y, exp_high)
        assert out_high.item() > out_low.item()

    def test_gradient_flows(self):
        loss = PoissonDeviance()
        mu = torch.rand(20) + 0.1
        mu.requires_grad_(True)
        y = torch.rand(20) + 0.1
        out = loss(mu, y)
        out.backward()
        assert mu.grad is not None

    def test_no_exposure_defaults_to_ones(self):
        loss = PoissonDeviance()
        mu, y, _ = _make_safe_inputs(20)
        out_no_exp = loss(mu, y)
        out_ones = loss(mu, y, torch.ones(20))
        assert torch.isclose(out_no_exp, out_ones, atol=1e-5)

    def test_formula_correctness(self):
        """Verify against manual calculation."""
        loss = PoissonDeviance()
        mu = torch.tensor([2.0])
        y = torch.tensor([1.5])
        v = torch.tensor([1.0])
        # 2 * v * (mu - y - y*log(mu/y))
        expected = 2 * 1.0 * (2.0 - 1.5 - 1.5 * np.log(2.0 / 1.5))
        out = loss(mu, y, v)
        assert abs(out.item() - expected) < 1e-5


# ─────────────────────────────────────────────
# GammaDeviance
# ─────────────────────────────────────────────

class TestGammaDeviance:
    def test_returns_scalar(self):
        loss = GammaDeviance()
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.shape == ()

    def test_non_negative(self):
        loss = GammaDeviance()
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.item() >= 0.0

    def test_zero_when_mu_equals_y(self):
        loss = GammaDeviance()
        n = 50
        y = torch.rand(n) + 0.1
        mu = y.clone()
        exp = torch.ones(n)
        out = loss(mu, y, exp)
        assert out.item() < 1e-5

    def test_gradient_flows(self):
        loss = GammaDeviance()
        mu = torch.rand(20) + 0.1
        mu.requires_grad_(True)
        y = torch.rand(20) + 0.1
        out = loss(mu, y)
        out.backward()
        assert mu.grad is not None

    def test_formula_correctness(self):
        loss = GammaDeviance()
        mu = torch.tensor([2.0])
        y = torch.tensor([3.0])
        v = torch.tensor([1.0])
        # 2 * v * (-log(y/mu) + (y - mu)/mu)
        expected = 2 * 1.0 * (-np.log(3.0 / 2.0) + (3.0 - 2.0) / 2.0)
        out = loss(mu, y, v)
        assert abs(out.item() - expected) < 1e-5


# ─────────────────────────────────────────────
# TweedieDeviance
# ─────────────────────────────────────────────

class TestTweedieDeviance:
    def test_returns_scalar(self):
        loss = TweedieDeviance(p=1.5)
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.shape == ()

    def test_non_negative(self):
        loss = TweedieDeviance(p=1.5)
        mu, y, exp = _make_safe_inputs()
        out = loss(mu, y, exp)
        assert out.item() >= 0.0

    def test_invalid_p_raises(self):
        with pytest.raises(ValueError, match="1 < p < 2"):
            TweedieDeviance(p=2.0)
        with pytest.raises(ValueError, match="1 < p < 2"):
            TweedieDeviance(p=1.0)
        with pytest.raises(ValueError, match="1 < p < 2"):
            TweedieDeviance(p=0.5)

    def test_gradient_flows(self):
        loss = TweedieDeviance(p=1.5)
        mu = torch.rand(20) + 0.1
        mu.requires_grad_(True)
        y = torch.rand(20) + 0.1
        out = loss(mu, y)
        out.backward()
        assert mu.grad is not None

    def test_different_p_values(self):
        for p in [1.1, 1.3, 1.5, 1.7, 1.9]:
            loss = TweedieDeviance(p=p)
            mu, y, exp = _make_safe_inputs(20)
            out = loss(mu, y, exp)
            assert not torch.isnan(out), f"NaN at p={p}"

    def test_handles_zero_y(self):
        """Tweedie should handle y=0 samples."""
        loss = TweedieDeviance(p=1.5)
        mu = torch.tensor([1.0, 2.0])
        y = torch.tensor([0.0, 1.0])
        exp = torch.ones(2)
        out = loss(mu, y, exp)
        assert not torch.isnan(out)


# ─────────────────────────────────────────────
# get_loss factory
# ─────────────────────────────────────────────

class TestGetLoss:
    def test_poisson(self):
        loss = get_loss("poisson")
        assert isinstance(loss, PoissonDeviance)

    def test_gamma(self):
        loss = get_loss("gamma")
        assert isinstance(loss, GammaDeviance)

    def test_tweedie(self):
        loss = get_loss("tweedie", p=1.5)
        assert isinstance(loss, TweedieDeviance)

    def test_case_insensitive(self):
        loss = get_loss("POISSON")
        assert isinstance(loss, PoissonDeviance)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("mse")
