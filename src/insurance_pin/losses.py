"""
Deviance loss functions for insurance frequency/severity modelling.

The paper (arXiv:2508.15678) uses Poisson deviance only. We add Gamma and
Tweedie for completeness — they share the same exposure-weighted structure.

Exposure enters as a multiplicative weight on the bracket, not as a log offset.
This is consistent with the paper's formulation and with classical GLM theory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_EPS = 1e-8  # Guard against log(0)


class PoissonDeviance(nn.Module):
    """
    Weighted Poisson deviance.

    L = (1/n) * sum_i 2 * v_i * (mu_i - Y_i - Y_i * log(mu_i / Y_i))

    where:
        mu_i = predicted frequency (must be positive)
        Y_i  = observed frequency = claims / exposure
        v_i  = exposure (years at risk); defaults to 1 if not provided

    When Y_i = 0, the term simplifies to 2 * v_i * mu_i (the log term vanishes
    because Y_i * log(...) -> 0 as Y_i -> 0).

    This is the standard Poisson deviance used in GLM/GAM insurance pricing.
    """

    def forward(
        self,
        mu: torch.Tensor,
        y: torch.Tensor,
        exposure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted frequency, shape (n,). Must be positive.
            y: Observed frequency (claims/exposure), shape (n,).
            exposure: Exposure weights v_i, shape (n,). Defaults to ones.

        Returns:
            Scalar loss.
        """
        if exposure is None:
            exposure = torch.ones_like(mu)

        mu = torch.clamp(mu, min=_EPS)
        # When y=0, the log term is 0 by convention. torch.where handles this.
        log_term = torch.where(
            y > 0,
            y * torch.log(y / mu),
            torch.zeros_like(y),
        )
        bracket = mu - y - log_term
        return 2.0 * (exposure * bracket).mean()


class GammaDeviance(nn.Module):
    """
    Weighted Gamma deviance.

    L = (1/n) * sum_i 2 * v_i * (-log(Y_i / mu_i) + (Y_i - mu_i) / mu_i)

    Used for severity (cost per claim). Y_i must be strictly positive.
    """

    def forward(
        self,
        mu: torch.Tensor,
        y: torch.Tensor,
        exposure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted severity, shape (n,). Must be positive.
            y: Observed severity, shape (n,). Must be positive.
            exposure: Claim counts (or other weights), shape (n,). Defaults to ones.

        Returns:
            Scalar loss.
        """
        if exposure is None:
            exposure = torch.ones_like(mu)

        mu = torch.clamp(mu, min=_EPS)
        y = torch.clamp(y, min=_EPS)
        bracket = -torch.log(y / mu) + (y - mu) / mu
        return 2.0 * (exposure * bracket).mean()


class TweedieDeviance(nn.Module):
    """
    Weighted Tweedie deviance.

    L = (1/n) * sum_i 2 * v_i * (
            Y_i^(2-p) / ((1-p)(2-p))
            - Y_i * mu_i^(1-p) / (1-p)
            + mu_i^(2-p) / (2-p)
        )

    The Tweedie family interpolates between Poisson (p=1) and Gamma (p=2) and
    is useful for pure premium modelling (frequency x severity in one step).

    Typical values: p=1.5 (common in actuarial practice), p=1.6-1.8.

    Args:
        p: Tweedie power. Must satisfy 1 < p < 2.
    """

    def __init__(self, p: float = 1.5) -> None:
        super().__init__()
        if not (1.0 < p < 2.0):
            raise ValueError(f"Tweedie power p must satisfy 1 < p < 2, got {p}")
        self.p = p

    def forward(
        self,
        mu: torch.Tensor,
        y: torch.Tensor,
        exposure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mu: Predicted mean, shape (n,). Must be positive.
            y: Observed response, shape (n,). Non-negative.
            exposure: Weights v_i, shape (n,). Defaults to ones.

        Returns:
            Scalar loss.
        """
        if exposure is None:
            exposure = torch.ones_like(mu)

        p = self.p
        mu = torch.clamp(mu, min=_EPS)
        y = torch.clamp(y, min=0.0)

        term1 = y.pow(2 - p) / ((1 - p) * (2 - p))
        term2 = y * mu.pow(1 - p) / (1 - p)
        term3 = mu.pow(2 - p) / (2 - p)

        bracket = term1 - term2 + term3
        return 2.0 * (exposure * bracket).mean()


def get_loss(name: str, **kwargs) -> nn.Module:
    """
    Factory for loss functions by name.

    Args:
        name: One of 'poisson', 'gamma', 'tweedie'.
        **kwargs: Passed to constructor (e.g., p=1.5 for Tweedie).

    Returns:
        Loss module instance.
    """
    name = name.lower()
    if name == "poisson":
        return PoissonDeviance()
    elif name == "gamma":
        return GammaDeviance()
    elif name == "tweedie":
        return TweedieDeviance(**kwargs)
    else:
        raise ValueError(f"Unknown loss '{name}'. Choose: poisson, gamma, tweedie.")
