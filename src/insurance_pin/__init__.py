"""
insurance-pin: Tree-like Pairwise Interaction Networks for insurance pricing.

PIN is a neural GA2M where the prediction decomposes as:

    f_PIN(x) = exp( sum_{j<=k} w_{jk} * h_{jk}(x) + b )

where h_{jk}(x) = centered_hard_sigmoid(f_theta(phi_j(x_j), phi_k(x_k), e_{jk})).

One shared network f_theta serves ALL feature pairs, differentiated by learned
interaction tokens e_{jk}. Diagonal terms (j=k) recover main effects.

References:
    Richman, Scognamiglio, Wüthrich. "Tree-like Pairwise Interaction Networks."
    arXiv:2508.15678 (August 2025).
"""

from insurance_pin.model import PINModel, PINEnsemble
from insurance_pin.diagnostics import PINDiagnostics
from insurance_pin.networks import centered_hard_sigmoid

__all__ = [
    "PINModel",
    "PINEnsemble",
    "PINDiagnostics",
    "centered_hard_sigmoid",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("insurance-pin")
except PackageNotFoundError:
    __version__ = "0.0.0"  # not installed
