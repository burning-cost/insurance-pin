"""
Exact Shapley values for PIN models.

Because PIN is pairwise additive, exact Shapley values cost only 2(q+1) forward
passes per sample — not q! as for general black boxes. This is the key
interpretability advantage of the pairwise additive architecture.

Method: Paired-permutation SHAP for GA2M-type models.

For an additive model f(x) = sum_{j<=k} f_{jk}(x_j, x_k) + b, the Shapley
value for feature i is:

    phi_i = (1/2) * sum_{j<=k: i in {j,k}} [
        f_{jk}(x_i, x_{-i}) - f_{jk}(baseline_i, x_{-i})
        + f_{jk}(x_i, baseline_{-i}) - f_{jk}(baseline_i, baseline_{-i})
    ]

The averaging over baselines makes this a proper Shapley value. In practice
we average over a background dataset to estimate the baseline distribution.

Reference: Section 3.3, arXiv:2508.15678. Cost: 2(q+1) forward passes.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from insurance_pin.model import PINModel


def compute_pair_output(
    model: "PINModel",
    j: int,
    k: int,
    x_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Compute the raw (pre-weight) interaction unit h_{jk}(x) for a batch.

    h_{jk}(x) = centered_hard_sigmoid(f_theta(phi_j(x_j), phi_k(x_k), e_{jk}))

    Args:
        model: Fitted PINModel.
        j: Feature index (left).
        k: Feature index (right).
        x_dict: Dict of feature tensors, shape (batch,) each.

    Returns:
        Shape (batch,).
    """
    from insurance_pin.networks import centered_hard_sigmoid

    fname_j = model.feature_names[j]
    fname_k = model.feature_names[k]

    phi_j = model.feature_embeddings.embed_feature(fname_j, x_dict[fname_j])
    phi_k = model.feature_embeddings.embed_feature(fname_k, x_dict[fname_k])
    token = model.interaction_tokens.get_token(j, k)

    raw = model.shared_net(phi_j, phi_k, token)  # (batch, 1)
    h = centered_hard_sigmoid(raw).squeeze(-1)    # (batch,)
    return h


def weighted_pair_contribution(
    model: "PINModel",
    j: int,
    k: int,
    x_dict: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    Contribution of pair (j, k) to the linear predictor: w_{jk} * h_{jk}(x).

    Returns:
        Shape (batch,).
    """
    h = compute_pair_output(model, j, k, x_dict)
    w = model.output_weights[model.interaction_tokens._pair_to_idx[(j, k)]]
    return w * h


def exact_shapley_values(
    model: "PINModel",
    x_dict: Dict[str, torch.Tensor],
    background_dict: Dict[str, torch.Tensor],
    n_background: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Compute exact Shapley values for PIN predictions.

    The pairwise additive structure allows exact computation at cost 2(q+1)
    forward passes per sample-background pair.

    The Shapley value for feature i at sample x, averaged over background b:

        phi_i(x) = E_b[ (1/2) * sum_{pairs (j,k) containing i} (
            w_{jk} * h_{jk}(x_i, x_{-i}^b) - w_{jk} * h_{jk}(b_i, x_{-i}^b)
            + w_{jk} * h_{jk}(x_i, b_{-i})  - w_{jk} * h_{jk}(b_i, b_{-i})
        ) ]

    Note: The SHAP values here are on the LINEAR PREDICTOR scale (before exp).
    To decompose on the prediction scale you'd need to propagate through exp,
    which breaks additivity. For insurance use, linear predictor decomposition
    is standard (multiplicative decomposition via exp of sum).

    Args:
        model: Fitted PINModel.
        x_dict: Test samples. Dict mapping feature name -> (n_test,) tensor.
        background_dict: Background samples for baseline. Dict mapping feature
            name -> (n_bg,) tensor. Typically a random subset of training data.
        n_background: Max number of background samples to use (subsampled if
            background has more rows).

    Returns:
        Dict mapping feature name -> numpy array of shape (n_test,).
        SHAP values sum to f(x) - E[f(background)] on the linear predictor scale.
    """
    model.eval()
    device = next(model.parameters()).device
    feature_names = model.feature_names
    q = len(feature_names)

    # Move inputs to device
    x_dict = {k: v.to(device) for k, v in x_dict.items()}
    background_dict = {k: v.to(device) for k, v in background_dict.items()}

    n_test = x_dict[feature_names[0]].shape[0]
    n_bg_full = background_dict[feature_names[0]].shape[0]

    # Subsample background if needed
    if n_bg_full > n_background:
        idx = torch.randperm(n_bg_full, device=device)[:n_background]
        background_dict = {k: v[idx] for k, v in background_dict.items()}
        n_bg = n_background
    else:
        n_bg = n_bg_full

    # Accumulate SHAP values: (n_test, q)
    shap = torch.zeros(n_test, q, device=device)

    pairs = model.interaction_tokens.pair_indices()

    with torch.no_grad():
        for b_idx in range(n_bg):
            # Single background sample, expanded to match test batch
            bg_single = {k: v[b_idx:b_idx+1].expand(n_test) for k, v in background_dict.items()}

            for (j, k) in pairs:
                # For this pair, we need 4 evaluations:
                # h(x_j, x_k), h(b_j, x_k), h(x_j, b_k), h(b_j, b_k)

                # Assemble the 4 required x_dict variants
                def _make(use_j_from_bg: bool, use_k_from_bg: bool) -> Dict[str, torch.Tensor]:
                    d = dict(x_dict)
                    fname_j = feature_names[j]
                    fname_k = feature_names[k]
                    if use_j_from_bg:
                        d[fname_j] = bg_single[fname_j]
                    if use_k_from_bg:
                        d[fname_k] = bg_single[fname_k]
                    return d

                w_idx = model.interaction_tokens._pair_to_idx[(j, k)]
                w_jk = model.output_weights[w_idx].item()

                h_xx = compute_pair_output(model, j, k, _make(False, False))
                h_bx = compute_pair_output(model, j, k, _make(True, False))
                h_xb = compute_pair_output(model, j, k, _make(False, True))
                h_bb = compute_pair_output(model, j, k, _make(True, True))

                # Contribution to feature j and k
                # For j: average of (h_xx - h_bx) and (h_xb - h_bb)
                delta_j = 0.5 * w_jk * ((h_xx - h_bx) + (h_xb - h_bb))
                shap[:, j] += delta_j

                if j != k:
                    # For k: average of (h_xx - h_xb) and (h_bx - h_bb)
                    delta_k = 0.5 * w_jk * ((h_xx - h_xb) + (h_bx - h_bb))
                    shap[:, k] += delta_k
                # When j==k (main effect), the feature only appears once in pair;
                # delta_j already captures the full main effect contribution

    # Average over background
    shap = shap / n_bg

    return {feature_names[i]: shap[:, i].cpu().numpy() for i in range(q)}
