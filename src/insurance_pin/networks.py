"""
Neural network components for PIN.

Three building blocks:
1. centered_hard_sigmoid — the activation function from the paper.
2. FeatureEmbedding — per-feature embedding networks (NOT shared).
3. SharedInteractionNet — one network for all pairs, keyed by interaction tokens.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def centered_hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    The activation used in PIN interaction units.

    Defined in the paper as:
        sigma_hard(x) = clamp((1 + x) / 2, 0, 1)

    This maps:
        x = 0  -> 0.5  (the 'centered' sense — zero input gives midpoint)
        x = -1 -> 0.0
        x =  1 -> 1.0

    Note: NOT torch.nn.Hardsigmoid, which uses clamp((x+3)/6, 0, 1) — a
    different shift chosen to match sigmoid(-3)=0.5 behaviour. The PIN paper
    uses the simpler (1+x)/2 formulation.

    Args:
        x: Input tensor, any shape.

    Returns:
        Tensor of same shape, values in [0, 1].
    """
    return torch.clamp((1.0 + x) / 2.0, min=0.0, max=1.0)


class ContinuousEmbedding(nn.Module):
    """
    Two-layer FNN embedding for a single continuous feature.

    Architecture (from paper):
        phi_j(x_j) = W_j^(2) * tanh(W_j^(1) * x_j + b_j^(1)) + b_j^(2)

    Input:  scalar (or 1-d) feature value
    Output: embedding vector of dimension d

    Args:
        embedding_dim: Output embedding dimension d.
        hidden_dim: Hidden layer width d' (default 20 in reference config).
    """

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (batch_size,) or (batch_size, 1).

        Returns:
            Shape (batch_size, embedding_dim).
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.net(x)


class CategoricalEmbedding(nn.Module):
    """
    Entity embedding for a single categorical feature.

    phi_j(x_j) = lookup W^(0)_j[x_j]  in R^d

    Standard nn.Embedding; each category gets a learned d-dimensional vector.

    Args:
        num_categories: Number of distinct categories.
        embedding_dim: Output dimension d.
    """

    def __init__(self, num_categories: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer tensor of shape (batch_size,).

        Returns:
            Shape (batch_size, embedding_dim).
        """
        return self.embedding(x.long())


class FeatureEmbeddings(nn.Module):
    """
    Collection of per-feature embeddings.

    Holds one embedding network per feature. Each is independent — the paper
    explicitly states embeddings are NOT shared. This is what allows PIN to
    learn feature-specific representations before the shared interaction net.

    Args:
        features: Dict mapping feature name to type. Type is either 'continuous'
            or an integer (number of categories for categoricals).
        embedding_dim: Shared output dimension d for all embeddings.
        hidden_dim: Hidden width for continuous FNN embeddings.
    """

    def __init__(
        self,
        features: Dict[str, str | int],
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.feature_names: List[str] = list(features.keys())
        self.feature_types: Dict[str, str | int] = dict(features)
        self.embedding_dim = embedding_dim

        modules: Dict[str, nn.Module] = {}
        for name, spec in features.items():
            if spec == "continuous":
                modules[name] = ContinuousEmbedding(embedding_dim, hidden_dim)
            elif isinstance(spec, int):
                modules[name] = CategoricalEmbedding(spec, embedding_dim)
            else:
                raise ValueError(
                    f"Feature '{name}': spec must be 'continuous' or int (num_categories). "
                    f"Got: {spec!r}"
                )
        # Use ModuleDict for proper parameter registration
        self.embeddings = nn.ModuleDict(modules)

    def embed_feature(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Embed a single named feature."""
        return self.embeddings[name](x)

    def embed_all(
        self, x_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Embed all features.

        Args:
            x_dict: Dict mapping feature name -> tensor of shape (batch_size,).

        Returns:
            Dict mapping feature name -> tensor of shape (batch_size, d).
        """
        return {name: self.embeddings[name](x_dict[name]) for name in self.feature_names}


class SharedInteractionNet(nn.Module):
    """
    The shared 3-layer FNN f_theta used for ALL feature pairs.

    Architecture:
        input  = [phi_j, phi_k, e_{jk}]  in R^{2d + d0}
        layer1 = ReLU(W1 * input + b1)    in R^{d1}
        layer2 = ReLU(W2 * layer1 + b2)   in R^{d2}
        output = W3 * layer2 + b3         in R^1

    One set of weights for all pairs. Pairs are differentiated by e_{jk}.

    The paper uses: no batch norm, no dropout, no weight decay.

    Args:
        embedding_dim: d (feature embedding dimension).
        token_dim: d0 (interaction token dimension).
        layer1_dim: d1 (first hidden layer width, default 30).
        layer2_dim: d2 (second hidden layer width, default 20).
    """

    def __init__(
        self,
        embedding_dim: int,
        token_dim: int,
        layer1_dim: int = 30,
        layer2_dim: int = 20,
    ) -> None:
        super().__init__()
        input_dim = 2 * embedding_dim + token_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, layer1_dim),
            nn.ReLU(),
            nn.Linear(layer1_dim, layer2_dim),
            nn.ReLU(),
            nn.Linear(layer2_dim, 1),
        )

    def forward(
        self,
        phi_j: torch.Tensor,
        phi_k: torch.Tensor,
        e_jk: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the pre-activation scalar for a single pair (j, k).

        Args:
            phi_j: Shape (batch_size, d).
            phi_k: Shape (batch_size, d).
            e_jk: Shape (batch_size, d0) or (d0,) — broadcast if needed.

        Returns:
            Shape (batch_size, 1).
        """
        if e_jk.dim() == 1:
            e_jk = e_jk.unsqueeze(0).expand(phi_j.size(0), -1)
        inp = torch.cat([phi_j, phi_k, e_jk], dim=-1)
        return self.net(inp)


class InteractionTokens(nn.Module):
    """
    Learnable interaction tokens e_{jk}, one per feature pair.

    For q features there are q*(q+1)/2 pairs (upper triangle including diagonal).
    Each token is a learned vector of dimension d0.

    Analogous to CLS tokens in BERT: they let one shared network produce
    pair-specific outputs. Main effects use diagonal tokens (j=j); interactions
    use off-diagonal tokens (j<k).

    Args:
        n_features: Number of features q.
        token_dim: Token dimension d0.
    """

    def __init__(self, n_features: int, token_dim: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.token_dim = token_dim
        n_pairs = n_features * (n_features + 1) // 2
        self.tokens = nn.Parameter(torch.randn(n_pairs, token_dim) * 0.01)
        # Build index: (j, k) -> flat index
        self._pair_to_idx: Dict[Tuple[int, int], int] = {}
        idx = 0
        for j in range(n_features):
            for k in range(j, n_features):
                self._pair_to_idx[(j, k)] = idx
                idx += 1

    def get_token(self, j: int, k: int) -> torch.Tensor:
        """
        Retrieve token for pair (j, k). Always j <= k.

        Returns:
            Shape (token_dim,).
        """
        if j > k:
            j, k = k, j
        idx = self._pair_to_idx[(j, k)]
        return self.tokens[idx]

    def all_tokens(self) -> torch.Tensor:
        """Return all tokens. Shape (n_pairs, token_dim)."""
        return self.tokens

    def pair_indices(self) -> List[Tuple[int, int]]:
        """List of all (j, k) pairs in canonical order."""
        return list(self._pair_to_idx.keys())
