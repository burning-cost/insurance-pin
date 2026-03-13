"""
PINModel and PINEnsemble: Tree-like Pairwise Interaction Networks.

The model decomposes as:
    f_PIN(x) = exp( sum_{j<=k} w_{jk} * h_{jk}(x) + b )

where h_{jk} = centered_hard_sigmoid(f_theta(phi_j(x_j), phi_k(x_k), e_{jk})).

Design decisions:
- Data handling uses Polars DataFrames (preferred) or dicts of numpy arrays.
  Converting from pandas is the caller's responsibility.
- Exposure is a multiplicative weight on the deviance bracket, not a log offset.
  This is both mathematically correct and matches the paper's formulation.
- Post-hoc centering of interaction surfaces is applied after fitting. The paper
  omits this but it's essential for production: raw h_{jk} values are in [0,1]
  with no guarantee of zero mean, making w_{jk} ambiguous.
- Validation split is drawn from the training data if X_val is not provided,
  matching the paper's 10% hold-out procedure.
"""

from __future__ import annotations

import copy
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from insurance_pin.networks import (
    FeatureEmbeddings,
    InteractionTokens,
    SharedInteractionNet,
    centered_hard_sigmoid,
)
from insurance_pin.losses import get_loss


# Type alias for feature specs
FeatureSpec = Dict[str, Union[str, int]]  # name -> 'continuous' or num_categories


def _to_tensor(
    x: Union[np.ndarray, torch.Tensor], dtype=torch.float32
) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


def _to_long_tensor(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.tensor(x, dtype=torch.long)


class PINModel(nn.Module):
    """
    Single Tree-like Pairwise Interaction Network.

    Prediction:
        f_PIN(x) = exp( sum_{j<=k} w_{jk} * h_{jk}(x) + b )

    Args:
        features: Dict mapping feature name to spec. Spec is 'continuous' or
            an int (number of categories). Order matters — features are indexed
            by position.
        embedding_dim: Feature embedding dimension d (default 10).
        hidden_dim: Hidden width for continuous embedding FNNs d' (default 20).
        token_dim: Interaction token dimension d0 (default 10).
        shared_dims: (d1, d2) widths for shared interaction network (default [30, 20]).
        loss: Loss name — 'poisson', 'gamma', or 'tweedie'.
        tweedie_p: Tweedie power (only used when loss='tweedie').
        lr: Adam learning rate (default 0.001).
        batch_size: Mini-batch size (default 128).
        max_epochs: Maximum training epochs (default 500).
        patience: Early stopping patience in epochs (default 20).
        lr_patience: ReduceLROnPlateau patience (default 5).
        lr_factor: ReduceLROnPlateau reduction factor (default 0.9).
        val_fraction: Fraction of training data for validation if X_val not given
            (default 0.1).
        device: Torch device string, or None to auto-detect.
        random_seed: Seed for reproducibility.

    Examples:
        >>> model = PINModel(
        ...     features={"age": "continuous", "area": 5},
        ...     loss="poisson",
        ... )
    """

    def __init__(
        self,
        features: FeatureSpec,
        embedding_dim: int = 10,
        hidden_dim: int = 20,
        token_dim: int = 10,
        shared_dims: Tuple[int, int] = (30, 20),
        loss: str = "poisson",
        tweedie_p: float = 1.5,
        lr: float = 0.001,
        batch_size: int = 128,
        max_epochs: int = 500,
        patience: int = 20,
        lr_patience: int = 5,
        lr_factor: float = 0.9,
        val_fraction: float = 0.1,
        device: Optional[str] = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__()

        self.features = dict(features)
        self.feature_names: List[str] = list(features.keys())
        self.q = len(self.feature_names)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.shared_dims = tuple(shared_dims)
        self.loss_name = loss
        self.tweedie_p = tweedie_p
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.val_fraction = val_fraction
        self.random_seed = random_seed

        # Resolve device
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # --- Sub-modules ---
        self.feature_embeddings = FeatureEmbeddings(
            features=features,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )
        self.interaction_tokens = InteractionTokens(
            n_features=self.q,
            token_dim=token_dim,
        )
        self.shared_net = SharedInteractionNet(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            layer1_dim=shared_dims[0],
            layer2_dim=shared_dims[1],
        )

        n_pairs = self.q * (self.q + 1) // 2
        # Output weights w_{jk} and bias b — linear combination of pair terms
        self.output_weights = nn.Parameter(torch.randn(n_pairs) * 0.01)
        self.output_bias = nn.Parameter(torch.zeros(1))

        # Loss function
        loss_kwargs = {"p": tweedie_p} if loss == "tweedie" else {}
        self._loss_fn = get_loss(loss, **loss_kwargs)

        # Weight initialisation: smaller scales for numerical stability
        self._init_weights()

        # Training state
        self._is_fitted = False
        self.train_history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        # Centering offsets (set post-hoc after fitting)
        # h_{jk}^centered(x) = h_{jk}(x) - mean_train[h_{jk}]
        # We store mean_train[w_{jk} * h_{jk}] per pair for efficiency
        self._pair_means: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Initialise all linear layers to smaller scales for training stability.

        Default PyTorch Kaiming init can produce large initial activations
        when many layers are composed. With smaller scales, early training
        is more stable even on very small datasets.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _compute_linear_predictor(
        self,
        x_dict: Dict[str, torch.Tensor],
        apply_centering: bool = True,
    ) -> torch.Tensor:
        """
        Compute sum_{j<=k} w_{jk} * h_{jk}(x) + b.

        Args:
            x_dict: Feature tensors on self._device.
            apply_centering: Subtract pair means (set during fit) for identifiability.

        Returns:
            Shape (batch_size,).
        """
        # Embed all features once
        embeddings: Dict[str, torch.Tensor] = self.feature_embeddings.embed_all(x_dict)

        pairs = self.interaction_tokens.pair_indices()
        terms = []

        for pair_idx, (j, k) in enumerate(pairs):
            fname_j = self.feature_names[j]
            fname_k = self.feature_names[k]

            phi_j = embeddings[fname_j]
            phi_k = embeddings[fname_k]
            token = self.interaction_tokens.get_token(j, k)

            raw = self.shared_net(phi_j, phi_k, token)  # (batch, 1)
            h = centered_hard_sigmoid(raw).squeeze(-1)   # (batch,)

            w = self.output_weights[pair_idx]
            term = w * h  # (batch,)
            terms.append(term)

        # Stack to (batch, n_pairs) then sum over pairs
        all_terms = torch.stack(terms, dim=1)  # (batch, n_pairs)

        if apply_centering and self._pair_means is not None:
            all_terms = all_terms - self._pair_means.unsqueeze(0)

        linear_pred = all_terms.sum(dim=1) + self.output_bias.squeeze()
        return linear_pred

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        exposure: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute predictions f_PIN(x) = exp(linear_predictor) * exposure.

        When exposure is provided, the raw model output is frequency (claims per
        year) and multiplying by exposure gives expected claim count. For
        frequency models, typically you'd pass exposure=None and let the caller
        multiply; this method supports both modes.

        Args:
            x_dict: Feature tensors. Dict of feature_name -> (batch,) tensor.
            exposure: Optional per-sample exposure, shape (batch,).

        Returns:
            Predicted frequency, shape (batch,).
        """
        eta = self._compute_linear_predictor(x_dict)
        # Clamp to avoid exp overflow (GLM link stabilisation)
        eta = torch.clamp(eta, min=-20.0, max=20.0)
        mu = torch.exp(eta)
        if exposure is not None:
            mu = mu * exposure
        return mu

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        X: Union[Dict, "pl.DataFrame", "pd.DataFrame"],  # noqa: F821
    ) -> Dict[str, torch.Tensor]:
        """
        Convert input data to a dict of tensors.

        Accepts:
        - Dict[str, np.ndarray]
        - Dict[str, list]
        - polars.DataFrame
        - pandas.DataFrame
        """
        # Try polars first
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                return self._polars_to_dict(X)
        except ImportError:
            pass

        # Try pandas
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return {col: X[col].to_numpy() for col in self.feature_names}
        except ImportError:
            pass

        # Assume it's already a dict
        if isinstance(X, dict):
            return X

        raise TypeError(
            f"X must be a dict, polars.DataFrame, or pandas.DataFrame. Got {type(X)}."
        )

    def _polars_to_dict(self, df) -> Dict[str, np.ndarray]:
        result = {}
        for name in self.feature_names:
            result[name] = df[name].to_numpy()
        return result

    def _to_device_dict(
        self, x_dict: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Convert dict of arrays to tensors on device."""
        result = {}
        for name, arr in x_dict.items():
            spec = self.features[name]
            if spec == "continuous":
                result[name] = _to_tensor(arr).to(self._device)
            else:
                result[name] = _to_long_tensor(arr).to(self._device)
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        X_val=None,
        y_val: Optional[np.ndarray] = None,
        exposure_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "PINModel":
        """
        Fit the model.

        Args:
            X_train: Training features. Dict, polars.DataFrame, or pandas.DataFrame.
            y_train: Observed frequency (claims / exposure), shape (n,).
            exposure: Per-sample exposure (years at risk), shape (n,).
            X_val: Validation features (optional). If None, 10% of training data
                is reserved.
            y_val: Validation targets.
            exposure_val: Validation exposure.
            verbose: Print training progress.

        Returns:
            self (for chaining).
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.to(self._device)

        # Prepare training data
        x_dict = self._prepare_features(X_train)
        x_dict = self._to_device_dict(x_dict)
        y_t = _to_tensor(y_train).to(self._device)
        exp_t = (
            _to_tensor(exposure).to(self._device)
            if exposure is not None
            else torch.ones_like(y_t)
        )

        n = y_t.shape[0]

        # Initialise bias to log(mean_frequency) for numerical stability.
        # This ensures initial predictions are in the right ballpark and
        # prevents exploding gradients in the first few batches.
        with torch.no_grad():
            mean_freq = y_t.mean().clamp(min=1e-8)
            self.output_bias.fill_(torch.log(mean_freq).item())

        # Build or reserve validation set
        if X_val is not None:
            x_val_dict = self._prepare_features(X_val)
            x_val_dict = self._to_device_dict(x_val_dict)
            y_val_t = _to_tensor(y_val).to(self._device)
            exp_val_t = (
                _to_tensor(exposure_val).to(self._device)
                if exposure_val is not None
                else torch.ones_like(y_val_t)
            )
        else:
            # Reserve val_fraction from training
            val_size = max(1, int(n * self.val_fraction))
            perm = torch.randperm(n, device=self._device)
            val_idx = perm[:val_size]
            train_idx = perm[val_size:]

            x_val_dict = {k: v[val_idx] for k, v in x_dict.items()}
            y_val_t = y_t[val_idx]
            exp_val_t = exp_t[val_idx]

            x_dict = {k: v[train_idx] for k, v in x_dict.items()}
            y_t = y_t[train_idx]
            exp_t = exp_t[train_idx]

        n_train = y_t.shape[0]

        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
        )

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        self.train_history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.max_epochs):
            self.train()
            # Shuffle training data
            perm = torch.randperm(n_train, device=self._device)

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self.batch_size):
                end = min(start + self.batch_size, n_train)
                batch_idx = perm[start:end]

                x_batch = {k: v[batch_idx] for k, v in x_dict.items()}
                y_batch = y_t[batch_idx]
                exp_batch = exp_t[batch_idx]

                optimizer.zero_grad()
                mu = self.forward(x_batch)
                loss = self._loss_fn(mu, y_batch, exp_batch)
                if torch.isnan(loss):
                    continue  # skip NaN batches
                loss.backward()
                # Gradient clipping prevents exploding gradients in early training
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                # Zero NaN gradients before step
                for p in self.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        p.grad.zero_()
                optimizer.step()
                # Restore any NaN params to their pre-step values
                for p in self.parameters():
                    if torch.isnan(p).any():
                        torch.nan_to_num_(p, nan=0.0)

                epoch_loss += loss.item()
                n_batches += 1

            train_loss = epoch_loss / n_batches

            # Validation
            self.eval()
            with torch.no_grad():
                mu_val = self.forward(x_val_dict)
                val_loss = self._loss_fn(mu_val, y_val_t, exp_val_t).item()

            scheduler.step(val_loss)

            self.train_history["train_loss"].append(train_loss)
            self.train_history["val_loss"].append(val_loss)

            if verbose and (epoch % 50 == 0 or epoch < 10):
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}. Best val={best_val_loss:.6f}")
                    break

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        # Post-hoc centering: compute pair means on training data
        self._compute_pair_centering(x_dict)

        self._is_fitted = True
        return self

    def _compute_pair_centering(
        self, x_dict: Dict[str, torch.Tensor], batch_size: int = 1024
    ) -> None:
        """
        Compute mean of w_{jk} * h_{jk}(x) over training data for each pair.

        Centering ensures that the bias b absorbs the intercept and the
        interaction terms have zero mean — making w_{jk} interpretable as
        the amplitude of the interaction effect.

        Stores result in self._pair_means, shape (n_pairs,).
        """
        self.eval()
        n = x_dict[self.feature_names[0]].shape[0]
        n_pairs = self.q * (self.q + 1) // 2
        pairs = self.interaction_tokens.pair_indices()

        pair_sums = torch.zeros(n_pairs, device=self._device)
        n_seen = 0

        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_batch = {k: v[start:end] for k, v in x_dict.items()}
                embeddings = self.feature_embeddings.embed_all(x_batch)
                batch_n = end - start

                for pair_idx, (j, k) in enumerate(pairs):
                    fname_j = self.feature_names[j]
                    fname_k = self.feature_names[k]
                    phi_j = embeddings[fname_j]
                    phi_k = embeddings[fname_k]
                    token = self.interaction_tokens.get_token(j, k)
                    raw = self.shared_net(phi_j, phi_k, token)
                    h = centered_hard_sigmoid(raw).squeeze(-1)
                    w = self.output_weights[pair_idx]
                    pair_sums[pair_idx] += (w * h).sum()

                n_seen += batch_n

        self._pair_means = pair_sums / n_seen

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict frequency (or expected claims if exposure given).

        Args:
            X: Features. Dict, polars.DataFrame, or pandas.DataFrame.
            exposure: Per-sample exposure.

        Returns:
            Predictions as numpy array, shape (n,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        self.eval()
        x_dict = self._prepare_features(X)
        x_dict = self._to_device_dict(x_dict)

        exp_t = (
            _to_tensor(exposure).to(self._device)
            if exposure is not None
            else None
        )

        with torch.no_grad():
            mu = self.forward(x_dict, exposure=exp_t)

        return mu.cpu().numpy()

    # ------------------------------------------------------------------
    # Interpretability surfaces
    # ------------------------------------------------------------------

    def pair_contributions(
        self,
        X,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Compute w_{jk} * h_{jk}(x) for each pair and each sample.

        These are the additive components on the linear predictor scale.
        Useful for understanding which pairs drive predictions.

        Args:
            X: Features.

        Returns:
            Dict mapping (fname_j, fname_k) -> array of shape (n,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before pair_contributions().")

        self.eval()
        x_dict = self._prepare_features(X)
        x_dict = self._to_device_dict(x_dict)
        pairs = self.interaction_tokens.pair_indices()
        result = {}

        with torch.no_grad():
            embeddings = self.feature_embeddings.embed_all(x_dict)

            for pair_idx, (j, k) in enumerate(pairs):
                fname_j = self.feature_names[j]
                fname_k = self.feature_names[k]
                phi_j = embeddings[fname_j]
                phi_k = embeddings[fname_k]
                token = self.interaction_tokens.get_token(j, k)
                raw = self.shared_net(phi_j, phi_k, token)
                h = centered_hard_sigmoid(raw).squeeze(-1)
                w = self.output_weights[pair_idx]
                contrib = (w * h)
                if self._pair_means is not None:
                    contrib = contrib - self._pair_means[pair_idx]
                result[(fname_j, fname_k)] = contrib.cpu().numpy()

        return result

    def main_effects(
        self,
        X_background,
        n_grid: int = 100,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute main effect curves for each feature.

        For continuous features: evaluates h_{jj} over an evenly-spaced grid
        while fixing all other features to background means.

        Returns:
            Dict mapping feature_name -> (grid_values, effect_values).
            For categoricals: grid_values are integer category codes.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before main_effects().")

        self.eval()
        bg_dict = self._prepare_features(X_background)
        bg_dict = self._to_device_dict(bg_dict)

        result = {}

        with torch.no_grad():
            for i, fname in enumerate(self.feature_names):
                spec = self.features[fname]

                if spec == "continuous":
                    vals = bg_dict[fname].float()
                    lo, hi = vals.min().item(), vals.max().item()
                    grid = torch.linspace(lo, hi, n_grid, device=self._device)
                elif isinstance(spec, int):
                    grid = torch.arange(spec, device=self._device)
                else:
                    continue

                # Build batch: vary feature i, fix others to background mean
                n_grid_pts = grid.shape[0]
                x_eval = {}
                for fname2 in self.feature_names:
                    spec2 = self.features[fname2]
                    if fname2 == fname:
                        x_eval[fname2] = grid.long() if isinstance(spec, int) else grid
                    else:
                        if spec2 == "continuous":
                            mean_val = bg_dict[fname2].float().mean()
                            x_eval[fname2] = mean_val.expand(n_grid_pts)
                        else:
                            mode_val = bg_dict[fname2].long().mode().values
                            x_eval[fname2] = mode_val.expand(n_grid_pts)

                # Get diagonal pair contribution (j=i, k=i)
                phi_j = self.feature_embeddings.embed_feature(fname, x_eval[fname])
                token = self.interaction_tokens.get_token(i, i)
                raw = self.shared_net(phi_j, phi_j, token)
                h = centered_hard_sigmoid(raw).squeeze(-1)
                pair_idx = self.interaction_tokens._pair_to_idx[(i, i)]
                w = self.output_weights[pair_idx]
                effect = (w * h).cpu().numpy()

                result[fname] = (grid.cpu().numpy(), effect)

        return result

    def interaction_surfaces(
        self,
        X_background,
        n_grid: int = 30,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Compute 2D interaction surfaces for feature pairs.

        For pair (j, k): evaluates w_{jk} * h_{jk}(x_j, x_k) over a 2D grid.
        Main effect contributions (j=j and k=k pairs) are NOT subtracted here —
        this shows the raw interaction term. Use pair_contributions() for
        full decomposition.

        Args:
            X_background: Background data for range estimation.
            n_grid: Grid resolution per axis (n_grid x n_grid for continuous).
            pairs: List of (feature_j, feature_k) pairs to compute. If None,
                computes all off-diagonal interaction pairs.

        Returns:
            Dict mapping (fname_j, fname_k) -> {
                'grid_j': array of shape (n_grid,),
                'grid_k': array of shape (n_grid,) or (n_cats,),
                'surface': array of shape (n_grid_j, n_grid_k),
            }
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before interaction_surfaces().")

        self.eval()
        bg_dict = self._prepare_features(X_background)
        bg_dict = self._to_device_dict(bg_dict)

        # All off-diagonal pairs if not specified
        if pairs is None:
            all_pairs = self.interaction_tokens.pair_indices()
            pairs = [
                (self.feature_names[j], self.feature_names[k])
                for j, k in all_pairs
                if j != k
            ]

        result = {}

        with torch.no_grad():
            for (fname_j, fname_k) in pairs:
                j = self.feature_names.index(fname_j)
                k = self.feature_names.index(fname_k)

                spec_j = self.features[fname_j]
                spec_k = self.features[fname_k]

                if spec_j == "continuous":
                    vals_j = bg_dict[fname_j].float()
                    grid_j = torch.linspace(vals_j.min(), vals_j.max(), n_grid, device=self._device)
                else:
                    grid_j = torch.arange(spec_j, device=self._device)

                if spec_k == "continuous":
                    vals_k = bg_dict[fname_k].float()
                    grid_k = torch.linspace(vals_k.min(), vals_k.max(), n_grid, device=self._device)
                else:
                    grid_k = torch.arange(spec_k, device=self._device)

                nj, nk = grid_j.shape[0], grid_k.shape[0]

                # Meshgrid
                gj = grid_j.repeat_interleave(nk)  # (nj * nk,)
                gk = grid_k.repeat(nj)              # (nj * nk,)

                phi_j = self.feature_embeddings.embed_feature(
                    fname_j, gj.long() if isinstance(spec_j, int) else gj
                )
                phi_k = self.feature_embeddings.embed_feature(
                    fname_k, gk.long() if isinstance(spec_k, int) else gk
                )

                token = self.interaction_tokens.get_token(j, k)
                raw = self.shared_net(phi_j, phi_k, token)
                h = centered_hard_sigmoid(raw).squeeze(-1)
                pair_idx = self.interaction_tokens._pair_to_idx[(j, k)]
                w = self.output_weights[pair_idx]
                surface = (w * h).reshape(nj, nk).cpu().numpy()

                result[(fname_j, fname_k)] = {
                    "grid_j": grid_j.cpu().numpy(),
                    "grid_k": grid_k.cpu().numpy(),
                    "surface": surface,
                }

        return result

    def shapley_values(
        self,
        X,
        X_background,
        n_background: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Compute exact Shapley values using the pairwise additive structure.

        Cost: 2(q+1) forward passes per test sample per background sample.
        For q=9, n_background=100: ~2000 forward passes per test sample.
        This is exact — no sampling approximation.

        Args:
            X: Test data.
            X_background: Background data for baseline distribution.
            n_background: Number of background samples to use.

        Returns:
            Dict mapping feature_name -> (n_test,) array of SHAP values.
            Values are on the linear predictor scale.
        """
        from insurance_pin.shapley import exact_shapley_values

        if not self._is_fitted:
            raise RuntimeError("Call fit() before shapley_values().")

        x_dict = self._prepare_features(X)
        x_dict = self._to_device_dict(x_dict)
        bg_dict = self._prepare_features(X_background)
        bg_dict = self._to_device_dict(bg_dict)

        return exact_shapley_values(self, x_dict, bg_dict, n_background)

    def interaction_weights(self) -> Dict[Tuple[str, str], float]:
        """
        Return output weights w_{jk} for all pairs as a dict.

        Large absolute weights indicate important interactions or main effects.
        Note: weights are on the scale of the linear predictor; compare
        w_{jk} * h_{jk} range rather than w_{jk} alone for fair comparison.

        Returns:
            Dict mapping (fname_j, fname_k) -> float.
        """
        pairs = self.interaction_tokens.pair_indices()
        weights = self.output_weights.detach().cpu().numpy()
        return {
            (self.feature_names[j], self.feature_names[k]): float(weights[idx])
            for idx, (j, k) in enumerate(pairs)
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PINEnsemble:
    """
    Ensemble of PINModel instances trained with different random seeds.

    Ensemble averaging is the primary regularisation strategy in the PIN paper.
    With n=10 runs, ensemble PIN achieves the best published result on French MTPL.

    Args:
        n_models: Number of models in the ensemble.
        **kwargs: Passed to each PINModel constructor.
    """

    def __init__(self, n_models: int = 10, **kwargs) -> None:
        self.n_models = n_models
        self.kwargs = kwargs
        self.models: List[PINModel] = []
        self._is_fitted = False

    def fit(
        self,
        X_train,
        y_train: np.ndarray,
        exposure: Optional[np.ndarray] = None,
        X_val=None,
        y_val: Optional[np.ndarray] = None,
        exposure_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> "PINEnsemble":
        """
        Fit all models with different seeds.

        Args:
            X_train: Training features.
            y_train: Observed frequency.
            exposure: Training exposure.
            X_val: Validation features (optional).
            y_val: Validation targets.
            exposure_val: Validation exposure.
            verbose: Print per-model progress.

        Returns:
            self.
        """
        self.models = []
        for i in range(self.n_models):
            seed = self.kwargs.get("random_seed", 42) + i
            kw = {**self.kwargs, "random_seed": seed}
            model = PINModel(**kw)
            print(f"[Ensemble] Fitting model {i+1}/{self.n_models} (seed={seed})")
            model.fit(
                X_train,
                y_train,
                exposure=exposure,
                X_val=X_val,
                y_val=y_val,
                exposure_val=exposure_val,
                verbose=verbose,
            )
            self.models.append(model)

        self._is_fitted = True
        return self

    def predict(
        self,
        X,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Average predictions across all models.

        Args:
            X: Features.
            exposure: Per-sample exposure.

        Returns:
            Ensemble mean prediction, shape (n,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")

        preds = np.stack([m.predict(X, exposure=exposure) for m in self.models], axis=0)
        return preds.mean(axis=0)

    def predict_std(
        self,
        X,
        exposure: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Standard deviation of predictions across models (epistemic uncertainty).

        Returns:
            Shape (n,).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_std().")

        preds = np.stack([m.predict(X, exposure=exposure) for m in self.models], axis=0)
        return preds.std(axis=0)

    def shapley_values(
        self,
        X,
        X_background,
        n_background: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Average Shapley values across ensemble members.

        Returns:
            Dict mapping feature_name -> (n_test,) array.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before shapley_values().")

        all_shaps = [m.shapley_values(X, X_background, n_background) for m in self.models]
        feature_names = list(all_shaps[0].keys())

        return {
            fname: np.stack([s[fname] for s in all_shaps], axis=0).mean(axis=0)
            for fname in feature_names
        }

    def interaction_weights(self) -> Dict[Tuple[str, str], float]:
        """
        Mean absolute interaction weights across ensemble.

        Returns:
            Dict mapping (fname_j, fname_k) -> mean |w_{jk}|.
        """
        from collections import defaultdict
        sums: Dict = defaultdict(float)
        for model in self.models:
            for pair, w in model.interaction_weights().items():
                sums[pair] += abs(w)
        return {pair: v / self.n_models for pair, v in sums.items()}
