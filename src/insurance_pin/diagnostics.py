"""
PINDiagnostics: visualisation and inspection for fitted PIN models.

The methods here are intended for interactive use in notebooks and for
producing figures for pricing documentation. They wrap the surface/effect
computation on PINModel into plots you'd actually show a pricing team.

Matplotlib is imported lazily (on first use) to avoid import-time failures
in environments where matplotlib is not installed or has version conflicts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _get_plt():
    """Lazy matplotlib.pyplot import."""
    import matplotlib.pyplot as plt
    return plt


class PINDiagnostics:
    """
    Visualisation tools for a fitted PINModel.

    Args:
        model: Fitted PINModel instance.
    """

    def __init__(self, model) -> None:
        self.model = model

    def interaction_heatmap(
        self,
        figsize: Tuple[int, int] = (8, 7),
        cmap: str = "RdBu_r",
        title: str = "PIN Interaction Weights |w_{jk}|",
        ax=None,
    ):
        """
        Heatmap of interaction weight magnitudes.

        Rows and columns are features. The upper triangle (j<k) shows
        interactions; the diagonal shows main effects.

        Large values indicate pairs that contribute more variance to the
        linear predictor. Note: weight magnitude alone doesn't measure
        importance — the range of h_{jk} also matters. Use
        weighted_importance() for a more meaningful ranking.

        Args:
            figsize: Figure size.
            cmap: Colormap.
            title: Plot title.
            ax: Existing axes to draw on (optional).

        Returns:
            (fig, ax) tuple.
        """
        plt = _get_plt()
        weights = self.model.interaction_weights()
        feature_names = self.model.feature_names
        q = len(feature_names)

        matrix = np.zeros((q, q))
        for (fname_j, fname_k), w in weights.items():
            j = feature_names.index(fname_j)
            k = feature_names.index(fname_k)
            matrix[j, k] = abs(w)
            if j != k:
                matrix[k, j] = abs(w)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, label="|w_{jk}|")

        ax.set_xticks(range(q))
        ax.set_yticks(range(q))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticklabels(feature_names)
        ax.set_title(title)

        for i in range(q):
            for j in range(q):
                ax.text(
                    j, i, f"{matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7,
                )

        fig.tight_layout()
        return fig, ax

    def weighted_importance(
        self,
        X_background,
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (8, 5),
        ax=None,
    ):
        """
        Rank interaction pairs by the range of w_{jk} * h_{jk}(x).

        The range (max - min) over the background data measures how much
        the pair actually varies — a large weight with a flat h_{jk} adds
        almost nothing. This gives a fairer importance metric than |w_{jk}|.

        Args:
            X_background: Background data for evaluating h_{jk}.
            top_n: Plot only top N pairs. None = all.
            figsize: Figure size.
            ax: Existing axes.

        Returns:
            (fig, ax, importance_dict).
        """
        plt = _get_plt()
        contribs = self.model.pair_contributions(X_background)

        importance = {}
        for (fname_j, fname_k), vals in contribs.items():
            label = f"{fname_j}" if fname_j == fname_k else f"{fname_j} x {fname_k}"
            importance[label] = float(vals.max() - vals.min())

        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            sorted_items = sorted_items[:top_n]

        labels = [it[0] for it in sorted_items]
        values = [it[1] for it in sorted_items]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        colors = [
            "#2196F3" if " x " not in label else "#FF5722"
            for label in labels
        ]

        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("Range of w_{jk} * h_{jk}(x) on background")
        ax.set_title("PIN Pair Importance (main effects=blue, interactions=orange)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()

        return fig, ax, dict(sorted_items)

    def plot_main_effect(
        self,
        feature: str,
        X_background,
        n_grid: int = 100,
        figsize: Tuple[int, int] = (6, 4),
        ax=None,
        color: str = "#2196F3",
    ):
        """
        Plot main effect curve for a single feature.

        Evaluates the diagonal pair term w_{jj} * h_{jj}(x_j) over a grid
        of feature values while fixing other features to background means.

        Args:
            feature: Feature name.
            X_background: Background data.
            n_grid: Grid resolution.
            figsize: Figure size.
            ax: Existing axes.
            color: Line colour.

        Returns:
            (fig, ax) tuple.
        """
        plt = _get_plt()
        effects = self.model.main_effects(X_background, n_grid=n_grid)

        if feature not in effects:
            raise ValueError(f"Feature '{feature}' not found. Available: {list(effects.keys())}")

        grid_vals, effect_vals = effects[feature]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        spec = self.model.features[feature]
        if isinstance(spec, int):
            ax.bar(grid_vals, effect_vals, color=color, alpha=0.8)
            ax.set_xlabel(f"{feature} (category)")
        else:
            ax.plot(grid_vals, effect_vals, color=color, linewidth=2)
            ax.fill_between(grid_vals, effect_vals, alpha=0.15, color=color)
            ax.set_xlabel(feature)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("w_{jj} * h_{jj}(x) (linear predictor scale)")
        ax.set_title(f"Main Effect: {feature}")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return fig, ax

    def plot_surface(
        self,
        feature_j: str,
        feature_k: str,
        X_background,
        n_grid: int = 30,
        figsize: Tuple[int, int] = (7, 5),
        cmap: str = "RdBu_r",
        ax=None,
    ):
        """
        Plot 2D interaction surface for a feature pair.

        Evaluates w_{jk} * h_{jk}(x_j, x_k) over a grid. Points with
        high absolute values indicate combinations where the interaction
        adds or subtracts substantially from the linear predictor.

        Args:
            feature_j: First feature (x-axis for continuous).
            feature_k: Second feature (y-axis or color dimension).
            X_background: Background data for axis range estimation.
            n_grid: Grid resolution per axis.
            figsize: Figure size.
            cmap: Colormap.
            ax: Existing axes.

        Returns:
            (fig, ax) tuple.
        """
        plt = _get_plt()
        surfaces = self.model.interaction_surfaces(
            X_background,
            n_grid=n_grid,
            pairs=[(feature_j, feature_k)],
        )

        key = (feature_j, feature_k)
        if key not in surfaces:
            key = (feature_k, feature_j)

        if key not in surfaces:
            raise ValueError(
                f"Pair ({feature_j}, {feature_k}) not found in surfaces. "
                "Ensure both features exist and j < k in feature order."
            )

        surf_data = surfaces[key]
        grid_j = surf_data["grid_j"]
        grid_k = surf_data["grid_k"]
        surface = surf_data["surface"]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        vmax = np.abs(surface).max()
        vmin = -vmax

        spec_j = self.model.features[feature_j]
        spec_k = self.model.features[feature_k]

        if isinstance(spec_k, int):
            for cat_idx in range(grid_k.shape[0]):
                ax.plot(
                    grid_j,
                    surface[:, cat_idx],
                    label=f"{feature_k}={int(grid_k[cat_idx])}",
                    alpha=0.8,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            ax.set_xlabel(feature_j)
            ax.set_ylabel("w * h (interaction)")
        else:
            im = ax.pcolormesh(
                grid_j, grid_k, surface.T,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            plt.colorbar(im, ax=ax, label="w_{jk} * h_{jk}(x)")
            ax.set_xlabel(feature_j)
            ax.set_ylabel(feature_k)

        ax.set_title(f"Interaction Surface: {feature_j} x {feature_k}")
        fig.tight_layout()

        return fig, ax

    def plot_training_history(
        self,
        figsize: Tuple[int, int] = (8, 4),
        ax=None,
    ):
        """
        Plot training and validation loss curves.

        Args:
            figsize: Figure size.
            ax: Existing axes.

        Returns:
            (fig, ax) tuple.
        """
        plt = _get_plt()
        history = self.model.train_history

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        epochs = range(len(history["train_loss"]))
        ax.plot(epochs, history["train_loss"], label="Train", color="#2196F3")
        ax.plot(epochs, history["val_loss"], label="Validation", color="#FF5722")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (deviance)")
        ax.set_title("PIN Training History")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        return fig, ax

    def summary(self, X_background=None) -> str:
        """
        Print a text summary of the fitted model.

        Args:
            X_background: Optional background data for pair importance ranking.

        Returns:
            Summary string.
        """
        model = self.model
        lines = [
            "=" * 60,
            "PIN Model Summary",
            "=" * 60,
            f"Features:   {model.q}",
            f"Parameters: {model.count_parameters():,}",
            f"Loss:       {model.loss_name}",
            f"Embedding:  d={model.embedding_dim}, d'={model.hidden_dim}, d0={model.token_dim}",
            f"Shared net: d1={model.shared_dims[0]}, d2={model.shared_dims[1]}",
            f"n_pairs:    {model.q * (model.q + 1) // 2} "
            f"({model.q} main + {model.q*(model.q-1)//2} interactions)",
            "",
            "Feature list:",
        ]
        for name in model.feature_names:
            spec = model.features[name]
            ftype = "continuous" if spec == "continuous" else f"categorical ({spec} levels)"
            lines.append(f"  {name}: {ftype}")

        lines.append("")
        lines.append("Output weights (|w_{jk}|, top 10):")
        weights = model.interaction_weights()
        sorted_w = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for (fj, fk), w in sorted_w:
            tag = "(main)" if fj == fk else "(interaction)"
            lines.append(f"  {fj} x {fk}: {w:+.4f} {tag}")

        if model.train_history["val_loss"]:
            best_val = min(model.train_history["val_loss"])
            lines.append(f"\nBest val loss: {best_val:.6f}")
            lines.append(f"Epochs run:   {len(model.train_history['val_loss'])}")

        lines.append("=" * 60)
        txt = "\n".join(lines)
        print(txt)
        return txt
