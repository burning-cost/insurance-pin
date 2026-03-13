# insurance-pin

**Tree-like Pairwise Interaction Networks for insurance pricing.**

PIN is a neural GA2M — the same additive structure as a GLM with interaction
terms, but each shape function is parameterised by a shared neural network
keyed by learned interaction tokens. One small network. All pairs. Best
published result on French MTPL at 4,147 parameters.

## The Problem

Pricing teams need models that:
1. Produce predictions that decompose — so you can explain why risk A is rated
   differently from risk B, not just that it is.
2. Capture interactions between features — bonus-malus and vehicle brand are
   not independent, and pretending they are leaves money on the table.
3. Beat GAMs and GLM-with-interactions on held-out deviance — otherwise why
   bother with the added complexity?

EBMs (GA2M) solve 1 and 2 but are tree-based and stage-wise. FNNs solve 3 but
fail 1 and 2. PIN solves all three.

## The Model

```
f_PIN(x) = exp( sum_{j<=k} w_{jk} * h_{jk}(x) + b )
```

where:

```
h_{jk}(x) = clamp((1 + f_theta(phi_j(x_j), phi_k(x_k), e_{jk})) / 2, 0, 1)
```

- `phi_j(x_j)` — per-feature embedding (not shared). 2-layer FNN for continuous,
  entity embedding for categorical. Output dimension d.
- `f_theta` — **one shared 3-layer FNN** for all pairs. Input: `[phi_j, phi_k, e_{jk}]`.
- `e_{jk}` — learned interaction token (d0-dimensional). This is the key idea:
  one network, pair-specific behaviour via tokens. Analogous to CLS tokens in BERT.
- `w_{jk}` — scalar output weight per pair.
- Diagonal terms (j=k) are main effects. Off-diagonal (j<k) are interactions.

**Paper:** Richman, Scognamiglio, Wüthrich. "Tree-like Pairwise Interaction Networks."
arXiv:2508.15678 (August 2025).

**Result on freMTPL2freq (Poisson deviance x10^-2):**

| Model | Deviance |
|-------|----------|
| Null model | 25.445 |
| Poisson GLM | 24.102 |
| Poisson GAM | 23.956 |
| Ensemble FNN | 23.783 |
| Ensemble CAFFT (27k params) | 23.726 |
| Ensemble Credibility TRM | 23.711 |
| **Ensemble PIN (4,147 params)** | **23.667** |

## Install

```bash
pip install insurance-pin
```

Requires: `torch>=2.0`, `polars>=0.20`, `numpy>=1.24`, `matplotlib>=3.7`.

## Usage

```python
from insurance_pin import PINModel, PINEnsemble, PINDiagnostics

# Feature specification
features = {
    "age_driver":  "continuous",
    "bonus_malus": "continuous",
    "density":     "continuous",
    "veh_age":     "continuous",
    "drive_age":   "continuous",
    "veh_power":   "continuous",
    "longitude":   "continuous",
    "veh_brand":   11,   # 11 categories
    "region":      22,   # 22 regions
}

# Reference config from paper: 4,147 parameters for 9 features
model = PINModel(
    features=features,
    embedding_dim=10,      # d
    hidden_dim=20,         # d' (continuous embedding hidden width)
    token_dim=10,          # d0
    shared_dims=(30, 20),  # d1, d2
    loss="poisson",        # "gamma" or "tweedie" also available
    lr=0.001,
    batch_size=128,
    max_epochs=300,
    patience=20,
)

# y is FREQUENCY (claims / exposure), not claim count
model.fit(X_train, y_train, exposure=exposure_train)

# Predict frequency
freq_pred = model.predict(X_test)

# Predict expected claims (frequency * exposure)
claims_pred = model.predict(X_test, exposure=exposure_test)
```

### Ensemble (recommended for production)

```python
ensemble = PINEnsemble(
    n_models=10,
    features=features,
    **same_kwargs_as_above,
)
ensemble.fit(X_train, y_train, exposure=exposure_train)

freq_pred = ensemble.predict(X_test)
uncertainty = ensemble.predict_std(X_test)  # epistemic uncertainty
```

### Interpretability

```python
diag = PINDiagnostics(model)

# Which pairs matter most?
diag.interaction_heatmap()
fig, ax, importance = diag.weighted_importance(X_background)

# Main effect curves
diag.plot_main_effect("bonus_malus", X_background)

# Interaction surfaces
diag.plot_surface("bonus_malus", "veh_brand", X_background)

# Exact SHAP — not an approximation
# Cost: 2*(q+1) forward passes per sample per background sample
shap_values = model.shapley_values(X_test, X_background, n_background=200)
```

### Access raw pair contributions

```python
# Returns w_{jk} * h_{jk}(x) for each pair, shape (n,)
contribs = model.pair_contributions(X_test)

# sum(contribs.values()) + bias ≈ log(prediction)  [linear predictor scale]
```

## Architecture Details

### Why shared weights?

A separate network per pair (like ANAM) would require O(q^2) networks. For q=9
features that's 45 networks. Instead, PIN trains one network f_theta and
differentiates pairs via learned tokens e_{jk}. This is what keeps the param
count at 4,147 while modelling all 45 pairs simultaneously.

### Interaction tokens

Each pair (j,k) gets a learned d0-dimensional vector. These are nn.Parameters
initialized near zero and trained alongside all other weights. The token tells
f_theta which pair it's computing, analogously to how BERT's CLS token identifies
the task.

### Centered hard sigmoid

The activation is `clamp((1+x)/2, 0, 1)`:
- x=0 maps to 0.5 (centred at origin)
- x=±1 are the saturation points
- Gradient is 0.5 in the linear region

This is **not** `torch.nn.Hardsigmoid`, which uses `clamp((x+3)/6, 0, 1)`.

### Post-hoc centering

After fitting, we subtract the training mean of each `w_{jk} * h_{jk}` term.
The paper doesn't do this, but it's needed for production: without it,
`w_{jk}` is not interpretable because the pair terms have non-zero means
that all absorb into the bias term in a non-transparent way.

### Exposure

Exposure enters as a multiplicative weight on the deviance bracket:
```
L = (1/n) sum 2 * v_i * (mu_i - Y_i - Y_i * log(mu_i / Y_i))
```
where `v_i` is exposure (years at risk) and `Y_i = claims_i / v_i` is
frequency. This is the standard actuarial formulation, not a log offset.

## Exact Shapley Values

Because PIN is pairwise additive, Shapley values are exact at cost 2(q+1)
forward passes per sample per background sample. For q=9 and 200 background
samples, decomposing 100 predictions takes ~20 seconds.

The values are on the linear predictor scale. Exponentiating gives
multiplicative relativities, which is the natural language of insurance rating.

## Comparison with Other Libraries

| Library | Architecture | Use when |
|---------|-------------|----------|
| [insurance-ebm](https://github.com/burning-cost/insurance-ebm) | Cyclic boosted trees (GA2M) | You need auditable lookup tables |
| [insurance-anam](https://github.com/burning-cost/insurance-anam) | Neural additive model with monotonicity | You need monotone constraints |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Post-hoc interaction detection (NID) | You want to rank interactions in an existing model |
| **insurance-pin** | Shared-weight neural GA2M | You want best predictive performance with full decomposability |

## Supported Loss Functions

- `poisson` — Poisson deviance (frequency modelling)
- `gamma` — Gamma deviance (severity modelling)
- `tweedie` — Tweedie deviance, 1 < p < 2 (pure premium)

## Performance

Benchmarked on French MTPL (freMTPL2freq, 610k policies, out-of-sample Poisson deviance x10^-2). Results from Richman, Scognamiglio & Wüthrich (arXiv:2508.15678, 2025):

| Model | Deviance (x10^-2) | Parameters |
|-------|-------------------|------------|
| Null model | 25.445 | — |
| Poisson GLM | 24.102 | — |
| Poisson GAM | 23.956 | — |
| Ensemble FNN | 23.783 | — |
| Ensemble CAFFT | 23.726 | 27,133 |
| Ensemble Credibility TRM | 23.711 | 1,746 |
| **Ensemble PIN** | **23.667** | **4,147** |

PIN is the best published result on this benchmark. It achieves this with 4,147 parameters — fewer than the CAFFT model — because the shared interaction network reuses weights across all pairs rather than training a separate network per pair. The improvement over GLM is 0.435 units of deviance; at portfolio scale this translates to meaningfully better risk discrimination and pricing accuracy.


## License

MIT
