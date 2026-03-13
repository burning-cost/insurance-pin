# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-pin: Tree-like Pairwise Interaction Networks
# MAGIC
# MAGIC **PIN** is a neural GA2M — same additive structure as a GLM with interaction terms,
# MAGIC but each shape function is parameterised by a shared neural network keyed by
# MAGIC learned interaction tokens.
# MAGIC
# MAGIC **Paper:** Richman, Scognamiglio, Wüthrich. "Tree-like Pairwise Interaction Networks."
# MAGIC arXiv:2508.15678 (August 2025).
# MAGIC
# MAGIC **Key result:** Ensemble PIN achieves the best published out-of-sample Poisson deviance
# MAGIC on French MTPL (freMTPL2freq) — better than GAMs, FNNs, CAFFT, and Credibility TRM,
# MAGIC at 4,147 parameters.
# MAGIC
# MAGIC **Prediction formula:**
# MAGIC ```
# MAGIC f_PIN(x) = exp( sum_{j<=k} w_{jk} * h_{jk}(x) + b )
# MAGIC ```
# MAGIC where `h_{jk}(x) = centered_hard_sigmoid(f_theta(phi_j(x_j), phi_k(x_k), e_{jk}))`.

# COMMAND ----------

# MAGIC %pip install insurance-pin

# COMMAND ----------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_pin import PINModel, PINEnsemble, PINDiagnostics

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic French MTPL-like Data
# MAGIC
# MAGIC We simulate a dataset with the structure of freMTPL2freq:
# MAGIC - 7 continuous features (age, driving experience, bonus-malus, density, etc.)
# MAGIC - 2 categorical features (vehicle brand category, region)
# MAGIC - Poisson frequency target with exposure

# COMMAND ----------

def generate_mtpl_like_data(n: int, seed: int = 42) -> dict:
    """
    Generate synthetic motor insurance data inspired by freMTPL2freq.

    Features:
        age_driver   - driver age (continuous)
        bonus_malus  - bonus-malus coefficient (continuous, 50-350)
        density      - area population density (continuous)
        veh_age      - vehicle age in years (continuous)
        drive_age    - years with licence (continuous)
        veh_power    - vehicle power (continuous)
        longitude    - geographic longitude (continuous)
        veh_brand    - vehicle brand category (11 levels)
        region       - region code (22 levels)

    Target:
        frequency    - claims per unit exposure
        exposure     - years at risk
    """
    rng = np.random.default_rng(seed)

    n_brand = 11
    n_region = 22

    age_driver = rng.uniform(18, 85, n).astype(np.float32)
    bonus_malus = rng.lognormal(np.log(100), 0.4, n).clip(50, 350).astype(np.float32)
    density = rng.lognormal(5, 1.5, n).clip(1, 100000).astype(np.float32)
    veh_age = rng.uniform(0, 25, n).astype(np.float32)
    drive_age = (age_driver - 18 + rng.exponential(2, n)).clip(0).astype(np.float32)
    veh_power = rng.uniform(40, 250, n).astype(np.float32)
    longitude = rng.uniform(-5, 10, n).astype(np.float32)
    veh_brand = rng.integers(0, n_brand, n)
    region = rng.integers(0, n_region, n)

    exposure = rng.uniform(0.1, 1.0, n).astype(np.float32)

    # True frequency: younger + high BM + high density -> more claims
    log_mu = (
        -3.5
        - 0.008 * (age_driver - 40)
        + 0.005 * (bonus_malus - 100)
        + 0.1 * np.log1p(density)
        + 0.01 * veh_age
        - 0.005 * drive_age
        + 0.002 * veh_power
        # True interaction: young drivers in dense areas
        + 0.0003 * np.maximum(0, 35 - age_driver) * np.log1p(density)
    )
    mu_true = np.exp(log_mu)
    claims = rng.poisson(mu_true * exposure).astype(np.float32)
    frequency = claims / np.maximum(exposure, 1e-6)

    X = {
        "age_driver": age_driver,
        "bonus_malus": bonus_malus,
        "density": density,
        "veh_age": veh_age,
        "drive_age": drive_age,
        "veh_power": veh_power,
        "longitude": longitude,
        "veh_brand": veh_brand,
        "region": region,
    }
    return X, frequency, exposure, mu_true


n_train = 50000
n_test = 10000

X_train, y_train, exp_train, mu_train_true = generate_mtpl_like_data(n_train, seed=0)
X_test, y_test, exp_test, mu_test_true = generate_mtpl_like_data(n_test, seed=1)
X_bg = {k: v[:2000] for k, v in X_train.items()}  # Background for SHAP

print(f"Train: {n_train:,} | Test: {n_test:,}")
print(f"Mean frequency (train): {y_train.mean():.4f}")
print(f"Mean exposure (train):  {exp_train.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define PIN Model (Reference Config)
# MAGIC
# MAGIC The paper's reference config for 9 features:
# MAGIC - d=10 (embedding dim)
# MAGIC - d'=20 (continuous embedding hidden width)
# MAGIC - d0=10 (interaction token dim)
# MAGIC - d1=30, d2=20 (shared net layers)
# MAGIC - Total: 4,147 parameters

# COMMAND ----------

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

model = PINModel(
    features=features,
    embedding_dim=10,
    hidden_dim=20,
    token_dim=10,
    shared_dims=(30, 20),
    loss="poisson",
    lr=0.001,
    batch_size=128,
    max_epochs=300,
    patience=20,
    lr_patience=5,
    lr_factor=0.9,
    val_fraction=0.1,
    random_seed=42,
)

print(f"Parameters: {model.count_parameters():,}")
# Expected: ~4,147

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit the Model

# COMMAND ----------

model.fit(X_train, y_train, exposure=exp_train, verbose=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Predict and Evaluate

# COMMAND ----------

def poisson_deviance(mu, y, exposure):
    """Compute out-of-sample Poisson deviance."""
    import numpy as np
    mu = np.maximum(mu, 1e-8)
    y_safe = np.maximum(y, 1e-8)
    bracket = mu - y - np.where(y > 0, y * np.log(mu / y_safe), 0.0)
    return 2 * (exposure * bracket).mean()


pred_test = model.predict(X_test)
dev_test = poisson_deviance(pred_test, y_test, exp_test)
print(f"Single PIN - Test Poisson deviance: {dev_test:.6f}")
print(f"(Paper reference: 23.74 x 10^-2 for single run on freMTPL2freq)")

# Null model deviance (for context)
mu_null = np.full(n_test, (y_train * exp_train).sum() / exp_train.sum())
dev_null = poisson_deviance(mu_null, y_test, exp_test)
print(f"Null model deviance: {dev_null:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Diagnostics: What Is the Model Doing?

# COMMAND ----------

diag = PINDiagnostics(model)

# 5a. Training history
fig, ax = diag.plot_training_history()
display(fig)
plt.close()

# COMMAND ----------

# 5b. Interaction weight heatmap
fig, ax = diag.interaction_heatmap(figsize=(9, 8))
display(fig)
plt.close()

# COMMAND ----------

# 5c. Pair importance (range of w_jk * h_jk over background)
fig, ax, importance = diag.weighted_importance(X_bg, top_n=15)
display(fig)
plt.close()

print("\nTop 10 pair contributions (range on background):")
for label, val in list(importance.items())[:10]:
    print(f"  {label}: {val:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Main Effect Curves

# COMMAND ----------

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for i, fname in enumerate(features.keys()):
    ax = axes[i // 3, i % 3]
    diag.plot_main_effect(fname, X_bg, n_grid=80, ax=ax)
    ax.set_title(fname)

plt.suptitle("PIN Main Effects (diagonal pair terms)", fontsize=14)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Interaction Surfaces
# MAGIC
# MAGIC The paper's top finding on French MTPL:
# MAGIC 1. BonusMalus x VehBrand (strongest interaction)
# MAGIC 2. BonusMalus x Region
# MAGIC 3. DrivAge x BonusMalus

# COMMAND ----------

# Plot the bonus_malus x age_driver interaction (analogue to paper's top pairs)
fig, ax = diag.plot_surface("bonus_malus", "age_driver", X_bg, n_grid=40)
display(fig)
plt.close()

# COMMAND ----------

fig, ax = diag.plot_surface("age_driver", "region", X_bg, n_grid=30)
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Exact Shapley Values
# MAGIC
# MAGIC Because PIN is pairwise additive, we compute EXACT Shapley values in
# MAGIC 2(q+1) forward passes per sample — no approximation.
# MAGIC
# MAGIC For q=9 features: 20 forward passes per sample.

# COMMAND ----------

X_explain = {k: v[:50] for k, v in X_test.items()}

shap_values = model.shapley_values(X_explain, X_bg, n_background=200)

print("SHAP values (first 5 samples, linear predictor scale):")
for fname, vals in shap_values.items():
    print(f"  {fname:15s}: mean={vals.mean():+.4f}, std={vals.std():.4f}")

# COMMAND ----------

# SHAP summary plot
fig, ax = plt.subplots(figsize=(8, 5))

feature_names = list(shap_values.keys())
mean_abs_shap = [np.abs(shap_values[f]).mean() for f in feature_names]

sorted_idx = np.argsort(mean_abs_shap)[::-1]

ax.barh(
    [feature_names[i] for i in sorted_idx[::-1]],
    [mean_abs_shap[i] for i in sorted_idx[::-1]],
    color="#2196F3",
)
ax.set_xlabel("Mean |SHAP value| (linear predictor scale)")
ax.set_title("Feature Importance via Exact PIN SHAP")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
display(fig)
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Forward Selection of Interactions
# MAGIC
# MAGIC The paper identifies important interactions by:
# MAGIC 1. Fitting a diagonal-only model (pure additive)
# MAGIC 2. Ranking off-diagonal pairs by validation deviance reduction
# MAGIC
# MAGIC Here we approximate by looking at |w_{jk}| for off-diagonal pairs after joint fitting.

# COMMAND ----------

weights = model.interaction_weights()
interaction_only = {
    pair: abs(w) for pair, w in weights.items()
    if pair[0] != pair[1]
}
sorted_interactions = sorted(interaction_only.items(), key=lambda x: x[1], reverse=True)

print("Top interaction pairs by |w_jk|:")
for (fj, fk), w in sorted_interactions[:10]:
    print(f"  {fj} x {fk}: |w| = {w:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Ensemble PIN
# MAGIC
# MAGIC The paper achieves best results with 10 models. Here we demonstrate with 3.

# COMMAND ----------

ensemble = PINEnsemble(
    n_models=3,
    features=features,
    embedding_dim=10,
    hidden_dim=20,
    token_dim=10,
    shared_dims=(30, 20),
    loss="poisson",
    lr=0.001,
    batch_size=128,
    max_epochs=200,
    patience=15,
    device=None,
)

ensemble.fit(X_train, y_train, exposure=exp_train, verbose=False)

# COMMAND ----------

ens_pred = ensemble.predict(X_test)
ens_std = ensemble.predict_std(X_test)

dev_ens = poisson_deviance(ens_pred, y_test, exp_test)
print(f"Ensemble PIN (3 models) - Test deviance: {dev_ens:.6f}")
print(f"Single model deviance:                   {dev_test:.6f}")
print(f"Improvement: {(dev_test - dev_ens) / dev_test * 100:.2f}%")

print(f"\nEpistemic uncertainty (std of predictions):")
print(f"  Mean std: {ens_std.mean():.4f}")
print(f"  Max std:  {ens_std.max():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Model Summary

# COMMAND ----------

diag.summary(X_bg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we built:**
# MAGIC - A neural GA2M with shared-weight architecture (4,147 params for 9 features)
# MAGIC - One shared network for all feature pairs, differentiated by learned interaction tokens
# MAGIC - Exact SHAP values at 2(q+1) forward passes — not an approximation
# MAGIC - Full diagnostics: interaction heatmaps, main effect curves, 2D surfaces
# MAGIC
# MAGIC **Why it matters for insurance pricing:**
# MAGIC - Same additive structure as a GLM with interaction terms — auditable
# MAGIC - Beats GAMs, FNNs, and TabNet-variants at a fraction of the parameters
# MAGIC - Interaction surfaces can be shown to a pricing team as 2D tables
# MAGIC - SHAP values decompose each prediction multiplicatively (via exp of sum)
# MAGIC
# MAGIC **Limitations:**
# MAGIC - No monotonicity constraints (unlike ANAM)
# MAGIC - No built-in smoothness penalty
# MAGIC - Inference is slower than EBM lookup tables
# MAGIC - Feature order doesn't affect output (by symmetry of tokens) but does affect surface labelling
