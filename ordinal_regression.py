"""
CORA Ordinal Regression
───────────────────────
Reads judge_results.csv, fits an ordinal logistic regression,
extracts empirical weights, compares to current CORA α values.
"""

import pandas as pd
import numpy as np
import mord
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ── Load results ──────────────────────────────────────────────────
df = pd.read_csv("judge_results.csv")

# Convert ground truth tier to integer (0-4)
TIER_MAP = {"Tier 0": 0, "Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}
df["tier_int"] = df["ground_truth_tier"].map(TIER_MAP)

# ── Features (X) and target (y) ───────────────────────────────────
# X = your 6 raw dimension scores from cognitive_module
# y = ground truth tier (what tier was actually needed)

feature_cols = [
    "reasoning_depth",
    "domain_specificity",
    "code_complexity",
    "creative_demand",
    "precision_required",
    "structural_complexity",
]

X = df[feature_cols].values.astype(float)
y = df["tier_int"].values

# Normalize X to 0-1 range (divide by 100)
X = X / 100.0

# ── Fit ordinal logistic regression ───────────────────────────────
model = mord.LogisticIT()  # LogisticIT = ordinal logistic regression
model.fit(X, y)

# ── Extract coefficients ───────────────────────────────────────────
# These are the empirically optimal weights for your routing task
coefficients = model.coef_

print("\n" + "="*62)
print("  ORDINAL REGRESSION — EMPIRICAL WEIGHTS")
print("="*62)
print(f"\n  {'Dimension':<25} {'Coefficient':>12} {'Normalized':>12}")
print("  " + "-"*52)

# Normalize coefficients to sum to 1 (same scale as your weights)
coef_sum = sum(abs(c) for c in coefficients)
normalized = [abs(c) / coef_sum for c in coefficients]

for name, coef, norm in zip(feature_cols, coefficients, normalized):
    print(f"  {name:<25} {coef:>12.4f} {norm:>12.4f}")

# ── Compare to CORA current effective weights ──────────────────────
# Current effective weights = (w_i * alpha_i) / Z from complexity_score.py
CURRENT_EFFECTIVE = {
    "reasoning_depth":       0.335,
    "domain_specificity":    0.187,
    "code_complexity":       0.000,  # not in NeMo directly
    "creative_demand":       0.235,
    "precision_required":    0.048,
    "structural_complexity": 0.158,
}

print(f"\n  {'Dimension':<25} {'Current w_eff':>13} {'Empirical':>10} {'Divergence':>11} {'Status':>8}")
print("  " + "-"*70)

updates_needed = []
for name, norm in zip(feature_cols, normalized):
    current = CURRENT_EFFECTIVE.get(name, 0)
    if current > 0:
        divergence = abs(norm - current) / current * 100
        status = "[OK]" if divergence < 15 else "[UPDATE]"
        if divergence >= 15:
            updates_needed.append((name, current, norm, divergence))
    else:
        divergence = 0
        status = "—"
    print(f"  {name:<25} {current:>13.3f} {norm:>10.3f} {divergence:>10.1f}% {status:>8}")

# ── Cross-validation accuracy ──────────────────────────────────────
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"\n  5-fold CV accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

# ── What to update ────────────────────────────────────────────────
if updates_needed:
    print(f"\n  >> {len(updates_needed)} dimension(s) diverge > 15% -- update alpha values:")
    print()
    for name, current, empirical, div in updates_needed:
        # Back-calculate what α should be
        # effective_weight = (w_nvidia * alpha) / Z
        # alpha = (effective_weight * Z) / w_nvidia
        W_NVIDIA = {
            "reasoning_depth":       0.25,
            "domain_specificity":    0.15,
            "creative_demand":       0.35,
            "structural_complexity": 0.15,
            "precision_required":    0.05,
            "code_complexity":       0.05,
        }
        Z = 1.045
        w = W_NVIDIA.get(name, 0.05)
        new_alpha = (empirical * Z) / w
        print(f"  {name}: alpha {current:.2f} -> {new_alpha:.2f}  (divergence: {div:.1f}%)")
else:
    print("\n  [OK] All dimensions within 15% -- alpha values are well calibrated!")

print("\n" + "="*62)
