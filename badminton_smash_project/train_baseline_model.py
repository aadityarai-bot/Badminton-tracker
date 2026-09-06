"""
Baseline Model: does response choice predict who wins the rally?
===================================================================
Trains a simple classifier on model_ready_dataset.csv to sanity-check the
preprocessing pipeline. Evaluated with GroupKFold (grouped by match) so no
match ever appears in both train and test -- this mirrors how the features
themselves were built and gives an honest read on generalization.

This is intentionally a simple model (interpretable, fast) -- the goal here
is validating the FEATURE PIPELINE, not squeezing out maximum accuracy.
Phase 4 in BUILD_PLAN.md covers upgrading the model itself.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "results" / "model_ready_dataset.csv"
df = pd.read_csv(DATA_PATH)

feature_cols = ["freq_feature", "winrate_feature", "n_samples_seen"] + \
    [c for c in df.columns if c.startswith("opp_") or c.startswith("resp_")]

X = df[feature_cols]
y = df["won_rally"]
groups = df["match_id"]

gkf = GroupKFold(n_splits=5)
aucs, accs = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    aucs.append(auc)
    accs.append(acc)
    print(f"Fold {fold}: AUC = {auc:.3f}   Accuracy = {acc:.3f}")

print(f"\nMean AUC:      {sum(aucs)/len(aucs):.3f}")
print(f"Mean Accuracy: {sum(accs)/len(accs):.3f}")
print("(A coin-flip baseline would score ~0.50 on both. Anything meaningfully")
print(" above that means the response choice + its historical frequency/win-rate")
print(" carries real signal about who wins the rally.)")

# Fit one final model on everything to inspect feature importance
final_model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
final_model.fit(X, y)
importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10).to_string())
