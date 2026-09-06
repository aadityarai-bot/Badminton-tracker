"""
TensorFlow Model: Response Win-Probability Predictor
======================================================
IMPORTANT FRAMING (read before extending this):
This model predicts P(responder wins the rally | opponent_move, response,
freq_feature, winrate_feature) for ONE instance at a time. It does NOT
directly output population statistics like "73% of players do X" -- that
number comes from aggregating over many rows (see
cross_player_generalization_matrix.csv), not from this model.

What this model IS good for: once you add per-instance context (shuttle
landing zone, player position, score situation -- Phase 2/3 in BUILD_PLAN.md),
it can estimate a win probability for a SPECIFIC situation that the raw
lookup tables don't have enough samples to cover individually. Right now,
without that extra context, it's mostly re-deriving what the tables already
say -- so this script exists mainly to (a) prove the architecture works and
(b) give you an honest baseline comparison against the Random Forest from
train_baseline_model.py before you invest more time in the NN.

Architecture: entity embeddings for the two categorical shot-type columns
(opponent_move, response), concatenated with the numeric engineered features,
then a small dense network. This is the standard approach for categorical
tabular data in TF/Keras -- much less prone to overfitting than raw one-hot
into a big dense layer, and lets the model discover which shot types behave
similarly to each other.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

tf.random.set_seed(42)
np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "results" / "phase8_model_ready_dataset.csv"
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# Encode categorical shot types as integer IDs (for embedding lookup)
# ---------------------------------------------------------------------------
opp_encoder = LabelEncoder()
resp_encoder = LabelEncoder()
df["opp_id"] = opp_encoder.fit_transform(df["opponent_move"])
df["resp_id"] = resp_encoder.fit_transform(df["response"])

n_opp_classes = df["opp_id"].nunique()
n_resp_classes = df["resp_id"].nunique()

numeric_cols = [
    "freq_feature",
    "winrate_feature",
    "n_samples_seen",
    "score_A_before",
    "score_B_before",
    "rally_length_so_far",
    "resp_aroundhead",
    "resp_backhand",
]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

y = df["won_rally"].values
groups = df["match_id"].values


def build_model():
    opp_input = keras.Input(shape=(1,), name="opp_id")
    resp_input = keras.Input(shape=(1,), name="resp_id")
    numeric_input = keras.Input(shape=(len(numeric_cols),), name="numeric")

    opp_emb = keras.layers.Embedding(n_opp_classes, 6)(opp_input)
    opp_emb = keras.layers.Flatten()(opp_emb)
    resp_emb = keras.layers.Embedding(n_resp_classes, 6)(resp_input)
    resp_emb = keras.layers.Flatten()(resp_emb)

    x = keras.layers.Concatenate()([opp_emb, resp_emb, numeric_input])
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[opp_input, resp_input, numeric_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# GroupKFold evaluation -- same discipline as the Random Forest baseline
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
aucs, accs = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups), 1):
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model()
    model.fit(
        {"opp_id": train_df["opp_id"], "resp_id": train_df["resp_id"], "numeric": train_df[numeric_cols]},
        y_train,
        epochs=15,
        batch_size=256,
        verbose=0,
        validation_split=0.1,
    )

    proba = model.predict(
        {"opp_id": test_df["opp_id"], "resp_id": test_df["resp_id"], "numeric": test_df[numeric_cols]},
        verbose=0,
    ).ravel()
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    aucs.append(auc)
    accs.append(acc)
    print(f"Fold {fold}: AUC = {auc:.3f}   Accuracy = {acc:.3f}")

print(f"\nTensorFlow model -- Mean AUC: {sum(aucs)/len(aucs):.3f}   Mean Accuracy: {sum(accs)/len(accs):.3f}")
print("Compare this to the Random Forest baseline (train_baseline_model.py): AUC ~0.539, Accuracy ~0.530")
print("\nIf the NN doesn't clearly beat the Random Forest here, that's an EXPECTED and honest result for")
print("data this size/shape -- it means the extra complexity isn't earning its keep YET. The real test")
print("is re-running this comparison after adding landing-zone/position context features (Phase 2).")
