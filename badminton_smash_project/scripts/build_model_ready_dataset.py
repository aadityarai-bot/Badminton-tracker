"""
Model-Ready Dataset Builder
============================
Turns the raw opponent-move -> response event log into a table a model can
actually learn from, by attaching two engineered numeric features to every
row:
  - freq_feature:    how often this exact response is chosen against this
                      exact opponent move (smoothed, out-of-fold)
  - winrate_feature: how often this exact response wins the rally when
                      played against this exact opponent move (smoothed,
                      out-of-fold)

Why "out-of-fold"?
  If we compute these stats using ALL the data (including the row itself),
  the model can partially "cheat" by memorizing outcomes instead of learning
  a generalizable pattern -- and evaluation numbers become misleadingly
  optimistic. So matches are split into folds; each row's features are
  computed only from the OTHER folds' matches.

Why "smoothed"?
  Rare (opponent_move, response) combos have few real examples. A raw
  win-rate of "2 out of 2 wins = 100%" is not trustworthy. Smoothing blends
  each cell toward the opponent move's overall average, weighted by sample
  size -- this is the same idea as Bayesian shrinkage / additive smoothing.

Output: results/model_ready_dataset.csv
  One row per real stroke-response event, with:
    match_id, opponent_move, response  (raw categorical fields)
    freq_feature, winrate_feature, n_samples_seen  (engineered numeric features)
    opp_<shot>, resp_<shot>  (one-hot encoded categorical columns)
    won_rally  (the target label)
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import GroupKFold

SHOT_TRANSLATION = {
    "放小球": "net shot", "擋小球": "return net", "殺球": "smash",
    "點扣": "wrist smash", "挑球": "lob", "防守回挑": "defensive return lob",
    "長球": "clear", "平球": "drive", "小平球": "driven flight",
    "後場抽平球": "back-court drive", "切球": "drop", "過渡切球": "passive drop",
    "推球": "push", "撲球": "rush", "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot", "發短球": "short service",
    "發長球": "long service", "未知球種": "unknown",
}

DATA_ROOTS = [
    "/home/claude/badminton_smash_project/datasets/ShuttleSet/set",
    "/home/claude/badminton_smash_project/datasets/ShuttleSet22/set",
]
EXCLUDE = {"unknown", "short service", "long service"}
N_FOLDS = 5
SMOOTHING_K = 15  # higher = trust the global average more when samples are scarce

# ---------------------------------------------------------------------------
# 1. Load every rally, build raw event table with a numeric match_id
# ---------------------------------------------------------------------------
def load_all_rallies():
    frames = []
    for root in DATA_ROOTS:
        source = "ShuttleSet22" if "ShuttleSet22" in root else "ShuttleSet"
        for filepath in glob.glob(os.path.join(root, "*", "set*.csv")):
            try:
                df = pd.read_csv(filepath)
            except Exception:
                continue
            if "type" not in df.columns or "rally" not in df.columns:
                continue
            df["match_folder"] = os.path.basename(os.path.dirname(filepath))
            df["set_file"] = os.path.basename(filepath)
            df["source"] = source
            df["match_key"] = df["source"] + "|" + df["match_folder"] + "|" + df["set_file"]
            df["match_rally_id"] = df["match_key"] + "|" + df["rally"].astype(str)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

raw = load_all_rallies()
raw["type_en"] = raw["type"].map(SHOT_TRANSLATION).fillna("unknown")

records = []
for rally_id, group in raw.groupby("match_rally_id", sort=False):
    g = group.sort_values("ball_round").reset_index(drop=True)
    winner_rows = g["getpoint_player"].dropna()
    rally_winner = winner_rows.iloc[-1] if len(winner_rows) else None
    if rally_winner is None:
        continue
    match_key = g["match_key"].iloc[0]
    for i in range(len(g) - 1):
        opp_move = g.loc[i, "type_en"]
        response_row = g.loc[i + 1]
        response = response_row["type_en"]
        responder = response_row["player"]
        if opp_move in EXCLUDE or response in EXCLUDE:
            continue
        records.append({
            "match_id": match_key,
            "opponent_move": opp_move,
            "response": response,
            "won_rally": int(responder == rally_winner),
        })

pairs = pd.DataFrame(records)
print(f"Loaded {len(pairs):,} events across {pairs['match_id'].nunique()} matches.\n")

# ---------------------------------------------------------------------------
# 2. Out-of-fold, smoothed target encoding for frequency + win rate
# ---------------------------------------------------------------------------
global_win_rate = pairs["won_rally"].mean()

gkf = GroupKFold(n_splits=N_FOLDS)
pairs["freq_feature"] = np.nan
pairs["winrate_feature"] = np.nan
pairs["n_samples_seen"] = np.nan

for train_idx, holdout_idx in gkf.split(pairs, groups=pairs["match_id"]):
    train_fold = pairs.iloc[train_idx]

    # frequency: P(response | opponent_move), computed from train_fold only
    opp_counts = train_fold.groupby("opponent_move").size()
    pair_counts = train_fold.groupby(["opponent_move", "response"]).size()
    freq_lookup = (pair_counts / opp_counts).to_dict()

    # win rate: P(win | opponent_move, response), smoothed toward
    # that opponent move's overall win rate, weighted by sample size
    opp_winrate = train_fold.groupby("opponent_move")["won_rally"].mean()
    pair_stats = train_fold.groupby(["opponent_move", "response"])["won_rally"].agg(["mean", "count"])

    holdout = pairs.iloc[holdout_idx]
    for idx, row in holdout.iterrows():
        key = (row["opponent_move"], row["response"])
        n = pair_counts.get(key, 0)
        pairs.at[idx, "n_samples_seen"] = n
        pairs.at[idx, "freq_feature"] = freq_lookup.get(key, 0.0)

        if key in pair_stats.index:
            raw_wr = pair_stats.loc[key, "mean"]
            n_pair = pair_stats.loc[key, "count"]
        else:
            raw_wr, n_pair = np.nan, 0
        prior = opp_winrate.get(row["opponent_move"], global_win_rate)
        # smoothing formula: blend raw win-rate toward the prior as n_pair -> 0
        smoothed = (n_pair * (raw_wr if not np.isnan(raw_wr) else prior) + SMOOTHING_K * prior) / (n_pair + SMOOTHING_K)
        pairs.at[idx, "winrate_feature"] = smoothed

print("Engineered features (freq_feature, winrate_feature) computed out-of-fold.\n")

# ---------------------------------------------------------------------------
# 3. One-hot encode the categorical shot types too, so the model can learn
#    patterns beyond just the aggregate stats (e.g. interactions with future
#    context features like landing zone, added in a later phase)
# ---------------------------------------------------------------------------
opp_dummies = pd.get_dummies(pairs["opponent_move"], prefix="opp")
resp_dummies = pd.get_dummies(pairs["response"], prefix="resp")

model_ready = pd.concat([
    pairs[["match_id", "opponent_move", "response", "freq_feature",
           "winrate_feature", "n_samples_seen", "won_rally"]],
    opp_dummies, resp_dummies,
], axis=1)

out_dir = "/home/claude/badminton_smash_project/results"
model_ready.to_csv(f"{out_dir}/model_ready_dataset.csv", index=False)
print(f"Saved: {out_dir}/model_ready_dataset.csv  ({model_ready.shape[0]:,} rows, {model_ready.shape[1]} columns)")
print("\nSample rows:")
print(model_ready[["opponent_move", "response", "freq_feature", "winrate_feature", "n_samples_seen", "won_rally"]].head(8).to_string(index=False))
