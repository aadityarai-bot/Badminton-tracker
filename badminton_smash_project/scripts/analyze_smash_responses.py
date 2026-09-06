"""
Smash-Response Analysis Model
==============================
Data source: ShuttleSet + ShuttleSet22 (Wang et al., KDD 2023 / IJCAI 2023)
Real stroke-by-stroke data from professional badminton singles matches
(Axelsen, Momota, Sindhu, Marin, An Se-young, Chen Yu Fei, etc.)

What this does:
1. Loads every rally from every match in both datasets.
2. Finds every moment a player is hit with a SMASH (or wrist smash).
3. Records what shot the receiving player used to respond.
4. Reports:
   - FREQUENCY: what % of the time players choose each response
   - WIN RATE: of the times a given response was used, what % of those
     rallies were eventually won by the responding player
   - An "optimal response" ranking based on win rate (not just popularity),
     since the most common response isn't necessarily the most effective one.
"""

import pandas as pd
import glob
import os

# ---------------------------------------------------------------------------
# 1. Shot type translation (from ShuttleSet README)
# ---------------------------------------------------------------------------
SHOT_TRANSLATION = {
    "放小球": "net shot",
    "擋小球": "return net",
    "殺球": "smash",
    "點扣": "wrist smash",
    "挑球": "lob",
    "防守回挑": "defensive return lob",
    "長球": "clear",
    "平球": "drive",
    "小平球": "driven flight",
    "後場抽平球": "back-court drive",
    "切球": "drop",
    "過渡切球": "passive drop",
    "推球": "push",
    "撲球": "rush",
    "防守回抽": "defensive return drive",
    "勾球": "cross-court net shot",
    "發短球": "short service",
    "發長球": "long service",
    "未知球種": "unknown",
}

SMASH_TYPES = {"smash", "wrist smash"}

# ---------------------------------------------------------------------------
# 2. Locate and load every set*.csv from both datasets
# ---------------------------------------------------------------------------
DATA_ROOTS = [
    "/home/claude/CoachAI-Projects/ShuttleSet/set",
    "/home/claude/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set",
]

def load_all_rallies():
    frames = []
    for root in DATA_ROOTS:
        pattern = os.path.join(root, "*", "set*.csv")
        for filepath in glob.glob(pattern):
            try:
                df = pd.read_csv(filepath)
            except Exception:
                continue
            if "type" not in df.columns or "rally" not in df.columns:
                continue
            df["match_folder"] = os.path.basename(os.path.dirname(filepath))
            df["set_file"] = os.path.basename(filepath)
            df["source"] = "ShuttleSet22" if "IJCAI2023" in filepath else "ShuttleSet"
            # unique match+set+rally id so rallies never collide across files
            df["match_rally_id"] = (
                df["source"] + "|" + df["match_folder"] + "|" + df["set_file"] + "|" + df["rally"].astype(str)
            )
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

print("Loading all rallies from ShuttleSet + ShuttleSet22 ...")
raw = load_all_rallies()
print(f"Loaded {len(raw):,} total strokes across {raw['match_rally_id'].nunique():,} rallies "
      f"from {raw['match_folder'].nunique()} matches.")

# translate shot types
raw["type_en"] = raw["type"].map(SHOT_TRANSLATION).fillna("unknown")

# ---------------------------------------------------------------------------
# 3. For each rally, walk stroke-by-stroke and capture (smash -> response) pairs
# ---------------------------------------------------------------------------
records = []

for rally_id, group in raw.groupby("match_rally_id", sort=False):
    g = group.sort_values("ball_round").reset_index(drop=True)
    # who won this rally (getpoint_player is filled on the final row)
    winner_rows = g["getpoint_player"].dropna()
    rally_winner = winner_rows.iloc[-1] if len(winner_rows) else None
    if rally_winner is None:
        continue

    for i in range(len(g) - 1):
        current_type = g.loc[i, "type_en"]
        if current_type not in SMASH_TYPES:
            continue
        response_row = g.loc[i + 1]
        response_type = response_row["type_en"]
        responding_player = response_row["player"]
        if response_type == "unknown":
            continue
        won_rally = (responding_player == rally_winner)
        records.append({
            "smash_variant": current_type,
            "response": response_type,
            "responder_won_rally": won_rally,
        })

pairs = pd.DataFrame(records)
print(f"\nFound {len(pairs):,} real instances of a player responding to a smash.")

# ---------------------------------------------------------------------------
# 4. Frequency table: how often is each response chosen after ANY smash?
# ---------------------------------------------------------------------------
freq = (
    pairs["response"]
    .value_counts(normalize=True)
    .mul(100)
    .round(1)
    .rename("frequency_pct")
)
counts = pairs["response"].value_counts().rename("n_instances")

# ---------------------------------------------------------------------------
# 5. Win-rate table: of the rallies where a player used this response,
#    what % did they go on to win?
# ---------------------------------------------------------------------------
win_rate = (
    pairs.groupby("response")["responder_won_rally"]
    .mean()
    .mul(100)
    .round(1)
    .rename("win_rate_pct")
)

summary = pd.concat([counts, freq, win_rate], axis=1).sort_values("frequency_pct", ascending=False)
summary.index.name = "response_shot"

# Only trust win-rate rankings for responses with a reasonable sample size
MIN_SAMPLE = 30
reliable = summary[summary["n_instances"] >= MIN_SAMPLE].copy()
reliable_ranked = reliable.sort_values("win_rate_pct", ascending=False)

print("\n" + "=" * 70)
print("HOW PLAYERS RESPOND TO A SMASH (real pro-match data)")
print("=" * 70)
print(summary.to_string())

print("\n" + "=" * 70)
print(f"OPTIMAL RESPONSE RANKING (win rate, min {MIN_SAMPLE} samples)")
print("=" * 70)
print(reliable_ranked.to_string())

if len(reliable_ranked):
    best = reliable_ranked.index[0]
    print(f"\n>>> Highest win-rate response to a smash: '{best}' "
          f"({reliable_ranked.loc[best, 'win_rate_pct']}% of rallies won, "
          f"used {reliable_ranked.loc[best,'frequency_pct']}% of the time)")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
out_dir = "/home/claude/badminton_project"
summary.to_csv(f"{out_dir}/smash_response_frequency_and_winrate.csv")
pairs.to_csv(f"{out_dir}/smash_response_raw_pairs.csv", index=False)
print(f"\nSaved: {out_dir}/smash_response_frequency_and_winrate.csv")
print(f"Saved: {out_dir}/smash_response_raw_pairs.csv  (raw event-level data, for training a model on)")
