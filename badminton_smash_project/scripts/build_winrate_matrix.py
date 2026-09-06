"""
Win-Rate Matrix (companion to the frequency matrix)
====================================================
Same rows/columns as shot_response_matrix_overall.csv:
  Rows    = opponent's move
  Columns = your response
But the cell value here is different:
  cell = % of rallies WON by the responding player, when they answered
         that opponent move with that specific response.

Read together with the frequency matrix:
  - frequency matrix cell tells you HOW OFTEN a response is chosen
  - win-rate matrix cell tells you HOW WELL that response actually performs
A high-frequency / low-win-rate response = a habit worth questioning.
A low-frequency / high-win-rate response = an underused, effective option.

Cells with too few real samples to be trustworthy are left blank (NaN)
rather than shown as a misleadingly precise percentage.
"""

import pandas as pd
import glob
import os

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
    "/home/claude/CoachAI-Projects/ShuttleSet/set",
    "/home/claude/CoachAI-Projects/CoachAI-Challenge-IJCAI2023/ShuttleSet22/set",
]

EXCLUDE = {"unknown", "short service", "long service"}
MIN_SAMPLES_PER_CELL = 20  # below this, we don't trust the win-rate estimate


def load_all_rallies():
    frames = []
    for root in DATA_ROOTS:
        source = "ShuttleSet22" if "IJCAI2023" in root else "ShuttleSet"
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
            df["match_rally_id"] = (
                df["source"] + "|" + df["match_folder"] + "|" + df["set_file"] + "|" + df["rally"].astype(str)
            )
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

    for i in range(len(g) - 1):
        opp_move = g.loc[i, "type_en"]
        response_row = g.loc[i + 1]
        response = response_row["type_en"]
        responder = response_row["player"]
        if opp_move in EXCLUDE or response in EXCLUDE:
            continue
        records.append({
            "opponent_move": opp_move,
            "response": response,
            "won_rally": responder == rally_winner,
        })

pairs = pd.DataFrame(records)
print(f"Built {len(pairs):,} opponent-move -> response transitions with known rally outcomes.\n")

# ---------------------------------------------------------------------------
# Win-rate matrix: mean of won_rally, grouped by (opponent_move, response)
# ---------------------------------------------------------------------------
grouped = pairs.groupby(["opponent_move", "response"])["won_rally"].agg(["mean", "count"])
grouped["win_rate_pct"] = (grouped["mean"] * 100).round(1)

win_rate_matrix = grouped["win_rate_pct"].unstack("response")
sample_count_matrix = grouped["count"].unstack("response")

# mask cells with insufficient sample size
win_rate_matrix_reliable = win_rate_matrix.where(sample_count_matrix >= MIN_SAMPLES_PER_CELL)

# order rows/cols by overall frequency, same order as the frequency matrix, for easy side-by-side comparison
order = pairs["opponent_move"].value_counts().index.tolist()
rows_order = [r for r in order if r in win_rate_matrix_reliable.index]
cols_order = [c for c in order if c in win_rate_matrix_reliable.columns]
win_rate_matrix_reliable = win_rate_matrix_reliable.loc[rows_order, cols_order]
sample_count_matrix = sample_count_matrix.loc[rows_order, cols_order]

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

print("=" * 110)
print(f"WIN-RATE MATRIX (%). Rows = opponent's move, Columns = your response.")
print(f"Blank cells = fewer than {MIN_SAMPLES_PER_CELL} real samples, not reliable enough to report.")
print("=" * 110)
print(win_rate_matrix_reliable.to_string())

out_dir = "/home/claude/badminton_project"
win_rate_matrix_reliable.to_csv(f"{out_dir}/shot_response_matrix_winrate.csv")
sample_count_matrix.to_csv(f"{out_dir}/shot_response_matrix_samplecounts.csv")

# ---------------------------------------------------------------------------
# Combined view specifically for smash (the motivating case) as a sanity check
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SMASH ROW: frequency vs. win rate side by side")
print("=" * 70)
freq_matrix = pd.read_csv(f"{out_dir}/shot_response_matrix_overall.csv", index_col=0)
smash_freq = freq_matrix.loc["smash"]
smash_win = win_rate_matrix_reliable.loc["smash"]
smash_n = sample_count_matrix.loc["smash"]
combined = pd.DataFrame({
    "frequency_pct": (smash_freq * 100).round(1),
    "win_rate_pct": smash_win,
    "n_samples": smash_n,
}).dropna(subset=["win_rate_pct"]).sort_values("win_rate_pct", ascending=False)
print(combined.to_string())

print(f"\nSaved: {out_dir}/shot_response_matrix_winrate.csv")
print(f"Saved: {out_dir}/shot_response_matrix_samplecounts.csv  (sample size behind every win-rate cell, for transparency)")
