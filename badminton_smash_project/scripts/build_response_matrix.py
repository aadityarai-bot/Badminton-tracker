"""
Relational Shot-Response Matrix
================================
Rows    = opponent's shot (the move that was just played against you)
Columns = your response shot
Cell    = P(you play this response | opponent just played this shot)
          i.e. if a player responds to a smash with another smash 100% of
          the time, cell [smash, smash] = 1.0. If 50% of the time, = 0.5.

Each ROW sums to 1.0 (it's a probability distribution over your possible
responses to that specific opponent shot).

Built from every real stroke-to-stroke transition in ShuttleSet + ShuttleSet22
(94 professional singles matches).
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

def load_match_name_maps():
    """Map (source, video_folder, 'A'/'B') -> real player name using match.csv,
    since the per-stroke data only labels players as A (match winner) / B (match loser)."""
    maps = {}
    for root in DATA_ROOTS:
        source = "ShuttleSet22" if "IJCAI2023" in root else "ShuttleSet"
        match_csv = os.path.join(root, "match.csv")
        mdf = pd.read_csv(match_csv)
        for _, row in mdf.iterrows():
            maps[(source, row["video"], "A")] = row["winner"]
            maps[(source, row["video"], "B")] = row["loser"]
    return maps

NAME_MAP = load_match_name_maps()

def load_all_rallies():
    frames = []
    for root in DATA_ROOTS:
        for filepath in glob.glob(os.path.join(root, "*", "set*.csv")):
            try:
                df = pd.read_csv(filepath)
            except Exception:
                continue
            if "type" not in df.columns or "rally" not in df.columns:
                continue
            df["match_folder"] = os.path.basename(os.path.dirname(filepath))
            df["set_file"] = os.path.basename(filepath)
            df["source"] = "ShuttleSet22" if "IJCAI2023" in filepath else "ShuttleSet"
            df["match_rally_id"] = (
                df["source"] + "|" + df["match_folder"] + "|" + df["set_file"] + "|" + df["rally"].astype(str)
            )
            source = df["source"].iloc[0]
            df["player_name"] = df["player"].map(
                lambda p, mf=df["match_folder"].iloc[0], src=source: NAME_MAP.get((src, mf, p), p)
            )
            df["player_global"] = df["source"] + "|" + df["match_folder"] + "|" + df["player_name"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True)

raw = load_all_rallies()
raw["type_en"] = raw["type"].map(SHOT_TRANSLATION).fillna("unknown")

# Shots to exclude from the matrix: serves (no "opponent move" precedes them)
# and unknown (unlabeled).
EXCLUDE = {"unknown", "short service", "long service"}

records = []
for rally_id, group in raw.groupby("match_rally_id", sort=False):
    g = group.sort_values("ball_round").reset_index(drop=True)
    for i in range(len(g) - 1):
        opp_move = g.loc[i, "type_en"]
        response = g.loc[i + 1, "type_en"]
        responder = g.loc[i + 1, "player_global"]
        if opp_move in EXCLUDE or response in EXCLUDE:
            continue
        records.append({"opponent_move": opp_move, "response": response, "player": responder})

pairs = pd.DataFrame(records)
print(f"Built {len(pairs):,} opponent-move -> response transitions.\n")

# ---------------------------------------------------------------------------
# OVERALL matrix: across all players pooled together
# ---------------------------------------------------------------------------
overall_matrix = pd.crosstab(pairs["opponent_move"], pairs["response"], normalize="index").round(3)

# order rows/cols by overall frequency for readability
order = pairs["opponent_move"].value_counts().index.tolist()
cols_order = [c for c in order if c in overall_matrix.columns]
rows_order = [r for r in order if r in overall_matrix.index]
overall_matrix = overall_matrix.loc[rows_order, cols_order]

print("=" * 100)
print("OVERALL RELATIONAL MATRIX  (rows = opponent's move, columns = your response, cell = probability)")
print("=" * 100)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(overall_matrix.to_string())

out_dir = "/home/claude/badminton_project"
overall_matrix.to_csv(f"{out_dir}/shot_response_matrix_overall.csv")

# ---------------------------------------------------------------------------
# PER-PLAYER matrices: e.g. "does THIS player respond to a smash with a smash?"
# Only keep players with enough data for meaningful percentages.
# ---------------------------------------------------------------------------
MIN_TRANSITIONS_PER_PLAYER = 150
player_counts = pairs["player"].value_counts()
qualified_players = player_counts[player_counts >= MIN_TRANSITIONS_PER_PLAYER].index

per_player_frames = []
for player in qualified_players:
    sub = pairs[pairs["player"] == player]
    mat = pd.crosstab(sub["opponent_move"], sub["response"], normalize="index").round(3)
    mat = mat.stack().rename("probability").reset_index()
    mat["player"] = player.split("|")[-1]  # just the player name
    mat["match"] = player.split("|")[1]
    per_player_frames.append(mat)

per_player_long = pd.concat(per_player_frames, ignore_index=True)
per_player_long = per_player_long.rename(columns={"opponent_move": "opponent_move", "response": "response"})
per_player_long = per_player_long[["player", "match", "opponent_move", "response", "probability"]]
per_player_long.to_csv(f"{out_dir}/shot_response_matrix_per_player.csv", index=False)

print(f"\nSaved: {out_dir}/shot_response_matrix_overall.csv  (pooled matrix, all players)")
print(f"Saved: {out_dir}/shot_response_matrix_per_player.csv  (long-format, one row per player/opponent-move/response)")
print(f"\n{len(qualified_players)} players had >= {MIN_TRANSITIONS_PER_PLAYER} logged transitions "
      f"and are included in the per-player file.")

# Example: show one specific player's response-to-smash row, if available
example = per_player_long[
    (per_player_long["opponent_move"] == "smash")
].sort_values(["player", "probability"], ascending=[True, False])
if len(example):
    print("\nExample -- how a few individual players respond specifically to a SMASH:")
    print(example.groupby("player").head(3).to_string(index=False))
