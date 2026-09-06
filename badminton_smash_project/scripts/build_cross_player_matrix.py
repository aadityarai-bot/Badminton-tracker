"""
Cross-Player Generalization Matrix
====================================
Problem with the earlier matrices: they were EVENT-weighted. If Viktor
Axelsen appears in 15 matches and Sameer Verma in 2, Axelsen's habits
dominate the pooled average -- that's not "how players in general respond,"
it's "how heavily-sampled players respond."

This script fixes that by computing stats PER PLAYER first, then averaging
ACROSS players (each player counts once, regardless of how many rallies
they contributed). This directly answers questions like:

    "What % of players respond to a smash with another smash, and what's
     their average success rate when they do?"

Two qualification thresholds keep this honest:
  MIN_PLAYER_OPP_SAMPLES  = a player must have faced this opponent move at
                             least this many times to be counted at all
                             (otherwise their "0% or 100%" is just noise)
  MIN_RESPONSE_SAMPLES    = a player must have used a given response at
                             least this many times for THEIR win rate on
                             that response to be included in the average
                             (prevents one lucky/unlucky point from skewing
                             the cross-player average)

Output: results/cross_player_generalization_matrix.csv
  opponent_move, response, n_qualifying_players, n_players_using_response,
  pct_players_using, avg_frequency_across_players, avg_winrate_across_players
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
    "/home/claude/badminton_smash_project/datasets/ShuttleSet/set",
    "/home/claude/badminton_smash_project/datasets/ShuttleSet22/set",
]
EXCLUDE = {"unknown", "short service", "long service"}
MIN_PLAYER_OPP_SAMPLES = 10   # player must have faced this opponent move >= N times to qualify
MIN_RESPONSE_SAMPLES = 3      # player must have used the response >= N times for their win rate to count


def load_match_name_maps():
    maps = {}
    for root in DATA_ROOTS:
        source = "ShuttleSet22" if "ShuttleSet22" in root else "ShuttleSet"
        match_csv = os.path.join(os.path.dirname(root), "match.csv") if False else os.path.join(root, "match.csv")
        mdf = pd.read_csv(match_csv)
        for _, row in mdf.iterrows():
            maps[(source, row["video"], "A")] = row["winner"]
            maps[(source, row["video"], "B")] = row["loser"]
    return maps


NAME_MAP = load_match_name_maps()


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
            match_folder = os.path.basename(os.path.dirname(filepath))
            df["match_folder"] = match_folder
            df["set_file"] = os.path.basename(filepath)
            df["source"] = source
            df["match_rally_id"] = source + "|" + match_folder + "|" + df["set_file"] + "|" + df["rally"].astype(str)
            df["player_name"] = df["player"].map(lambda p: NAME_MAP.get((source, match_folder, p), p))
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
        responder_code = response_row["player"]
        responder_name = response_row["player_name"]
        if opp_move in EXCLUDE or response in EXCLUDE:
            continue
        records.append({
            "player": responder_name,
            "opponent_move": opp_move,
            "response": response,
            "won_rally": int(responder_code == rally_winner),
        })

events = pd.DataFrame(records)
print(f"Loaded {len(events):,} events across {events['player'].nunique()} distinct named players.\n")

# ---------------------------------------------------------------------------
# Step 1: per-player stats for every (opponent_move, response) they used
# ---------------------------------------------------------------------------
player_totals = events.groupby(["player", "opponent_move"]).size().rename("player_opp_total")
player_pair = events.groupby(["player", "opponent_move", "response"]).agg(
    player_pair_count=("won_rally", "size"),
    player_pair_winrate=("won_rally", "mean"),
).reset_index()

player_pair = player_pair.merge(player_totals, on=["player", "opponent_move"])
player_pair["player_frequency"] = player_pair["player_pair_count"] / player_pair["player_opp_total"]

# a player "qualifies" for an opponent_move if they faced it enough times overall
qualifying_players = player_totals[player_totals >= MIN_PLAYER_OPP_SAMPLES].reset_index()[["player", "opponent_move"]]
qualified = player_pair.merge(qualifying_players, on=["player", "opponent_move"])

# ---------------------------------------------------------------------------
# Step 2: cross-player aggregation -- each qualifying player counts ONCE
# ---------------------------------------------------------------------------
results = []
for (opp_move, response), sub in qualified.groupby(["opponent_move", "response"]):
    n_qualifying_players = qualifying_players[qualifying_players["opponent_move"] == opp_move]["player"].nunique()

    # every qualifying player contributes a frequency value for this response
    # (players who never used it at all get 0, pulled in via reindex)
    all_qualifying_for_move = qualifying_players[qualifying_players["opponent_move"] == opp_move]["player"]
    freqs = sub.set_index("player")["player_frequency"].reindex(all_qualifying_for_move, fill_value=0.0)
    avg_frequency = freqs.mean()

    users = sub[sub["player_pair_count"] >= MIN_RESPONSE_SAMPLES]
    n_players_using = users["player"].nunique()
    pct_players_using = 100 * n_players_using / n_qualifying_players if n_qualifying_players else float("nan")
    avg_winrate = users["player_pair_winrate"].mean() * 100 if len(users) else float("nan")

    results.append({
        "opponent_move": opp_move,
        "response": response,
        "n_qualifying_players": n_qualifying_players,
        "n_players_using_response": n_players_using,
        "pct_players_using": round(pct_players_using, 1) if pct_players_using == pct_players_using else None,
        "avg_frequency_across_players_pct": round(avg_frequency * 100, 1),
        "avg_winrate_across_players_pct": round(avg_winrate, 1) if avg_winrate == avg_winrate else None,
    })

cross_player = pd.DataFrame(results).sort_values(["opponent_move", "pct_players_using"], ascending=[True, False])

out_dir = "/home/claude/badminton_smash_project/results"
cross_player.to_csv(f"{out_dir}/cross_player_generalization_matrix.csv", index=False)

pd.set_option("display.width", 150)
print("=" * 100)
print("SMASH ROW -- cross-player generalized view (each player counted once, not each rally)")
print("=" * 100)
print(cross_player[cross_player["opponent_move"] == "smash"].to_string(index=False))

print(f"\nSaved: {out_dir}/cross_player_generalization_matrix.csv")
