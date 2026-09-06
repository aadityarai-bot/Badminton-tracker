"""
ShuttleSet22 → Cleaned Relational Dataset
==========================================
Produces three layered output tables:

  1. shuttleset22_master.csv          — one row per stroke, all context columns
                                        cleaned and translated to English
  2. shuttleset22_transitions.csv     — one row per opponent-move → response
                                        pair, with full context + outcome labels
  3. shuttleset22_cross_player_summary.csv — model-facing table:
                                        one row per (opponent_move, response)
                                        with cross-player frequency %,
                                        win_score %, and OPTIMAL flag marking
                                        the best response per opponent move

Win score (your spec):
  win_score = 1  → game continues OR responder won the point directly
  win_score = 0  → responder directly lost the point on this response
"""

import pandas as pd
import numpy as np
import glob, os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "datasets" / "ShuttleSet22" / "set"
OUT_DIR = PROJECT_ROOT / "results"

ROOT = str(DATA_ROOT)
OUT = str(OUT_DIR)
os.makedirs(OUT, exist_ok=True)

SHOT_EN = {
    "放小球":"net shot","擋小球":"return net","殺球":"smash","點扣":"wrist smash",
    "挑球":"lob","防守回挑":"defensive lob","長球":"clear","平球":"drive",
    "小平球":"driven flight","後場抽平球":"back-court drive","切球":"drop",
    "過度切球":"passive drop","過渡切球":"passive drop","推球":"push","撲球":"rush",
    "防守回抽":"defensive drive","勾球":"cross-court net","發短球":"short service",
    "發長球":"long service","未知球種":"unknown",
}
LOSE_EN = {
    "掛網":"net fault","對手落地致勝":"opponent winner","出界":"out of bounds",
    "落點判斷失誤":"misjudged landing","未過網":"failed to clear net",
}
WIN_EN = {
    "對手掛網":"opponent net fault","落地致勝":"winner","對手出界":"opponent out of bounds",
    "對手落點判斷失誤":"opponent misjudged","對手未過網":"opponent failed net",
}
AREA_ZONE = {
    1:"front",2:"front",3:"front",4:"mid",5:"mid",6:"mid",
    7:"back",8:"back",9:"back",10:"front",11:"mid",12:"back",
    13:"front",14:"mid",15:"back",16:"back",
}
EXCLUDE = {"unknown","short service","long service"}
MIN_PLAYER_OPP   = 8
MIN_RESP_USES    = 3

# ── 1. Match metadata ──────────────────────────────────────────────────────
meta = pd.read_csv(os.path.join(ROOT, "match.csv"))
meta["video"] = meta["video"].str.strip()
v2win  = dict(zip(meta["video"], meta["winner"]))
v2lose = dict(zip(meta["video"], meta["loser"]))
v2tour = dict(zip(meta["video"], meta["tournament"]))
v2rnd  = dict(zip(meta["video"], meta["round"]))
v2yr   = dict(zip(meta["video"], meta["year"]))

# ── 2. Load all CSVs ───────────────────────────────────────────────────────
print("Loading CSVs ...")
frames = []
for fp in sorted(glob.glob(os.path.join(ROOT, "*", "set*.csv"))):
    try: df = pd.read_csv(fp)
    except: continue
    if "type" not in df.columns: continue
    vk = os.path.basename(os.path.dirname(fp)).strip()
    df["match_id"]    = vk + "|" + os.path.basename(fp)
    df["video_key"]   = vk
    df["rally_id"]    = df["match_id"] + "|r" + df["rally"].astype(str)
    df["player_name"] = df["player"].map({"A": v2win.get(vk,"A"), "B": v2lose.get(vk,"B")})
    df["tournament"]  = v2tour.get(vk, "")
    df["round"]       = v2rnd.get(vk, "")
    df["year"]        = v2yr.get(vk, "")
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"  {len(raw):,} strokes | {raw['match_id'].nunique()} match-sets | "
      f"{raw['video_key'].nunique()} matches | {raw['player_name'].nunique()} players")

# ── 3. Clean & translate ───────────────────────────────────────────────────
raw["shot_type"]       = raw["type"].map(SHOT_EN).fillna("unknown")
raw["lose_reason_en"]  = raw["lose_reason"].map(LOSE_EN)
raw["win_reason_en"]   = raw["win_reason"].map(WIN_EN)
raw["landing_zone"]    = raw["landing_area"].map(lambda x: AREA_ZONE.get(int(x),"unk") if pd.notna(x) else "unk")
raw["hit_zone"]        = raw["hit_area"].map(lambda x: AREA_ZONE.get(int(x),"unk") if pd.notna(x) else "unk")
raw["player_zone"]     = raw["player_location_area"].map(lambda x: AREA_ZONE.get(int(x),"unk") if pd.notna(x) else "unk")
raw["score_A"]         = pd.to_numeric(raw["roundscore_A"], errors="coerce").fillna(0).astype(int)
raw["score_B"]         = pd.to_numeric(raw["roundscore_B"], errors="coerce").fillna(0).astype(int)
raw["aroundhead"]      = (raw["aroundhead"] == 1).astype(bool)
raw["backhand"]        = raw["backhand"].notna() & (raw["backhand"] == 1)
raw["hit_height_label"]= raw["hit_height"].map({1:"below_net",2:"above_net"}).fillna("unknown")
raw["ends_rally"]      = raw["getpoint_player"].notna()

# broadcast rally winner to every stroke
rw = raw[raw["ends_rally"]].drop_duplicates("rally_id").set_index("rally_id")["getpoint_player"].to_dict()
raw["rally_winner_code"] = raw["rally_id"].map(rw)

# ── 4. Master stroke table ─────────────────────────────────────────────────
MASTER_COLS = [
    "match_id","rally_id","ball_round","player","player_name",
    "tournament","round","year",
    "shot_type","aroundhead","backhand","hit_height_label",
    "hit_zone","hit_x","hit_y","landing_zone","landing_x","landing_y",
    "player_zone","player_location_x","player_location_y",
    "opponent_location_area","opponent_location_x","opponent_location_y",
    "score_A","score_B","lose_reason_en","win_reason_en",
    "ends_rally","rally_winner_code",
]
master = raw[[c for c in MASTER_COLS if c in raw.columns]].copy()
master.to_csv(f"{OUT}/shuttleset22_master.csv", index=False)
print(f"\nMaster: {len(master):,} rows × {master.shape[1]} cols  →  shuttleset22_master.csv")

# ── 5. Transition table ────────────────────────────────────────────────────
print("Building transition table ...")
rows = []
for rid, g in raw.groupby("rally_id", sort=False):
    g = g.sort_values("ball_round").reset_index(drop=True)
    rwc = rw.get(rid)
    for i in range(len(g) - 1):
        o, r = g.iloc[i], g.iloc[i+1]
        os_, rs_ = o["shot_type"], r["shot_type"]
        if os_ in EXCLUDE or rs_ in EXCLUDE: continue
        rc = r["player"]
        lost_pt  = r["ends_rally"] and r["getpoint_player"] != rc
        cont     = not r["ends_rally"]
        won_pt   = r["ends_rally"] and r["getpoint_player"] == rc
        rows.append({
            "match_id":o["match_id"],"rally_id":rid,"ball_round":r["ball_round"],
            "tournament":o["tournament"],"year":o["year"],
            "responder_name":r["player_name"],"responder_code":rc,
            "opponent_move":os_,"response":rs_,
            "opp_landing_zone":o["landing_zone"],"opp_landing_x":o["landing_x"],
            "opp_landing_y":o["landing_y"],"opp_hit_height":o["hit_height_label"],
            "resp_aroundhead":r["aroundhead"],"resp_backhand":r["backhand"],
            "resp_player_zone":r["player_zone"],
            "resp_landing_zone":r["landing_zone"],
            "resp_landing_x":r["landing_x"],"resp_landing_y":r["landing_y"],
            "score_A_before":o["score_A"],"score_B_before":o["score_B"],
            "rally_continues":int(cont),
            "responder_won_point":int(won_pt),
            "responder_lost_point":int(lost_pt),
            "win_score": 0 if lost_pt else 1,
            "responder_won_rally": int(rc == rwc) if rwc else None,
        })

trans = pd.DataFrame(rows)
trans.to_csv(f"{OUT}/shuttleset22_transitions.csv", index=False)
print(f"Transitions: {len(trans):,} rows × {trans.shape[1]} cols  →  shuttleset22_transitions.csv")

# ── 6. Cross-player summary ────────────────────────────────────────────────
print("Building cross-player frequency + win-score summary ...")

opp_totals  = trans.groupby(["responder_name","opponent_move"]).size().rename("opp_total")
pair_stats  = trans.groupby(["responder_name","opponent_move","response"]).agg(
    n_uses=("win_score","size"), ws_mean=("win_score","mean")).reset_index()
pair_stats  = pair_stats.merge(opp_totals, on=["responder_name","opponent_move"])
pair_stats["freq_pct"] = (pair_stats["n_uses"] / pair_stats["opp_total"] * 100).round(1)

qualifying  = opp_totals[opp_totals >= MIN_PLAYER_OPP].reset_index()[["responder_name","opponent_move"]]
qualified   = pair_stats.merge(qualifying, on=["responder_name","opponent_move"])

summary_rows = []
for opp_move, qdf in qualified.groupby("opponent_move"):
    q_players = qualifying[qualifying["opponent_move"]==opp_move]["responder_name"].unique()
    n_q = len(q_players)
    for response, rdf in qdf.groupby("response"):
        freqs = rdf.set_index("responder_name")["freq_pct"].reindex(q_players, fill_value=0.0)
        reliable = rdf[rdf["n_uses"] >= MIN_RESP_USES]
        n_rel    = len(reliable)
        summary_rows.append({
            "opponent_move":        opp_move,
            "response":             response,
            "n_qualifying_players": n_q,
            "n_players_using":      (freqs > 0).sum(),
            "pct_players_using":    round((freqs > 0).mean() * 100, 1),
            "avg_frequency_pct":    round(freqs.mean(), 1),
            "n_reliable_players":   n_rel,
            "avg_win_score_pct":    round(reliable["ws_mean"].mean() * 100, 1) if n_rel >= 2 else None,
        })

summary = pd.DataFrame(summary_rows).sort_values(
    ["opponent_move","pct_players_using"], ascending=[True,False])

# mark OPTIMAL: highest avg_win_score_pct per opponent_move (min 2 reliable players)
summary["optimal_response"] = False
for om, grp in summary[summary["avg_win_score_pct"].notna()].groupby("opponent_move"):
    summary.at[grp["avg_win_score_pct"].idxmax(), "optimal_response"] = True

summary.to_csv(f"{OUT}/shuttleset22_cross_player_summary.csv", index=False)
print(f"Summary:  {len(summary):,} rows × {summary.shape[1]} cols  →  shuttleset22_cross_player_summary.csv")

# ── 7. Print results ───────────────────────────────────────────────────────
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 15)

print("\n" + "═"*90)
print("SMASH ROW — how players respond to a smash (sorted by win score ↓)")
print("═"*90)
smash = summary[summary["opponent_move"]=="smash"].sort_values("avg_win_score_pct",ascending=False)
print(smash.to_string(index=False))

print("\n" + "═"*90)
print("OPTIMAL RESPONSE TABLE — highest-win-score response for every opponent move")
print("═"*90)
opt = summary[summary["optimal_response"]].sort_values("avg_win_score_pct",ascending=False)
print(opt[["opponent_move","response","pct_players_using","avg_frequency_pct",
           "avg_win_score_pct","n_reliable_players"]].to_string(index=False))
