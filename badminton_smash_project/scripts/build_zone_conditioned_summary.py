"""
Build landing-zone-conditioned, player-weighted response summaries.

Input:
    results/shuttleset22_transitions.csv

Output:
    results/shuttleset22_zone_summary.csv

Each output row represents one (opponent_move, opp_landing_zone, response)
combination. Players qualify for an opponent-move/zone bucket after facing
that bucket at least eight times. A player's win score is included only when
that player used the response at least three times in the bucket.
"""

from pathlib import Path

import pandas as pd


MIN_PLAYER_OPP_ZONE_SAMPLES = 8
MIN_RESPONSE_ZONE_SAMPLES = 3
KNOWN_LANDING_ZONES = ("front", "mid", "back")
REQUIRED_COLUMNS = {
    "responder_name",
    "opponent_move",
    "response",
    "opp_landing_zone",
    "win_score",
}


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "shuttleset22_transitions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "shuttleset22_zone_summary.csv"


def build_zone_summary(transitions: pd.DataFrame) -> pd.DataFrame:
    """Return player-weighted response statistics for each landing-zone bucket."""
    missing_columns = REQUIRED_COLUMNS.difference(transitions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transitions CSV is missing required columns: {missing}")

    scoped = transitions[
        transitions["opp_landing_zone"].isin(KNOWN_LANDING_ZONES)
    ].copy()
    if scoped.empty:
        raise ValueError("No transitions use the known front/mid/back landing zones")

    scoped["win_score"] = pd.to_numeric(scoped["win_score"], errors="coerce")
    scoped = scoped.dropna(subset=["responder_name", "win_score"])
    if scoped.empty:
        raise ValueError("No valid transitions remain after cleaning win_score")

    bucket_columns = ["responder_name", "opponent_move", "opp_landing_zone"]
    pair_columns = bucket_columns + ["response"]

    player_totals = (
        scoped.groupby(bucket_columns)
        .size()
        .rename("player_opp_zone_total")
    )
    player_pairs = (
        scoped.groupby(pair_columns)
        .agg(
            player_pair_count=("win_score", "size"),
            player_pair_win_score=("win_score", "mean"),
        )
        .reset_index()
        .merge(player_totals, on=bucket_columns, how="left")
    )
    player_pairs["player_frequency"] = (
        player_pairs["player_pair_count"] / player_pairs["player_opp_zone_total"]
    )

    qualifying = (
        player_totals[player_totals >= MIN_PLAYER_OPP_ZONE_SAMPLES]
        .reset_index()[bucket_columns]
    )
    qualified_pairs = player_pairs.merge(qualifying, on=bucket_columns, how="inner")

    result_rows = []
    for (opponent_move, landing_zone), bucket in qualifying.groupby(
        ["opponent_move", "opp_landing_zone"], sort=True
    ):
        qualifying_players = bucket["responder_name"].unique()
        n_qualifying_players = len(qualifying_players)
        responses = qualified_pairs[
            (qualified_pairs["opponent_move"] == opponent_move)
            & (qualified_pairs["opp_landing_zone"] == landing_zone)
        ]

        for response, response_rows in responses.groupby("response", sort=True):
            frequencies = response_rows.set_index("responder_name")[
                "player_frequency"
            ].reindex(qualifying_players, fill_value=0.0)
            reliable = response_rows[
                response_rows["player_pair_count"] >= MIN_RESPONSE_ZONE_SAMPLES
            ]
            n_reliable_players = len(reliable)
            result_rows.append(
                {
                    "opponent_move": opponent_move,
                    "opp_landing_zone": landing_zone,
                    "response": response,
                    "n_qualifying_players": n_qualifying_players,
                    "n_players_using": int((frequencies > 0).sum()),
                    "pct_players_using": round(
                        frequencies.gt(0).mean() * 100, 1
                    ),
                    "avg_frequency_pct": round(frequencies.mean() * 100, 1),
                    "n_reliable_players": n_reliable_players,
                    "avg_win_score_pct": (
                        round(reliable["player_pair_win_score"].mean() * 100, 1)
                        if n_reliable_players >= 2
                        else None
                    ),
                }
            )

    summary = pd.DataFrame(result_rows)
    if summary.empty:
        raise ValueError("No response cells met the player qualification threshold")

    summary["optimal_response"] = False
    eligible = summary[summary["avg_win_score_pct"].notna()]
    for _, group in eligible.groupby(
        ["opponent_move", "opp_landing_zone"], sort=False
    ):
        best_index = group["avg_win_score_pct"].idxmax()
        summary.loc[best_index, "optimal_response"] = True

    return summary.sort_values(
        ["opponent_move", "opp_landing_zone", "avg_win_score_pct", "response"],
        ascending=[True, True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    transitions = pd.read_csv(DEFAULT_INPUT)
    summary = build_zone_summary(transitions)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(DEFAULT_OUTPUT, index=False)

    print(
        f"Loaded {len(transitions):,} transitions; "
        f"used {len(summary):,} qualified response cells."
    )
    print(f"Saved: {DEFAULT_OUTPUT}")

    smash = summary[summary["opponent_move"] == "smash"]
    if not smash.empty:
        print("\nSMASH RESPONSES BY OPPONENT LANDING ZONE")
        print(smash.to_string(index=False))


if __name__ == "__main__":
    main()
