"""
Rank evidence-backed responses to an opponent move.

This is a transparent statistical recommender, not a computer-vision model.
It uses player-weighted summaries so each qualifying player contributes once.

Examples:
    python recommend_response.py --opponent_move smash
    python recommend_response.py --opponent_move smash --landing_zone back
"""

import argparse
import math
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OVERALL_SUMMARY = PROJECT_ROOT / "results" / "shuttleset22_cross_player_summary.csv"
ZONE_SUMMARY = PROJECT_ROOT / "results" / "shuttleset22_zone_summary.csv"
VALID_ZONES = {"front", "mid", "back"}
REQUIRED_COLUMNS = {
    "opponent_move",
    "response",
    "n_qualifying_players",
    "n_players_using",
    "pct_players_using",
    "avg_frequency_pct",
    "n_reliable_players",
    "avg_win_score_pct",
}


def load_summary(opponent_move: str, landing_zone: str | None) -> pd.DataFrame:
    """Load the overall or landing-zone-conditioned summary."""
    path = ZONE_SUMMARY if landing_zone else OVERALL_SUMMARY
    if not path.exists():
        raise FileNotFoundError(
            f"Required summary not found: {path}. Run the Phase 6/7 builder first."
        )

    summary = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(
            f"Summary CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    filtered = summary[summary["opponent_move"].eq(opponent_move)].copy()
    if landing_zone:
        filtered = filtered[filtered["opp_landing_zone"].eq(landing_zone)]
    if filtered.empty:
        context = f"{opponent_move} ({landing_zone})" if landing_zone else opponent_move
        raise ValueError(f"No summary rows found for {context!r}")
    return filtered


def rank_responses(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank responses by win score weighted by the number of reliable players."""
    ranked = summary.copy()
    ranked["avg_win_score_pct"] = pd.to_numeric(
        ranked["avg_win_score_pct"], errors="coerce"
    )
    ranked["n_reliable_players"] = pd.to_numeric(
        ranked["n_reliable_players"], errors="coerce"
    )
    ranked = ranked.dropna(subset=["avg_win_score_pct", "n_reliable_players"])
    ranked = ranked[ranked["n_reliable_players"] >= 2].copy()
    ranked["recommendation_score"] = ranked["avg_win_score_pct"] * ranked[
        "n_reliable_players"
    ].map(lambda count: math.log(count + 1))
    return ranked.sort_values(
        ["recommendation_score", "avg_win_score_pct", "n_reliable_players"],
        ascending=False,
    ).reset_index(drop=True)


def confidence(n_reliable_players: int) -> str:
    """Label evidence strength without disguising sparse results as certainty."""
    if n_reliable_players >= 10:
        return "HIGH"
    if n_reliable_players >= 5:
        return "MEDIUM"
    return "LOW"


def print_recommendations(
    ranked: pd.DataFrame, opponent_move: str, landing_zone: str | None
) -> None:
    context = opponent_move.upper()
    if landing_zone:
        context += f" ({landing_zone}-court landing)"
    print(f"Opponent move: {context}")
    print("Rank  Response              Players  Use rate  Continue / win  Point lost  Confidence")
    print("-" * 86)

    for rank, row in ranked.iterrows():
        win_score = row["avg_win_score_pct"]
        print(
            f"{rank + 1:>4}  {row['response']:<20} "
            f"{int(row['n_reliable_players']):>7}  "
            f"{row['pct_players_using']:>7.1f}%  "
            f"{win_score:>14.1f}%  {100 - win_score:>10.1f}%  "
            f"{confidence(int(row['n_reliable_players']))}\n"
        )

    if ranked.empty:
        print("No response has at least two reliable players in this bucket.")
    else:
        print(
            "\nUse rate is the percentage of qualifying players who used the response. "
            "Continue / win is the cross-player average win_score."
        )
        print("Point lost is the complementary direct-loss rate, not whole-rally loss.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent_move", required=True, help="For example: smash")
    parser.add_argument(
        "--landing_zone",
        choices=sorted(VALID_ZONES),
        help="Optional opponent landing zone; uses the Phase 7 summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = load_summary(args.opponent_move, args.landing_zone)
    except (FileNotFoundError, ValueError) as error:
        print(f"No recommendation available: {error}")
        return
    ranked = rank_responses(summary)
    print_recommendations(ranked, args.opponent_move, args.landing_zone)


if __name__ == "__main__":
    main()
