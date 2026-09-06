"""
Build the Phase 8 leakage-safe model table from ShuttleSet22 transitions.

The target is responder_won_rally. Frequency and win-rate encodings are
computed from other match groups only. Context fields are retained for the
Random Forest and TensorFlow models:
opp_landing_zone, resp_player_zone, score_A_before, score_B_before,
rally_length_so_far, resp_aroundhead, and resp_backhand.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "results" / "shuttleset22_transitions.csv"
OUTPUT_PATH = PROJECT_ROOT / "results" / "phase8_model_ready_dataset.csv"
N_FOLDS = 5
SMOOTHING_K = 15

REQUIRED_COLUMNS = {
    "match_id",
    "opponent_move",
    "response",
    "opp_landing_zone",
    "resp_player_zone",
    "score_A_before",
    "score_B_before",
    "ball_round",
    "resp_aroundhead",
    "resp_backhand",
    "responder_won_rally",
}


def build_dataset(transitions: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(transitions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Transitions CSV is missing required columns: {missing}")

    data = transitions.dropna(subset=["responder_won_rally"]).copy()
    data["won_rally"] = data["responder_won_rally"].astype(int)
    data["rally_length_so_far"] = (
        pd.to_numeric(data["ball_round"], errors="coerce").fillna(1) - 1
    ).clip(lower=0)

    global_win_rate = data["won_rally"].mean()
    data["freq_feature"] = np.nan
    data["winrate_feature"] = np.nan
    data["n_samples_seen"] = np.nan

    groups = data["match_id"]
    if groups.nunique() < N_FOLDS:
        raise ValueError(f"Need at least {N_FOLDS} match groups for GroupKFold")

    group_kfold = GroupKFold(n_splits=N_FOLDS)
    for train_indices, holdout_indices in group_kfold.split(data, groups=groups):
        train_fold = data.iloc[train_indices]
        holdout = data.iloc[holdout_indices]

        opponent_counts = train_fold.groupby("opponent_move").size()
        pair_counts = train_fold.groupby(["opponent_move", "response"]).size()
        frequency_lookup = (pair_counts / opponent_counts).to_dict()
        opponent_win_rates = train_fold.groupby("opponent_move")["won_rally"].mean()
        pair_stats = train_fold.groupby(["opponent_move", "response"])[
            "won_rally"
        ].agg(["mean", "count"])

        for index, row in holdout.iterrows():
            key = (row["opponent_move"], row["response"])
            pair_sample_count = int(pair_counts.get(key, 0))
            prior = opponent_win_rates.get(row["opponent_move"], global_win_rate)
            if key in pair_stats.index:
                pair_mean = pair_stats.loc[key, "mean"]
                pair_count = pair_stats.loc[key, "count"]
            else:
                pair_mean = prior
                pair_count = 0

            data.at[index, "freq_feature"] = frequency_lookup.get(key, 0.0)
            data.at[index, "n_samples_seen"] = pair_sample_count
            data.at[index, "winrate_feature"] = (
                pair_count * pair_mean + SMOOTHING_K * prior
            ) / (pair_count + SMOOTHING_K)

    output_columns = [
        "match_id",
        "opponent_move",
        "response",
        "opp_landing_zone",
        "resp_player_zone",
        "score_A_before",
        "score_B_before",
        "rally_length_so_far",
        "resp_aroundhead",
        "resp_backhand",
        "freq_feature",
        "winrate_feature",
        "n_samples_seen",
        "won_rally",
    ]
    return data[output_columns]


def main() -> None:
    transitions = pd.read_csv(INPUT_PATH)
    dataset = build_dataset(transitions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {OUTPUT_PATH} ({len(dataset):,} rows, "
        f"{dataset.shape[1]} columns, {dataset['match_id'].nunique()} matches)"
    )
    print(dataset.head(5).to_string(index=False))


if __name__ == "__main__":
    main()