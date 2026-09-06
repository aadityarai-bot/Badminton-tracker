# Badminton Smash-Response Analysis

**Goal:** Figure out, from real professional match data, what the optimal
response to a smash actually is — not just what pros habitually do, but what
statistically wins them the most rallies. Personal motivation: struggling to
return smashes in my own games (🎯 primary target: the `smash` row in every
matrix below).

## Data sources

Both datasets are stroke-by-stroke annotations of real professional singles
badminton matches, produced by academic researchers (Wang et al., KDD 2023 /
IJCAI 2023). Full copies are included in `datasets/`.

| Dataset | Matches | Rallies | Strokes | Players |
|---|---|---|---|---|
| **ShuttleSet** | 44 | 3,685 | 36,492 | 27 top-ranking singles players (2018–2021) |
| **ShuttleSet22** | ~50 | ~3,900 | ~33,600 | High-ranking 2022 matches (Axelsen, Sindhu, Marin, An Se-young, etc.) |

Source repo: https://github.com/wywyWang/CoachAI-Projects
Papers: ShuttleSet (arXiv:2306.04948), ShuttleSet22 (arXiv:2306.15664)

Each `set{1,2,3}.csv` file logs, per stroke: rally number, shot order,
player (A = eventual match winner, B = loser), **shot type** (18 possible —
smash, drop, clear, lob, net shot, etc.), landing coordinates, player
position, and — critically — **which player won the rally** (`getpoint_player`,
filled on the final stroke of each rally).

`match.csv` in each dataset maps the A/B labels back to real player names
(winner/loser columns), which is how the per-player breakdown resolves to
names like "Viktor AXELSEN" instead of just "A".

## What's been built so far

### 1. Smash-specific frequency + win-rate analysis
`scripts/analyze_smash_responses.py` → `results/smash_response_frequency_and_winrate.csv`

Filters the data down to every real instance (**8,451 events**) of a player
being smashed at, and reports how often each response shot is chosen vs. how
often that response actually wins the rally.

**Key finding:** players block back with a **return net 79.7%** of the time,
but it only wins **46.1%** of rallies. A **lob**, used just 0.5% of the time,
wins **56.1%**. Popularity ≠ effectiveness.

### 2. Full relational transition matrix (frequency)
`scripts/build_response_matrix.py` → `results/shot_response_matrix_overall.csv`
(+ `results/shot_response_matrix_per_player.csv` for per-player breakdowns)

A row-normalized probability matrix across **all 15 shot types** (not just
smash), built from **66,882 real stroke transitions**:

- Rows = opponent's move
- Columns = your response
- Cell = P(you play this response | opponent just played this move)
- Each row sums to 1.0

Example: row `smash` → column `return net` = **0.828**, meaning 82.8% of the
time, players respond to a smash with a block.

The per-player file breaks this down by named player and match, so you can
see individual tendencies (e.g., some players mix in a defensive lob against
a smash more than others).

### 3. Win-rate matrix (the effectiveness layer)
`scripts/build_winrate_matrix.py` → `results/shot_response_matrix_winrate.csv`
(+ `results/shot_response_matrix_samplecounts.csv` for transparency)

Same row/column structure as the frequency matrix, but each cell is instead
**% of rallies won** when that specific response was used against that
specific opponent move. Cells with fewer than 20 real samples are left blank
rather than shown as an unreliable percentage — `shot_response_matrix_samplecounts.csv`
shows exactly how many real events back every number.

**The smash row, frequency vs. win rate side by side:**

| Response to a smash | Frequency | Win rate | Sample size |
|---|---|---|---|
| Cross-court net shot | 2.9% | **54.1%** | 148 |
| Net shot | 1.0% | 50.0% | 48 |
| Return net (the default block) | 82.8% | 45.3% | 4,174 |
| Defensive return lob | 6.1% | 44.3% | 309 |
| Defensive return drive | 6.2% | 43.7% | 311 |

**Takeaway:** the reflexive block is the safe, high-frequency choice, but a
well-placed **cross-court net shot** — rare, but backed by 148 real
instances — statistically outperforms it.

**Caveat:** this is correlational, not causal. A player might only manage a
cross-court net shot when the incoming smash was already weaker or slower,
which would inflate its apparent win rate independent of the shot choice
itself. See `BUILD_PLAN.md` for how to control for this.

## Repository structure

```
badminton_smash_project/
├── README.md                  <- this file
├── BUILD_PLAN.md               <- roadmap for the next stages of the project
├── datasets/
│   ├── ShuttleSet/             <- full raw dataset + its README
│   └── ShuttleSet22/           <- full raw dataset + its README
├── scripts/
│   ├── analyze_smash_responses.py
│   ├── build_response_matrix.py
│   └── build_winrate_matrix.py
└── results/
    ├── smash_response_frequency_and_winrate.csv
    ├── smash_response_raw_pairs.csv        <- event-level data, ready for ML training
    ├── shot_response_matrix_overall.csv
    ├── shot_response_matrix_per_player.csv
    ├── shot_response_matrix_winrate.csv
    └── shot_response_matrix_samplecounts.csv
```

## How to reproduce

```bash
cd scripts
python3 analyze_smash_responses.py     # smash-focused frequency + win rate
python3 build_response_matrix.py       # full 15x15 frequency matrix
python3 build_winrate_matrix.py        # full 15x15 win-rate matrix
```

All three scripts read directly from `../datasets/ShuttleSet/set/` and
`../datasets/ShuttleSet22/set/` — adjust the `DATA_ROOTS` list at the top of
each script if you relocate the folders.

## Update: cross-player generalization (event-weighting fixed)

The matrices above were **event-weighted** — a heavily-sampled player's habits
could dominate the pooled average. `scripts/build_cross_player_matrix.py`
fixes this by computing stats per player first, then averaging **across
players equally** (each of the 43 named players counts once, regardless of
how many rallies they contributed). Players must have faced a given
opponent move 10+ times to qualify, and must have used a given response 3+
times for their personal win rate on it to count toward the average.

**Smash row, cross-player generalized:**

| Response | % of players using it | Avg. win rate across players |
|---|---|---|
| Return net | 100% (41/41 players) | 43.6% |
| Defensive return drive | 63.4% | 41.1% |
| Defensive return lob | 58.5% | 42.9% |
| **Cross-court net shot** | 34.1% (14/41 players) | **57.5%** |
| Net shot | 17.1% | 37.2% |

This confirms the earlier finding wasn't an artifact of one or two heavily
sampled players: the cross-court net shot's advantage holds up as a
genuine, shared pattern across 14 independent professional players, not a
statistical fluke.

Output: `results/cross_player_generalization_matrix.csv`
