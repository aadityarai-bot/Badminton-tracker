# ARCHITECTURE
## Badminton Smash-Response System — Data, Pipeline & Model Design

---

## 1. System overview

```
Raw CSVs (ShuttleSet22)
        │
        ▼
  build_shuttleset22_relational.py
        │
        ├──► shuttleset22_master.csv        ← one row per stroke, all context
        ├──► shuttleset22_transitions.csv   ← one row per opponent→response pair
        └──► shuttleset22_cross_player_summary.csv   ← model-facing freq + win-score
                          │
                          ▼
              build_model_ready_dataset.py
                          │
                          ▼
              model_ready_dataset.csv        ← leakage-safe training data
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
   train_baseline_model.py   train_tensorflow_model.py
   (Random Forest)           (Embedding NN)
```

---

## 2. Raw data schema — ShuttleSet22 set*.csv

Every row is one stroke in a rally. Columns as annotated by Wang et al.:

| Column | Type | Meaning |
|---|---|---|
| `rally` | int | Rally number within the set |
| `ball_round` | int | Stroke number within the rally (1 = first stroke) |
| `time` | str | Timestamp in the match video (MM:SS) |
| `frame_num` | int | Video frame number |
| `roundscore_A` | int | Player A's score at this point |
| `roundscore_B` | int | Player B's score at this point |
| `player` | str | Who played this stroke: `A` (match winner) or `B` (match loser) |
| `server` | int | 1 = Player A serves; 2 = Player B serves |
| `type` | str | Shot type in Chinese (18 categories — see translation table below) |
| `aroundhead` | float | 1 if played around-the-head; NaN otherwise |
| `backhand` | float | 1 if backhand; NaN otherwise |
| `hit_height` | float | 1 = below net height, 2 = above net height |
| `hit_area` | float | Court zone where shuttle was hit (1–16 grid) |
| `hit_x` | float | X pixel coordinate of hit position |
| `hit_y` | float | Y pixel coordinate of hit position |
| `landing_area` | float | Court zone where shuttle landed (1–16 grid) |
| `landing_x` | float | X pixel coordinate of landing position |
| `landing_y` | float | Y pixel coordinate of landing position |
| `landing_height` | float | 1 = below net, 2 = above net (on landing) |
| `lose_reason` | str | Why the rally ended against this player (Chinese, final stroke only) |
| `win_reason` | str | Why the rally ended for this player (Chinese, final stroke only) |
| `getpoint_player` | str | `A` or `B` — who won this rally (final stroke only, NaN otherwise) |
| `flaw` | float | Annotated flaw flag (rarely used) |
| `player_location_area` | float | Court zone the responder occupied |
| `player_location_x` | float | X coordinate of responder's position |
| `player_location_y` | float | Y coordinate of responder's position |
| `opponent_location_area` | float | Court zone the opponent occupied |
| `opponent_location_x` | float | X coordinate of opponent's position |
| `opponent_location_y` | float | Y coordinate of opponent's position |

### Shot type translation table

| Chinese | English label |
|---|---|
| 殺球 | smash |
| 點扣 | wrist smash |
| 切球 | drop |
| 過度/渡切球 | passive drop |
| 長球 | clear |
| 後場抽平球 | back-court drive |
| 平球 | drive |
| 小平球 | driven flight |
| 挑球 | lob |
| 防守回挑 | defensive lob |
| 防守回抽 | defensive drive |
| 放小球 | net shot |
| 擋小球 | return net |
| 勾球 | cross-court net |
| 推球 | push |
| 撲球 | rush |
| 發短球 | short service |
| 發長球 | long service |
| 未知球種 | unknown |

### Court area grid → zone mapping

The dataset uses a 16-cell grid (areas 1–16). Simplified to 3 zones:

```
 Court (net at top, baseline at bottom, player's perspective):
 ┌───────────────────────────────┐
 │  1  │  2  │  3  │ ← FRONT    │  (net zone)
 ├─────┼─────┼─────┤            │
 │  4  │  5  │  6  │ ← MID      │  (service line area)
 ├─────┼─────┼─────┤            │
 │  7  │  8  │  9  │ ← BACK     │  (baseline area)
 └───────────────────────────────┘
  Areas 10–16: extended/side zones, mapped to nearest front/mid/back
```

### Null rates in ShuttleSet22

| Column | Null % | Reason |
|---|---|---|
| `hit_x`, `hit_y` | ~11% | Some service strokes not tracked |
| `landing_x`, `landing_y` | ~1.7% | Occasional annotation gap |
| `player_location_x/y` | ~1.6% | Same |
| `getpoint_player` | ~90.8% | Only filled on the final stroke of each rally |
| `lose_reason`, `win_reason` | ~90.8% | Same — rally-ending strokes only |

---

## 3. Engineered datasets

### 3a. shuttleset22_master.csv (52,356 rows × 30 cols)

One row per stroke. Cleans and enriches the raw CSV with:
- `shot_type` — English shot name
- `lose_reason_en`, `win_reason_en` — English outcome reasons
- `landing_zone`, `hit_zone`, `player_zone` — front/mid/back labels
- `aroundhead`, `backhand` — boolean (not float)
- `hit_height_label` — `below_net` / `above_net` / `unknown`
- `score_A`, `score_B` — integer scores (not float)
- `ends_rally` — boolean: is this the final stroke?
- `rally_winner_code` — `A` or `B`, broadcast to every row in the rally
- `player_name`, `tournament`, `round`, `year` — from match.csv

### 3b. shuttleset22_transitions.csv (42,381 rows × 26 cols)

One row per consecutive stroke pair within a rally.
Row i = "opponent played `opponent_move`, you responded with `response`."
Excludes serves and unknowns on either side.

Key columns:

| Column | Meaning |
|---|---|
| `opponent_move` | Shot type the opponent just played (English) |
| `response` | Shot type you played in response (English) |
| `opp_landing_zone` | Where the opponent's shot landed (front/mid/back) |
| `opp_hit_height` | Whether opponent's hit was above or below net height |
| `resp_player_zone` | Court zone you were in when you played the response |
| `resp_landing_zone` | Where your response landed |
| `resp_aroundhead` | Whether you played around-the-head |
| `resp_backhand` | Whether you played backhand |
| `score_A_before` | Score state when the opponent played their shot |
| `score_B_before` | Score state when the opponent played their shot |
| `rally_continues` | 1 if the rally went on after your response; 0 if it ended |
| `responder_won_point` | 1 if your response directly won the point |
| `responder_lost_point` | 1 if your response directly lost the point |
| `win_score` | **Primary label:** 1 = game continues or you won the point; 0 = you lost the point directly on this response |
| `responder_won_rally` | 1 if you eventually won the whole rally |

**win_score vs responder_won_rally — why they differ:**
- `win_score=1` for any response that keeps play alive, even if you
  later lose the rally. It measures whether THIS specific shot was a
  direct error or not — closest to "was this response tactically safe?"
- `responder_won_rally=1` only if you ultimately won all the points in
  the rally. This captures long-run effectiveness but blends in everything
  that happened in the 5–15 subsequent shots.
  Use `win_score` for shot-specific feedback; use `responder_won_rally`
  for broader tactical pattern analysis.

### 3c. shuttleset22_cross_player_summary.csv (186 rows × 9 cols)

One row per `(opponent_move, response)` pair. The final model-facing table.

| Column | Meaning |
|---|---|
| `opponent_move` | The shot the opponent played |
| `response` | The shot the responder played |
| `n_qualifying_players` | Players who faced this opponent move ≥ 8 times |
| `n_players_using` | Of those, how many used this specific response ≥ 1 time |
| `pct_players_using` | % of qualifying players who used this response ≥ 1 time |
| `avg_frequency_pct` | Average % of the time (across qualifying players) this response is chosen when facing this opponent move |
| `n_reliable_players` | Players who used this response ≥ 3 times (win-score is trustworthy for these) |
| `avg_win_score_pct` | Average `win_score` % across reliable players. `NaN` if fewer than 2 reliable players |
| `optimal_response` | `True` for the highest `avg_win_score_pct` response per opponent move (requiring ≥ 2 reliable players) |

**How to read a row:**
```
opponent_move=smash, response=defensive lob
  n_qualifying_players=35     → 35 pros faced smashes ≥ 8 times
  pct_players_using=91.4%     → 32 of those 35 players use a defensive lob in response to smashes
  avg_frequency_pct=5.8%      → on average, they use it 5.8% of the time when smashed at
  n_reliable_players=18       → 18 of those players used it ≥ 3 times (reliable win-score)
  avg_win_score_pct=92.2%     → averaged across those 18 players, 92.2% of the time
                                 a defensive lob in response to a smash either kept the
                                 rally going or won the point directly
  optimal_response=True       → this is the highest win-score response to a smash
```

### 3d. model_ready_dataset.csv (66,664 rows × 37 cols)

Built from the combined ShuttleSet + ShuttleSet22 corpus.
Adds two engineered numeric features using out-of-fold encoding:
- `freq_feature` — how often this response is chosen vs this opponent move,
  computed from OTHER matches only (no data leakage)
- `winrate_feature` — win rate for this response vs this opponent move,
  smoothed with Laplace k=15 toward the opponent-move average,
  computed from other matches only
- `n_samples_seen` — how many training-fold instances backed the above
- `opp_<shot>` × 15, `resp_<shot>` × 15 — one-hot encoded shot types
- `won_rally` — binary target: did the responder win the whole rally?

---

## 4. Model architecture — TensorFlow embedding network

**File:** `scripts/train_tensorflow_model.py`

### Why embeddings, not one-hot into a dense layer?

One-hot input to a large dense layer (15 × 2 = 30 binary inputs → 128 units)
wastes parameters and overfits. Embeddings learn a compact 6-dimensional
representation of each shot type, so the model can discover that "smash"
and "wrist smash" behave similarly without being told explicitly.

### Network diagram

```
opponent_move (int ID)              response (int ID)
        │                                   │
  Embedding(15 → 6)              Embedding(15 → 6)
        │                                   │
      Flatten                             Flatten
        │                                   │
        └───────────────┬───────────────────┘
                        │
               Concatenate ◄── numeric features (3 values:
                        │       freq_feature, winrate_feature,
                        │       n_samples_seen)
                        │
                   Dense(32, relu)
                        │
                   Dropout(0.3)
                        │
                   Dense(16, relu)
                        │
                   Dense(1, sigmoid)
                        │
                   won_rally prediction (probability 0–1)
```

### Training details

| Parameter | Value | Rationale |
|---|---|---|
| Embedding dim | 6 | Keeps model small; 15 shot types don't need large embeddings |
| Hidden layers | 32 → 16 | Small network prevents overfitting on 66K rows |
| Dropout | 0.3 | Standard regularisation for tabular networks |
| Optimizer | Adam, lr=1e-3 | Default; no tuning done yet |
| Loss | Binary crossentropy | Binary outcome (won_rally 0 or 1) |
| Batch size | 256 | Stable gradients on this dataset size |
| Epochs | 15 | Enough to converge without severe overfitting |
| Validation split | 10% of training fold | Early convergence check |

### Evaluation discipline

**GroupKFold(n_splits=5) grouped by match_id** — no match ever appears in
both train and test. This is non-negotiable. Rows from the same match are
highly correlated (same players, same physical conditions, same tactical
tendencies that day). Splitting by row rather than by match would create
severe data leakage and produce optimistically wrong AUC numbers.

### Current benchmark (as of Phase 5)

| Model | Mean AUC | Mean Accuracy |
|---|---|---|
| Random Forest (baseline) | 0.539 | 0.530 |
| TF Embedding NN | 0.534 | 0.530 |
| Coin-flip baseline | 0.500 | 0.500 |

Both models are statistically equivalent right now. This is expected and
correct — the inputs (shot category + aggregate win-rate stats) carry only
moderate predictive signal about individual rally outcomes. The gap between
models will only open up once Phase 7 (landing zone features) are added.

---

## 5. Feature roadmap — from Phase 7 onward

### Features already in shuttleset22_transitions.csv, not yet used by the model

| Feature | Column | Expected impact |
|---|---|---|
| Where opponent's shot landed | `opp_landing_zone` | HIGH — controls for smash quality |
| Where responder was standing | `resp_player_zone` | HIGH — harder to play certain shots from certain positions |
| Score state | `score_A_before`, `score_B_before` | MEDIUM — pressure affects shot selection |
| Backhand / aroundhead | `resp_backhand`, `resp_aroundhead` | MEDIUM — shot difficulty context |
| Where response landed | `resp_landing_zone` | HIGH — placement quality, not just shot type |

### How to add them to the TF model (Phase 8 update)

Add new input heads to the existing network:

```python
zone_input    = keras.Input(shape=(1,), name="opp_landing_zone_id")
zone_emb      = Embedding(4, 3)(zone_input)   # front/mid/back/unknown → 3-dim

player_zone_input = keras.Input(shape=(1,), name="resp_player_zone_id")
player_zone_emb   = Embedding(4, 3)(player_zone_input)

numeric_input = keras.Input(shape=(6,), name="numeric")
# numeric: freq_feature, winrate_feature, n_samples_seen,
#          score_A_before, score_B_before, rally_length_so_far

x = Concatenate()([opp_emb, resp_emb, zone_emb, player_zone_emb, numeric_input])
# rest of the network unchanged
```

---

## 6. File inventory

```
badminton_smash_project/
├── BUILD_PLAN.md                                ← this file's companion
├── ARCHITECTURE.md                              ← this file
├── README.md                                    ← project overview and findings
│
├── datasets/
│   ├── ShuttleSet/set/                          ← 44 matches, 2018–2021
│   │   ├── match.csv                            ← player names, tournament metadata
│   │   └── <match_folder>/set{1,2,3}.csv        ← stroke-level data
│   └── ShuttleSet22/set/                        ← 58 matches, 2021–2022 ← PRIMARY
│       ├── match.csv
│       └── <match_folder>/set{1,2,3}.csv
│
├── scripts/
│   ├── build_shuttleset22_relational.py         ← PRIMARY — builds all 3 output tables
│   ├── build_model_ready_dataset.py             ← ML-ready features with leakage control
│   ├── train_baseline_model.py                  ← Random Forest benchmark
│   ├── train_tensorflow_model.py                ← TF embedding NN
│   ├── analyze_smash_responses.py               ← early smash-only analysis
│   ├── build_response_matrix.py                 ← 15×15 frequency matrix
│   ├── build_winrate_matrix.py                  ← 15×15 win-rate matrix
│   └── build_cross_player_matrix.py             ← player-weighted cross-generalization
│
└── results/
    ├── shuttleset22_master.csv                  ← PRIMARY — 52,356 strokes
    ├── shuttleset22_transitions.csv             ← PRIMARY — 42,381 transitions
    ├── shuttleset22_cross_player_summary.csv    ← PRIMARY — 186-row model table
    ├── model_ready_dataset.csv                  ← ML training data
    ├── shot_response_matrix_overall.csv         ← 15×15 frequency matrix
    ├── shot_response_matrix_winrate.csv         ← 15×15 win-rate matrix
    ├── shot_response_matrix_samplecounts.csv    ← sample sizes per cell
    ├── shot_response_matrix_per_player.csv      ← per-player breakdown
    ├── cross_player_generalization_matrix.csv   ← earlier player-weighted version
    ├── smash_response_frequency_and_winrate.csv ← smash-specific summary
    └── smash_response_raw_pairs.csv             ← raw smash event pairs
```

---

## 7. Key design decisions and rationale

**Why player-weighted instead of event-weighted?**
Viktor Axelsen appears in 12 matches; a lower-ranked player in 2. Event-weighting
makes Axelsen's habits dominate the "average player" figure. Player-weighting
ensures each of the 35 players counts equally regardless of how many matches
they contributed. The number you want to say is "X% of players do this,"
not "X% of strokes are played this way."

**Why win_score instead of responder_won_rally as the primary label?**
`responder_won_rally` conflates the quality of this specific response with
everything that happened in the next 5–15 shots. A perfectly placed lob
response that leads to a later unforced error by the responder counts as
a "0" even though the lob was the right choice. `win_score` is tighter:
it measures whether THIS stroke directly cost a point. It's closer to
"was this response tactically sound?" which is the actual question.

**Why out-of-fold encoding for freq_feature / winrate_feature?**
If you compute "how often does a cross-court net win against a smash" using
all 42,381 rows and then train a model on those same rows with that as a
feature, the model has implicitly seen the test rows' outcomes during
feature construction. GroupKFold ensures each row's features are computed
only from other matches — the same discipline used in the evaluation itself.

**Why smoothing (Laplace k=15) on the win-rate feature?**
A cell with 2 observations showing 100% win rate is not reliably better than
a cell with 200 observations showing 92%. Smoothing blends the rare cell's
rate toward the opponent-move average, weighted by sample size. This prevents
the model from over-indexing on noise from sparse cells.
