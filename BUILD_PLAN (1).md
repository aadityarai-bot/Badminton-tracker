perly # BUILD PLAN
## Badminton Smash-Response Analysis — From Raw Data to Model

---

## What is already done ✅

### Phase 0 — Data acquisition
- Sourced ShuttleSet + ShuttleSet22 from the CoachAI-Projects repository
  (Wang et al., KDD 2023 / IJCAI 2023)
- 94 professional singles matches, 35 named players (Axelsen, Momota,
  Sindhu, Marin, An Se-young, CHEN Yu Fei, LEE Zii Jia, etc.)
- Both datasets stored in `datasets/ShuttleSet/` and `datasets/ShuttleSet22/`

### Phase 1 — Initial analysis (smash-specific)
**Script:** `scripts/analyze_smash_responses.py`
**Output:** `results/smash_response_frequency_and_winrate.csv`

Isolated every smash → response event (8,451 instances) and measured:
- How often each response is chosen
- What % of those rallies the responder eventually won

**Finding:** players block (return net) 79.7% of the time but only win 46.1%
of those rallies. A lob or cross-court net shot is rarer but statistically
more effective.

### Phase 2 — Transition matrices (all 15 shot types)
**Scripts:** `scripts/build_response_matrix.py` + `scripts/build_winrate_matrix.py`
**Outputs:**
- `results/shot_response_matrix_overall.csv` — P(response | opponent_move), 15×15
- `results/shot_response_matrix_winrate.csv` — win rate per cell, same shape
- `results/shot_response_matrix_samplecounts.csv` — n per cell for transparency
- `results/shot_response_matrix_per_player.csv` — same, broken out per named player

Issue identified: both matrices were **event-weighted**. A heavily-sampled
player (e.g. someone in 15 matches) dominates the pooled average.

### Phase 3 — Cross-player generalization (event-weighting fixed)
**Script:** `scripts/build_cross_player_matrix.py`
**Output:** `results/cross_player_generalization_matrix.csv`

Each qualifying player counted exactly once regardless of matches played.
Qualification thresholds: must have faced the opponent move ≥ 10 times;
must have used a given response ≥ 3 times for their win rate to count.

**Smash finding (cross-player, 41 qualifying players):**
| Response | % of players using it | Avg win rate |
|---|---|---|
| Return net | 100% | 43.6% |
| Defensive return drive | 63.4% | 41.1% |
| Defensive return lob | 58.5% | 42.9% |
| Cross-court net shot | 34.1% | **57.5%** |

### Phase 4 — Model-ready dataset
**Script:** `scripts/build_model_ready_dataset.py`
**Output:** `results/model_ready_dataset.csv` (66,664 rows × 37 columns)

Converted raw events into ML-ready rows using:
- **Out-of-fold encoding** (GroupKFold by match): prevents data leakage where
  a row's own outcome inflates its frequency/win-rate features
- **Laplace/Bayesian smoothing** (k=15): blends rare cell win-rates toward
  the per-opponent-move average, preventing 2-sample cells from showing
  misleadingly extreme probabilities
- **One-hot encoding** of all 15 shot types for both opponent_move and response

Columns: `opponent_move, response, freq_feature, winrate_feature,
n_samples_seen, won_rally, opp_<shot>×15, resp_<shot>×15`

### Phase 5 — Baseline model validation
**Script:** `scripts/train_baseline_model.py`
**Result:** Random Forest AUC 0.539, Accuracy 0.530

**Script:** `scripts/train_tensorflow_model.py`
**Result:** TF embedding NN AUC 0.534, Accuracy 0.530

Both models confirm: `winrate_feature` and `freq_feature` are the top two
most predictive features. AUC ~0.54 is honest and expected — a single shot
exchange only explains a small fraction of rally outcomes. The bottleneck
is missing features (landing zone, position), not model architecture.

### Phase 6 — ShuttleSet22 relational dataset (most recent, primary output)
**Script:** `scripts/build_shuttleset22_relational.py`
**Inputs:** 140 set CSVs from `datasets/ShuttleSet22/set/`
**Outputs:**
- `results/shuttleset22_master.csv` (52,356 rows × 30 cols)
- `results/shuttleset22_transitions.csv` (42,381 rows × 26 cols)
- `results/shuttleset22_cross_player_summary.csv` (186 rows × 9 cols)

This is the cleanest, most accurate dataset. Key improvements over all
earlier outputs:
- Real player names resolved from match.csv (not A/B labels)
- Win score defined precisely: `win_score=1` if game continues or responder
  wins point; `win_score=0` if responder directly loses the point on that
  stroke. This is tighter than "won the whole rally" — it measures whether
  a specific response immediately costs a point.
- Court zones (front / mid / back) derived from area codes
- Hit height, aroundhead, backhand, score context all preserved

**Smash row from cross_player_summary (35 qualifying players):**
| Response | % players using | Avg win score | Optimal? |
|---|---|---|---|
| Defensive lob | 91.4% | **92.2%** | ✅ OPTIMAL |
| Cross-court net | 34.3% | 91.7% | |
| Return net | 100.0% | 86.6% | |
| Defensive drive | 82.9% | 84.4% | |

---

## What comes next ⬜

### Phase 7 — Landing-zone conditioned analysis
**Goal:** answer "given a smash landing in zone X, what is the optimal response?"
**How:**
- Use `opp_landing_zone` (front/mid/back) already in `shuttleset22_transitions.csv`
- Re-run the cross-player summary **within each landing zone bucket**
- This removes the confound where "rare responses only appear when the smash
  was already weak" — by fixing zone, you control for smash quality

**Expected challenge:** sample size thins out fast (42,381 transitions × 4
opponent shot types × 3 zones × 10 responses). May need to merge zones or
lower reliability thresholds with wider confidence intervals.

**Script to write:** `scripts/build_zone_conditioned_summary.py`
**Output:** `results/shuttleset22_zone_summary.csv`

### Phase 8 — Retrain TF model with zone + position features
**Goal:** push AUC above 0.54 by adding the landing-zone and player-zone
features that are already in `shuttleset22_transitions.csv` but not yet
used by the model.

**New feature set:**
- `opp_landing_zone` (categorical → embed or one-hot)
- `resp_player_zone` (categorical)
- `score_A_before`, `score_B_before` (numeric — pressure context)
- `resp_aroundhead`, `resp_backhand` (boolean — shot difficulty context)
- Existing: `freq_feature`, `winrate_feature` from cross-player summary

**Expected improvement:** significant, because landing zone is the main
missing confound identified in Phase 5's feature importance analysis.

**Re-run comparison:** always re-run both `train_baseline_model.py` and
`train_tensorflow_model.py` with the new features. If the TF model starts
pulling ahead of the Random Forest, that confirms the embeddings are now
earning their complexity.

**Script to update:** `scripts/train_tensorflow_model.py` — add the new
input heads (zone embedding, numeric context inputs)

### Phase 9 — Recommendation engine
**Goal:** replace two separate lookup tables with one unified ranked output:
given an opponent move (and optionally a landing zone), return a ranked list
of responses with their cross-player frequency %, win score %, and a
confidence tier (high / medium / low based on n_reliable_players).

**Logic:**
```
score = avg_win_score_pct × log(n_reliable_players + 1)
```
Penalizes high win-score from thin evidence (e.g., 100% from 2 players)
relative to a solid 92% from 18 players.

**Script to write:** `scripts/recommend_response.py`
**Usage example:**
```
python3 recommend_response.py --opponent_move smash --landing_zone back
```
**Output:**
```
Opponent move: SMASH (back-court landing)
─────────────────────────────────────────────────────────
Rank  Response          Players  Win Score  Confidence
  1   defensive lob       18      92.2%      HIGH
  2   cross-court net      4      91.7%      MEDIUM
  3   return net          35      86.6%      HIGH  (most common, not most effective)
```

### Phase 10 — Rally-sequence model (longer term)
**Goal:** rather than predicting one shot at a time, model the full rally
as a sequence and predict optimal play at each step given all prior shots.

**Architecture:** Transformer or LSTM over the sequence of
`(player, shot_type, landing_zone, player_zone)` tuples — similar to what
the ShuttleNet paper (IJCAI 2023) does, but with win-score supervision
rather than pure shot imitation.

**Prerequisite:** Phases 7–9 must be complete and working first. The
single-shot model is the sanity baseline; the sequence model extends it.

---

## Reproduction order

```bash
cd scripts/

# Phase 6 (primary — runs everything needed for the model)
python3 build_shuttleset22_relational.py

# Phase 4 + 5 (older pipeline, for comparison)
python3 build_model_ready_dataset.py
python3 train_baseline_model.py
python3 train_tensorflow_model.py

# Phase 7 (next to build)
python3 build_zone_conditioned_summary.py   # does not exist yet
```

## Known limitations

- **Correlation not causation:** a "rare but high win-score" response may
  correlate with easy incoming shots, not prove the response itself is the
  cause. Zone-conditioning (Phase 7) is the main mitigation.
- **Elite singles only:** all data is top-tier pro singles. Amateur and
  doubles patterns will differ.
- **Small samples in the tail:** many (opponent_move, response) combos have
  fewer than 3 reliable-player instances and are correctly excluded from
  win-score ranking. Don't read into cells marked `NaN`.
- **Static snapshot:** ShuttleSet22 covers 2021–2022. Tactical trends in
  professional badminton evolve; these numbers may shift as new data becomes
  available.
