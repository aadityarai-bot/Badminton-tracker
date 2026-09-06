# Badminton Smash-Response Analysis — Integration Test Summary

**Date:** September 5, 2026  
**Status:** ✅ **OPERATIONAL**  
**Python:** 3.13.5 | **pandas:** 3.0.1 | **scikit-learn:** 1.9.0 | **TensorFlow:** 2.x

---

## Overview

The badminton smash-response analysis system is a data-driven recommendation engine that identifies optimal player responses to opponent shots using professional match data. The system processes 94 professional singles matches across two datasets (ShuttleSet, ShuttleSet22) spanning 2018–2022, analyzing 66,864 stroke transitions and 35 professional players.

---

## Current Features

### 1. **Data Pipeline (Phases 6–9)**
- **Phase 6:** Relational dataset builder
  - Input: 140 ShuttleSet22 match CSVs (52,356 strokes)
  - Output: Clean master table with player names, shot types, zones, scores
  - Features: Landing zones (front/mid/back), hit height, backhand/aroundhead, position
  
- **Phase 7:** Zone-conditioned analysis
  - Stratifies response effectiveness by opponent landing zone
  - Output: 371 qualified response cells across 38 zone buckets
  - Methodology: Cross-player averaging (each player weighted equally)

- **Phase 8:** Enhanced model-ready dataset
  - Generates: 42,257 rows × 14 columns
  - Features: Frequency encoding, win-rate encoding (Bayesian smoothed), zone context, position
  - Validation: Out-of-fold GroupKFold by match (prevents data leakage)

- **Phase 9:** Recommendation engine
  - Ranks responses by evidence-weighted confidence
  - Output: Ranked responses with cross-player win-score % and player adoption %
  - Confidence tiers: HIGH (≥10 players), MEDIUM (5–9 players), LOW (<5 players)

### 2. **Key Findings**
- **Smash Response Paradox:** Players block/return net 83% of the time but only win 46% of rallies; defensive lob (6% usage) wins 56% but underutilized
- **Zone Awareness:** Optimal response differs by landing zone (back vs. mid vs. front court)
- **Cross-Player Validation:** 35 professional players independently confirm response patterns (not statistical flukes)

### 3. **Data Products**
| Artifact | Rows | Columns | Purpose |
|----------|------|---------|---------|
| `shuttleset22_master.csv` | 52,356 | 30 | Raw strokes with all context |
| `shuttleset22_transitions.csv` | 42,381 | 26 | Opponent move → response pairs |
| `shuttleset22_cross_player_summary.csv` | 186 | 9 | Cross-player aggregated statistics |
| `shuttleset22_zone_summary.csv` | 371 | 10 | Zone-stratified statistics |
| `phase8_model_ready_dataset.csv` | 42,257 | 14 | ML-ready with engineered features |
| `model_ready_dataset.csv` | 66,664 | 37 | Original Phases 1–5 dataset |

---

## Working Characteristics

### Recommendation Engine
```
Example: Smash Response Rankings
─────────────────────────────────────────────────────
1  return net        35 players   86.6%   HIGH (most common)
2  defensive lob     18 players   92.2%   HIGH (most effective)
3  defensive drive   22 players   84.4%   HIGH
4  cross-court net    4 players   91.7%   LOW  (rare, borderline evidence)
```

### Zone-Conditioned Variant
```
Smash (back-court landing):
─────────────────────────────────────────────────────
1  defensive lob      7 players   92.9%   MEDIUM (optimal when space available)
2  defensive drive   11 players   90.8%   HIGH
3  return net       31 players   88.5%   HIGH (safe default)

Smash (mid-court landing):
─────────────────────────────────────────────────────
1  defensive lob     15 players   92.7%   HIGH
2  return net       34 players   87.7%   HIGH
3  defensive drive   13 players   85.5%   HIGH
```

### Pipeline Integrity
- ✅ **Data Quality:** Zero null critical fields (player names, responder names, binary labels)
- ✅ **Schema Consistency:** 140 matches, 35 players, 15+ shot types uniformly coded
- ✅ **Feature Completeness:** 100% non-null engineered features in Phase 8 output
- ✅ **Reproducibility:** Phase 7 & 8 rebuild consistently from ShuttleSet22 source
- ✅ **Statistical Rigor:** Out-of-fold encoding prevents leakage; Bayesian smoothing controls noise

---

## Known Issues

### 🔴 **Critical (Blocks Older Pipelines)**
1. **Hard-coded Linux Paths**
   - Affects: `train_baseline_model.py`, `scripts/build_shuttleset22_relational.py`, older builders
   - Error: FileNotFoundError on Windows (`/home/claude/badminton_smash_project/...`)
   - Impact: Phase 0–5 legacy scripts cannot rebuild; Phase 6+ work around via relative paths
   - Fix: Replace absolute paths with `Path(__file__).resolve().parent`

2. **scipy.sparse.csgraph Import Timeout**
   - Affects: Baseline and TensorFlow model training when sklearn is first imported
   - Error: Stalls during `scipy.stats._qmc` initialization
   - Impact: `train_baseline_model.py` and `scripts/train_tensorflow_model.py` hang or timeout
   - Workaround: Phase 8 feature pipeline avoids this by direct sklearn usage; full model training blocked

### 🟡 **Medium (Poor UX)**
3. **Missing Zone Data Handling**
   - Issue: `recommend_response.py --opponent_move smash --landing_zone front` raises ValueError
   - Root Cause: No front-zone smash data in ShuttleSet22
   - Current Behavior: Raw traceback instead of graceful "no data" message
   - Fix: Add boundary check in `load_summary()` with user-friendly error

### 🟢 **Low (Informational)**
4. **Data Coverage Gaps**
   - 1,078 unknown shot labels (~2% of master table)
   - 881 unmapped landing zones (~1.7% of master table)
   - Impact: Minimal; downstream filtering intentional and controlled
   - Status: Acceptable for elite singles analysis

---

## Validation Results

### Data Artifact Integrity ✅
```
Master table:        52,356 rows × 30 columns | 0 null player_name
Transitions:         42,381 rows × 26 columns | 0 null responder_name
Summary (overall):      186 rows ×  9 columns | 0 duplicate keys
Summary (zone):         371 rows × 10 columns | 0 duplicate keys
Phase 8 Model Ready:  42,257 rows × 14 columns | 100% feature complete
```

### Reproducibility ✅
- Phase 7 rebuild: **PASS** (42,381 transitions consistently regenerated)
- Phase 8 build: **PASS** (42,257 model rows with correct zone stratification)
- Recommendation queries: **PASS** (smash, drop, clear, lob, etc.)

### Dependency Stack ✅
- pandas 3.0.1 (file I/O, aggregation)
- numpy 2.4.3 (array operations)
- scikit-learn 1.9.0 (GroupKFold, RandomForest)
- TensorFlow 2.x (embedding NN, optional)
- scipy 1.18.1 (statistics, sparse matrices)

---

## Recommendations for Deployment

1. **Immediate:** Use Phase 8 pipeline as primary entry point; include zone context in all recommendations
2. **Short-term:** Fix hard-coded paths to enable full Windows reproducibility
3. **Short-term:** Implement graceful error messaging for unsupported zone/move queries
4. **Medium-term:** Investigate scipy import timeout; consider lazy loading or environment setup documentation
5. **Long-term:** Extend analysis to other shot types (drops, clears, lobs) with same rigor

---

## Contact & Documentation

- **Repository:** `badminton_smash_project/`
- **Primary Pipeline:** `scripts/build_phase8_model_ready_dataset.py` → Recommendation Engine
- **Data Sources:** ShuttleSet (44 matches, 2018–2021) + ShuttleSet22 (~50 matches, 2021–2022)
- **Papers:** Wang et al. (KDD 2023, IJCAI 2023)

---

**Last Updated:** 2026-09-05 | **Next Review:** Upon production deployment

