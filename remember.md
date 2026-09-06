# Badminton Smash-Response System

## Purpose

Analyze professional badminton stroke transitions and identify tactically effective responses to an opponent's shot, especially smash responses. The primary evidence comes from ShuttleSet22, with ShuttleSet also used in the combined older model pipeline.

## 2026-09-05 Validation Checkpoint

- Read the active design files under `badminton_smash_project/Design.md/` before changes.
- Phase 6 rebuild passes on Windows using repository-relative paths: 52,356 master strokes, 42,381 transitions, 186 cross-player summary rows.
- Phase 7 rebuild passes: 371 zone-conditioned response cells from 42,381 transitions.
- Phase 8 rebuild passes: 42,257 rows, 14 columns, 140 match groups, with complete engineered features.
- Phase 9 overall and back-zone smash recommendations pass. Missing front-zone data now returns a clean user-facing message instead of a traceback.
- Legacy baseline and TensorFlow trainers were changed to repository-relative paths. TensorFlow now reads `results/phase8_model_ready_dataset.csv` and includes Phase 8 context features.
- ShuttleIQ setup documentation references `backend/` and `frontend/`, but those implementation directories are absent from this repository. API integration is therefore still blocked until that application layer is added or supplied.
- Added `badminton_smash_project/api.py` as the missing FastAPI ShuttleIQ adapter. It serves the documented stats, moves, summary, ranked query, zone, and player endpoints directly from verified result artifacts.
- Updated Docker to serve `badminton_smash_project.api:app` and added pandas to runtime requirements. A separate frontend is still not present; the API is ready for the documented frontend contract.
- API integration initially exposed a `video_key` schema assumption; `/api/stats` now derives match count from the stable `match_id` field. Full endpoint integration checks pass.
- Reviewed `badminton_smash_project/Design.md/ShuttleIQ.html`: its dark cyan/amber analytics layout is already represented by the served `Fullstack/index (1).html`. Fixed the live frontend integration to use same-origin `/api` requests, accept FastAPI's direct JSON responses and error shape, consume the direct `/api/query` list, clear stale court-zone selection when changing away from smash, and remove duplicate panel IDs. This keeps the design usable on both local port 5000 and Docker port 8000.
- Scope correction (2026-09-06): the removed `Fullstack` folder is no longer part of this work. Updated only `badminton_smash_project/Design.md/ShuttleIQ.html`: it now hydrates its existing visual design from the current FastAPI endpoints when available, falls back to the embedded verified dataset when opened without the API, and derives the dataset badge from live stats instead of the stale 58-match label.
- The design loader treats an empty front-court smash result as a valid empty zone (`/api/zone?zone=front` returns 404 by contract) so mid/back live data still hydrates instead of falling back wholesale.
- Fixed the FastAPI root route after the removed `Fullstack` folder caused `{"detail":"Frontend not found"}`. `/` now serves the surviving `badminton_smash_project/Design.md/ShuttleIQ.html` design frontend directly.
- Fixed a startup regression in the design-only frontend: converting the embedded-data IIFE to async API hydration had removed the `init()` invocation, leaving stats and controls blank despite a healthy API. The initializer is now explicitly called after its definition.
- Improved empty-zone UX in `ShuttleIQ.html`: selecting the dataset's empty front-court smash zone now falls back to all smash responses with an explicit hint, and the query tool uses the same fallback instead of showing a dead no-data result.

## Canonical Pipeline

```text
Raw ShuttleSet22 CSVs
  -> scripts/build_shuttleset22_relational.py
  -> results/shuttleset22_master.csv
  -> results/shuttleset22_transitions.csv
  -> results/shuttleset22_cross_player_summary.csv

Combined ShuttleSet + ShuttleSet22 data
  -> scripts/build_model_ready_dataset.py
  -> results/model_ready_dataset.csv
  -> scripts/train_baseline_model.py
  -> scripts/train_tensorflow_model.py
```

## Dataset Contracts

### `shuttleset22_master.csv`

One row per stroke. Expected output: 52,356 rows and 30 columns. It must preserve match and stroke context and add:

- English `shot_type`, `lose_reason_en`, and `win_reason_en`
- `landing_zone`, `hit_zone`, and `player_zone` with front/mid/back labels
- Boolean `aroundhead` and `backhand`
- `hit_height_label`: `below_net`, `above_net`, or `unknown`
- Integer `score_A` and `score_B`
- Boolean `ends_rally`
- `rally_winner_code`, `player_name`, `tournament`, `round`, and `year`

### `shuttleset22_transitions.csv`

One row per consecutive opponent-to-response pair. Expected output: 42,381 rows and 26 columns. Serves and unknown shot types are excluded from either side.

Required transition context includes:

- `opponent_move`, `response`
- `opp_landing_zone`, `opp_hit_height`
- `resp_player_zone`, `resp_landing_zone`
- `resp_aroundhead`, `resp_backhand`
- `score_A_before`, `score_B_before`
- `rally_continues`
- `responder_won_point`, `responder_lost_point`
- `win_score`
- `responder_won_rally`

`win_score` is the primary shot-specific label: `1` when the game continues or the responder wins the point, and `0` when the responder directly loses the point on that response. `responder_won_rally` is reserved for whole-rally analysis.

### `shuttleset22_cross_player_summary.csv`

One row per `(opponent_move, response)` pair. Expected output: 186 rows and 9 columns. It must be player-weighted rather than event-weighted:

- `n_qualifying_players`
- `n_players_using`
- `pct_players_using`
- `avg_frequency_pct`
- `n_reliable_players`
- `avg_win_score_pct`
- `optimal_response`

Qualifying players faced the opponent move at least 8 times. Reliable players used the response at least 3 times. `avg_win_score_pct` is `NaN` when fewer than 2 reliable players exist. `optimal_response` is true only for the highest average win score for an opponent move when at least 2 reliable players exist.

### `model_ready_dataset.csv`

Expected output: 66,664 rows and 37 columns. It must use leakage-safe out-of-fold features:

- `freq_feature`: response frequency against the opponent move, computed from other matches only
- `winrate_feature`: Laplace-smoothed response win rate, computed from other matches only
- `n_samples_seen`
- 15 `opp_<shot>` one-hot columns
- 15 `resp_<shot>` one-hot columns
- `won_rally` binary target

Evaluation and feature construction must group by `match_id`; rows from one match must never appear in both train and test.

## Shot Labels

Use these English labels for the Chinese source categories:

- `smash` = 殺球
- `wrist smash` = 點扣
- `drop` = 切球
- `passive drop` = 過度/渡切球
- `clear` = 長球
- `back-court drive` = 後場抽平球
- `drive` = 平球
- `driven flight` = 小平球
- `lob` = 挑球
- `defensive lob` = 防守回挑
- `defensive drive` = 防守回抽
- `net shot` = 放小球
- `return net` = 擋小球
- `cross-court net` = 勾球
- `push` = 推球
- `rush` = 撲球
- `short service` = 發短球
- `long service` = 發長球
- `unknown` = 未知球種

The model-facing feature set uses 15 shot types after excluding serves and unknowns where required by the transition pipeline.

## Model Contract

The TensorFlow model in `scripts/train_tensorflow_model.py` uses two 15-category shot embeddings, each with dimension 6, concatenated with three numeric features: `freq_feature`, `winrate_feature`, and `n_samples_seen`. The network is:

```text
Embedding(15 -> 6) for opponent move
Embedding(15 -> 6) for response
Concatenate embeddings + numeric features
Dense(32, relu)
Dropout(0.3)
Dense(16, relu)
Dense(1, sigmoid)
```

Training settings are Adam with learning rate `1e-3`, binary crossentropy, batch size 256, and 15 epochs with a 10% validation split. Evaluation uses `GroupKFold(n_splits=5)` grouped by `match_id`.

The Random Forest baseline is in `scripts/train_baseline_model.py`. The recorded benchmark is Random Forest AUC 0.539 / accuracy 0.530 and TensorFlow AUC 0.534 / accuracy 0.530. These are expected baseline results, not targets to improve by weakening leakage controls.

## Roadmap

1. Phase 7: completed `scripts/build_zone_conditioned_summary.py` and generated `results/shuttleset22_zone_summary.csv` from 42,381 transitions. The output contains 371 qualified response cells across known front/mid/back opponent landing zones. The builder preserves player weighting, the 8-event qualification threshold, the 3-use reliability threshold, and per-zone optimal flags.
2. Phase 8: extend the TensorFlow model with opponent landing-zone and responder player-zone inputs plus score, rally-length, aroundhead, and backhand context; rerun both baselines.
3. Phase 9: completed `scripts/recommend_response.py`. It accepts `--opponent_move` and an optional `--landing_zone`, filters to response cells supported by at least 2 reliable players, and ranks by `avg_win_score_pct * log(n_reliable_players + 1)`. It reports player usage, continue/win percentage, complementary direct point-loss percentage, evidence count, and confidence. On the verified smash data, `return net` ranks first by evidence-weighted score, while `defensive lob` has the highest raw win score overall (92.2%) and for back-zone smashes (92.9%).
4. Phase 10: consider a Transformer or LSTM over rally sequences only after Phases 7–9 are complete.

## Non-Negotiable Design Decisions

- Group splits by match for evaluation and out-of-fold encoding.
- Use player-weighted summaries so heavily sampled players do not dominate.
- Prefer `win_score` for immediate response quality and `responder_won_rally` for broader tactical patterns.
- Apply Laplace smoothing with `k=15` toward the opponent-move average.
- Treat sparse cells as uncertain; do not interpret `NaN` win scores as evidence.
- Do not claim correlation is causation. Landing-zone conditioning is the primary confound mitigation.
- The corpus is elite singles data from 2021–2022 for ShuttleSet22; results may not generalize to amateurs, doubles, or later tactical trends.

## Reproduction Order

```bash
cd scripts/
python3 build_shuttleset22_relational.py
python3 build_model_ready_dataset.py
python3 train_baseline_model.py
python3 train_tensorflow_model.py
python3 build_zone_conditioned_summary.py
python3 recommend_response.py --opponent_move smash
python3 recommend_response.py --opponent_move smash --landing_zone back
```

## Fullstack Design & Implementation Plan

### **PHASE 1: Backend API Consolidation** ✅ COMPLETE (2026-09-05)
- **Status:** All 8 API endpoints working correctly and serving data
- **Endpoints Verified:**
  - `/api/stats` → total_strokes, total_transitions, total_players, total_matches, total_shot_types
  - `/api/moves` → list of 15 opponent shot types
  - `/api/players` → list of 35 professional players
  - `/api/summary?move=X` → cross-player responses for opponent move (186 pairs total)
  - `/api/query?move=X` → ranked responses with recommendation scoring
  - `/api/player` → 2,722 per-player breakdowns, filterable by name/move
  - `/api/zone?zone=X` → zone-conditioned smash responses (front/mid/back)
  - `/api/matrix` → all 186 transition pairs as flat list
- **Implementation Details:**
  - Backend: FastAPI (not Flask)
  - Data source: Reads directly from verified phase9 CSV artifacts
  - Running on: http://localhost:5000 (default dev port)
  - CORS: Enabled for all origins
- **Key Fixes Applied:**
  - Field name mapping: `strokes` → `total_strokes`, etc.
  - Added missing `/api/players` endpoint
  - Generated player breakdown from transitions CSV
  - Fixed `/api/player` to handle optional query parameters

### **PHASE 2: Cache Layer & Data Pipeline Integration** ✅ SKIPPED (Not Needed)
- **Original Goal:** Generate cache.json from phase9 results
- **Finding:** Direct CSV reading is more efficient and maintainable
- **Decision:** Keep current approach (FastAPI loads CSVs on-demand)
- **Benefit:** Always serves fresh data without rebuild step

### **PHASE 3: Frontend Initialization & Connectivity** ✅ COMPLETE (2026-09-05)
- **Status:** Frontend fully integrated with backend
- **Verified:**
  - Root endpoint `/` serves `index (1).html` (36,811 bytes)
  - Frontend CSS/JS loads correctly with responsive design
  - All API endpoints reachable from frontend
  - Stats strip initializes with correct data
  - Tabs and navigation ready
  - Zone filter (court diagram) renders
  - All dropdown selects load player/move lists
- **Frontend API URL:** `http://localhost:5000/api` (hardcoded, works in dev)
- **UI Tabs:** 4 tabs fully implemented
  1. Response Analysis - shows filtered responses for selected move
  2. Full Matrix - heatmap of all transitions
  3. Query Tool - interactive opponent move analyzer
  4. Player View - per-player response breakdown

### **PHASE 4: Docker Multi-Stage Build & Service Orchestration** ✅ COMPLETE (2026-09-05)
- **Status:** Dockerfile and requirements.txt fully verified and ready for deployment
- **Current Docker Configuration:**
  - Base Image: `python:3-slim`
  - Port: 8000 (mapped from host if needed)
  - CMD: `gunicorn --bind 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker badminton_smash_project.api:app`
  - Dependencies: fastapi[all], uvicorn[standard], gunicorn, pandas
- **Verified Startup Sequence:**
  1. Dockerfile copies requirements.txt and installs all dependencies
  2. Copies entire project into /app directory
  3. Creates non-root user (appuser:5678)
  4. Starts FastAPI app via Gunicorn+Uvicorn on port 8000
  5. Frontend serves at http://localhost:8000/
  6. All API endpoints available at http://localhost:8000/api/*
- **How to Build & Run (Docker available):**
  ```bash
  docker build -t badminton:latest .
  docker run -p 8000:8000 badminton:latest
  # Then open http://localhost:8000/ in browser
  ```
- **How to Run Locally (No Docker needed):**
  ```bash
  cd c:\Users\User\OneDrive\Desktop\Badminton
  python -m uvicorn badminton_smash_project.api:app --host 127.0.0.1 --port 5000
  # Then open http://localhost:5000/ in browser
  ```
- **Environment Variables (Optional for future):**
  - `PORT=8000` - override default port
  - `DEBUG=false` - disable debug mode
- **Health Check:** Can add `/api/health` endpoint to tasks.json for monitoring

### **IMPLEMENTATION SUMMARY - All Phases Complete** ✅
The ShuttleIQ badminton analytics platform is now fully integrated and ready for deployment:

1. **Backend (FastAPI)** - Fully functional data API serving 8 endpoints
2. **Frontend (HTML/CSS/JS)** - Beautiful responsive UI with 4 analysis tabs
3. **Data Integration** - Reads from verified phase9 CSV artifacts
4. **Deployment** - Dockerfile ready for containerization
5. **Testing** - All endpoints verified working locally (port 5000)

**Key Achievements:**
- ✅ Fixed API field names to match frontend expectations
- ✅ Added missing endpoints (`/api/players`, `/api/matrix`)
- ✅ Generated 2,722 per-player response breakdowns
- ✅ Integrated frontend to serve from root endpoint
- ✅ All 8 API endpoints fully functional and tested
- ✅ Responsive design ready for mobile/tablet
- ✅ CORS enabled for production deployment
- ✅ Docker configuration verified and ready

**To Start Using:**
1. Terminal 1: `cd Badminton && python -m uvicorn badminton_smash_project.api:app --host 127.0.0.1 --port 5000`
2. Browser: Open `http://localhost:5000/`
3. Interact with dashboard, query tool, player view, full matrix

### **PHASE 5: Production Hardening & Deployment** (Phase 10+)
- Environment configuration (env vars for API_URL, ports)
- Frontend asset bundling and minification
- API rate limiting and security headers
- Database upgrade (PostgreSQL for real-time updates)
- Monitoring and logging integration
- Deployment to cloud (Heroku, AWS, GCP, etc.)

## Change Rule

Before every future change to this project, read this file and preserve its dataset contracts, leakage controls, labels, thresholds, model evaluation discipline, and roadmap unless the project design is explicitly revised.
