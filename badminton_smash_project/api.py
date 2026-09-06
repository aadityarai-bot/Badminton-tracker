from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
OVERALL_SUMMARY_PATH = RESULTS_DIR / "shuttleset22_cross_player_summary.csv"
ZONE_SUMMARY_PATH = RESULTS_DIR / "shuttleset22_zone_summary.csv"
MASTER_PATH = RESULTS_DIR / "shuttleset22_master.csv"
VALID_ZONES = {"front", "mid", "back"}
FRONTEND_FILE = PROJECT_ROOT / "Design.md" / "ShuttleIQ.html"

app = FastAPI(title="ShuttleIQ API", version="1.0.0")

# CORS configuration - must be added BEFORE other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to browser
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Data artifact is missing: {path.name}")
    return pd.read_csv(path)


def ranked(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["avg_win_score_pct"] = pd.to_numeric(result["avg_win_score_pct"], errors="coerce")
    result["n_reliable_players"] = pd.to_numeric(result["n_reliable_players"], errors="coerce")
    result = result.dropna(subset=["avg_win_score_pct"])
    result = result[result["n_reliable_players"] >= 2].copy()
    result["recommendation_score"] = result["avg_win_score_pct"] * (
        result["n_reliable_players"] + 1
    ).map(__import__("math").log)
    return result.sort_values(
        ["recommendation_score", "avg_win_score_pct"], ascending=False
    )


def records(summary: pd.DataFrame) -> list[dict]:
    return summary.astype(object).where(summary.notna(), None).to_dict(orient="records")


@app.get("/api/stats")
def stats() -> dict:
    master = load_csv(MASTER_PATH)
    transitions = load_csv(RESULTS_DIR / "shuttleset22_transitions.csv")
    # Count unique shot types (15 types used in transitions, excluding serves and unknown)
    shot_types = ["smash", "wrist smash", "drop", "passive drop", "clear", 
                  "back-court drive", "drive", "driven flight", "lob", 
                  "defensive lob", "defensive drive", "net shot", "return net", 
                  "cross-court net", "push", "rush"]
    return {
        "total_strokes": len(master),
        "total_transitions": len(transitions),
        "total_players": int(master["player_name"].nunique()),
        "total_matches": int(master["match_id"].str.split("|", n=1).str[0].nunique()),
        "total_shot_types": 15,
    }


@app.get("/api/moves")
def moves() -> list[str]:
    summary = load_csv(OVERALL_SUMMARY_PATH)
    return sorted(summary["opponent_move"].dropna().unique().tolist())


@app.get("/api/players")
def players() -> list[str]:
    """Return all unique player names from the dataset."""
    master = load_csv(MASTER_PATH)
    return sorted(master["player_name"].dropna().unique().tolist())


@app.get("/api/summary")
def summary(move: str = Query(..., min_length=1)) -> list[dict]:
    data = load_csv(OVERALL_SUMMARY_PATH)
    filtered = data[data["opponent_move"].eq(move)]
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No summary found for move: {move}")
    return records(filtered)


@app.get("/api/matrix")
def matrix() -> list[dict]:
    """Return all transition pairs as a flat list (equivalent to /api/summary without filtering)."""
    data = load_csv(OVERALL_SUMMARY_PATH)
    return records(data)


@app.get("/api/query")
def query(move: str = Query(..., min_length=1)) -> list[dict]:
    data = load_csv(OVERALL_SUMMARY_PATH)
    filtered = data[data["opponent_move"].eq(move)]
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No summary found for move: {move}")
    return records(ranked(filtered))


@app.get("/api/zone")
def zone(zone: str = Query(..., pattern="^(front|mid|back)$")) -> list[dict]:
    data = load_csv(ZONE_SUMMARY_PATH)
    filtered = data[
        data["opponent_move"].eq("smash") & data["opp_landing_zone"].eq(zone)
    ]
    if filtered.empty:
        raise HTTPException(status_code=404, detail=f"No smash data found for zone: {zone}")
    return records(ranked(filtered))


@app.get("/api/player")
def player(
    name: str = Query(None, min_length=1), move: str = Query(None, min_length=1)
) -> list[dict]:
    """
    Return per-player response breakdown by opponent move.
    - Without params: return all player breakdowns
    - With params: filter by player name and/or opponent move
    """
    transitions = load_csv(RESULTS_DIR / "shuttleset22_transitions.csv")
    
    # Group by player, opponent move, and response to get stats
    grouped = transitions.groupby(
        ["responder_name", "opponent_move", "response"], as_index=False
    ).agg({
        "win_score": ["sum", "count"],
    }).copy()
    
    # Flatten column names
    grouped.columns = ["responder_name", "opponent_move", "response", "wins", "total"]
    
    # Calculate win percentage and frequency
    grouped["ws_pct"] = (grouped["wins"] / grouped["total"] * 100).round(1)
    grouped["freq_pct"] = (grouped["total"] / len(transitions) * 100).round(1)
    grouped["n"] = grouped["total"]
    
    # Keep only relevant columns for frontend
    grouped = grouped[["responder_name", "opponent_move", "response", "ws_pct", "freq_pct", "n"]]
    
    # Filter by player name if provided
    if name:
        grouped = grouped[grouped["responder_name"].str.lower() == name.lower()]
    
    # Filter by opponent move if provided
    if move:
        grouped = grouped[grouped["opponent_move"].str.lower() == move.lower()]
    
    if grouped.empty and (name or move):
        raise HTTPException(status_code=404, detail=f"No player data found for filters")
    
    return records(grouped)


@app.get("/")
def root():
    """Serve the ShuttleIQ frontend."""
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_FILE, media_type="text/html")
