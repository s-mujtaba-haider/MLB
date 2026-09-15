"""Pick API + dashboard.

    uvicorn api.server:app --reload --port 8000

Endpoints
    GET /api/picks?date=YYYY-MM-DD   today's picks (or a given date)
    GET /api/markets                 per-market gate verdict and deployment
    GET /api/performance             realised results on graded picks
    GET /api/health                  liveness + freshness of the pick ledger
    GET /                            the dashboard

The dashboard is deliberately plain: the brief is that picks must actually
display and work end to end, with styling handled separately later.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlbedge import config as C  # noqa: E402
from mlbedge.picks import PICKS_PATH  # noqa: E402

app = FastAPI(title="MLB Edge", version="1.0")

MANIFEST = ROOT / "models" / "manifest.json"
GATE_CSV = C.REPORTS / "gate_results.csv"


def _clean(obj):
    """NaN/Inf are not valid JSON; convert to null rather than emitting NaN."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is pd.NaT:
        return None
    return obj


def _load_picks() -> pd.DataFrame:
    if not PICKS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PICKS_PATH)


@app.get("/api/health")
def health():
    p = _load_picks()
    last = p["generated_at"].max() if not p.empty else None
    dates = sorted(p["game_date"].unique().tolist()) if not p.empty else []
    return _clean({
        "status": "ok",
        "picks_total": len(p),
        "last_generated_at": last,
        "dates_covered": dates[-7:],
        "models_trained": MANIFEST.exists(),
        "now": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    })


@app.get("/api/markets")
def markets():
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text())
    else:
        man = {}
    rows = []
    for m in C.MARKETS:
        info = man.get(m.name, {})
        rows.append({
            "market": m.name, "label": m.label, "kind": m.kind,
            "verdict": info.get("verdict", "UNTESTED"),
            "deployment": info.get("deployment", "killed"),
            "threshold": info.get("threshold"),
            "cause": info.get("cause", ""),
            "metrics": info.get("metrics", {}),
        })
    return JSONResponse(_clean(rows))


@app.get("/api/picks")
def picks(date: str | None = Query(None), market: str | None = Query(None),
          min_confidence: float = Query(0.0), limit: int = Query(500)):
    p = _load_picks()
    if p.empty:
        return JSONResponse([])
    if date is None:
        date = p["game_date"].max()
    p = p[p["game_date"] == date]
    if market:
        p = p[p["market"] == market]
    if min_confidence:
        p = p[p["confidence"] >= min_confidence]
    p = p.sort_values("ev", ascending=False).head(limit)
    return JSONResponse(_clean(p.to_dict("records")))


@app.get("/api/performance")
def performance():
    """Realised results, on picks that have been graded."""
    p = _load_picks()
    if p.empty or "result" not in p.columns:
        return JSONResponse({"graded": 0,
                             "note": "no graded picks yet; run "
                                     "scripts/grade_picks.py after games finish"})
    g = p[p["result"].isin(["win", "loss", "push"])]
    if g.empty:
        return JSONResponse({"graded": 0})
    by = (g.groupby("market")
            .agg(picks=("profit", "size"), roi=("profit", "mean"),
                 profit=("profit", "sum"),
                 hit=("result", lambda s: (s == "win").mean()))
            .reset_index())
    return JSONResponse(_clean({
        "graded": int(len(g)),
        "roi": float(g["profit"].mean()),
        "profit": float(g["profit"].sum()),
        "by_market": by.to_dict("records"),
    }))


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8")
