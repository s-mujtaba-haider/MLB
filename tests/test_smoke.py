"""Every module imports, and the report renders.

A module that nothing else imports can be syntactically broken and still let
the suite pass. These tests exist so that cannot happen.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

MODULES = [
    "mlbedge.config", "mlbedge.oddsapi", "mlbedge.espn", "mlbedge.ingest_odds",
    "mlbedge.ingest_results", "mlbedge.normalize", "mlbedge.match",
    "mlbedge.grade", "mlbedge.odds", "mlbedge.devig", "mlbedge.features",
    "mlbedge.dataset", "mlbedge.model", "mlbedge.backtest", "mlbedge.leakage",
    "mlbedge.gate", "mlbedge.report", "mlbedge.production", "mlbedge.picks",
    "mlbedge.scheduler", "mlbedge.pipeline", "mlbedge.calibrate",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_every_market_is_fully_specified():
    from mlbedge import config as C
    assert len(C.MARKETS) == 11
    for m in C.MARKETS:
        assert m.kind in ("featured", "prop")
        assert m.side in ("batting", "pitching", "team")
        assert m.regions, f"{m.name} has no regions"
        assert m.label, f"{m.name} has no label"
        # Every market must be requestable from at least one region's plan.
        plans = (C.PROP_REQUEST_PLAN if m.kind == "prop"
                 else C.FEATURED_REQUEST_PLAN)
        assert any(m.api_key in v for v in plans.values()), \
            f"{m.name} is never requested by the ingestor"


def test_report_renders_for_pass_and_fail(tmp_path, rng):
    from conftest import efficient, real_edge
    from mlbedge import gate as G
    from mlbedge.report import write_market_report

    good = G.judge("batter_hits", real_edge(rng), closing=None)
    bad = G.judge("h2h", efficient(rng), closing=None)
    bets = pd.concat([real_edge(rng).assign(market="batter_hits"),
                      efficient(rng).assign(market="h2h")], ignore_index=True)
    out = tmp_path / "r.md"
    write_market_report([good, bad], bets, out, seasons=(2025,), folds=())
    text = out.read_text(encoding="utf-8")
    assert "batter_hits" in text and "h2h" in text
    assert "PASS" in text and "FAIL" in text
    # A failure must carry its lever into the written report.
    assert "Lever." in text


def test_scripts_parse():
    import ast
    root = Path(__file__).resolve().parents[1]
    for p in sorted((root / "scripts").glob("*.py")):
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    ast.parse((root / "api" / "server.py").read_text(encoding="utf-8"))


def test_feature_lists_never_duplicate_a_join_key():
    """`line` is both a proposition key and a model feature. Projecting the
    key list plus the feature list without de-duplicating makes pandas reject
    the merge -- it has bitten both the backtest and the trainer."""
    from mlbedge import config as C
    from mlbedge import dataset as D

    props = pd.DataFrame({
        "event_id": ["e1"], "market": ["totals"], "subject": ["game"],
        "line": [8.5], "n_books_prop": [6.0], "hold_med": [0.04],
        "lead_min": [30.0], "n_quotes": [12.0], "book_std": [0.01],
        "book_spread": [0.02], "logit_cons": [0.1], "won": [1.0],
    })
    feats = D.feature_columns(props, "totals")
    projection = D.PROP_KEY + [f for f in feats
                               if f in props.columns and f not in D.PROP_KEY]
    assert len(projection) == len(set(projection)), projection
