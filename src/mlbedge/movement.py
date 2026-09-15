"""Line movement for the featured markets, reconstructed from cached data.

Which way a number moved before first pitch is one of the few genuinely
predictive things available at decision time: it is where the informed money
went. We were throwing it away for no reason.

The whole-slate endpoint returns *every* upcoming game in one response, so a
snapshot taken to price the 5pm games also contains the 10pm ones. The
normaliser discards those extra events because they belong to a different
bucket -- but they are already on disk, paid for. Reading them back gives each
game a price history across the afternoon at zero additional API cost.

Only the featured markets get this. Props come from a per-event endpoint,
where a second timestamp is a second charge.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from .devig import book_views, consensus, logit
from .normalize import _finish, _lead_min, _parse
from .oddsapi import OddsClient

# Lead-time checkpoints, in minutes before first pitch. The consensus is
# sampled at the latest snapshot at or before each, so a move is measured
# between comparable moments rather than between whatever happened to exist.
CHECKPOINTS = (360, 180, 90)


def featured_history(events: pd.DataFrame, client: OddsClient | None = None
                     ) -> pd.DataFrame:
    """Every featured quote for these events, from every cached snapshot."""
    from .ingest_odds import featured_buckets

    client = client or OddsClient(cache_only=True)
    meta = {e.event_id: (e.commence_time, e.home_team, e.away_team)
            for e in events.itertuples()}
    wanted = set(meta)

    from .normalize import _featured_outcomes

    rows: list[dict] = []
    for want_ts in featured_buckets(events):
        for region, markets in C.FEATURED_REQUEST_PLAN.items():
            r = client.historical_featured(want_ts, region, markets)
            if not r.data:
                continue
            snap = r.snapshot_ts
            for ev in r.data:
                eid = ev.get("id")
                if eid not in wanted:
                    continue
                commence, home, away = meta[eid]
                lead = _lead_min(snap, commence)
                # Keep the whole afternoon, not just this bucket's games --
                # that is the entire point.
                if not np.isfinite(lead) or lead <= 0 or lead > 12 * 60:
                    continue
                for bk in ev.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        m = C.BY_API_KEY.get(mk.get("key"))
                        if m is None or m.kind != "featured":
                            continue

                        rows.extend(_featured_outcomes(
                            mk, m, eid, bk.get("key"), region, snap, commence,
                            lead, home, away, "hist"))
    if not rows:
        return pd.DataFrame()
    return _finish(pd.DataFrame(rows))


def consensus_history(hist: pd.DataFrame, method: str = "shin"
                      ) -> pd.DataFrame:
    """Consensus per (event, market, line, snapshot), with its lead time."""
    if hist.empty:
        return hist
    h = hist.copy()
    # Each snapshot is its own market state; tag it so the consensus is built
    # within a snapshot rather than across the afternoon.
    h["tag"] = h["snapshot_ts"].astype(str)
    bv = consensus(book_views(h, method=method), min_books=2)
    if bv.empty:
        return pd.DataFrame()
    out = (bv.drop_duplicates(["event_id", "market", "subject", "line", "tag"])
             [["event_id", "market", "subject", "line", "tag", "p_cons_all"]]
             .rename(columns={"tag": "snapshot_ts"}))
    lead = (h.drop_duplicates(["event_id", "snapshot_ts"])
             [["event_id", "snapshot_ts", "lead_min"]])
    lead["snapshot_ts"] = lead["snapshot_ts"].astype(str)
    return out.merge(lead, on=["event_id", "snapshot_ts"], how="left")


def movement_features(hist_cons: pd.DataFrame) -> pd.DataFrame:
    """Per (event, market, line): how far the number moved, and how fast.

    Positive `move_*` means the consensus drifted toward the over/home side
    between that checkpoint and the last observation before first pitch.
    """
    if hist_cons.empty:
        return pd.DataFrame()
    d = hist_cons.dropna(subset=["p_cons_all", "lead_min"]).copy()
    d["l"] = logit(d["p_cons_all"].to_numpy(dtype=float))
    key = ["event_id", "market", "subject", "line"]

    # The decision-time value: latest snapshot still before first pitch.
    d = d.sort_values(key + ["lead_min"])
    last = d.groupby(key, dropna=False, observed=True).first().reset_index()
    last = last.rename(columns={"l": "l_decision",
                                "lead_min": "lead_decision"})

    out = last[key + ["l_decision", "lead_decision"]].copy()
    for cp in CHECKPOINTS:
        earlier = d[d["lead_min"] >= cp]
        if earlier.empty:
            out[f"move_{cp}m"] = np.nan
            continue
        # Latest snapshot at or before the checkpoint.
        ref = (earlier.sort_values(key + ["lead_min"])
                      .groupby(key, dropna=False, observed=True)
                      .first().reset_index()
                      .rename(columns={"l": f"l_{cp}"}))
        out = out.merge(ref[key + [f"l_{cp}"]], on=key, how="left")
        out[f"move_{cp}m"] = out["l_decision"] - out[f"l_{cp}"]
        out = out.drop(columns=[f"l_{cp}"])

    out["n_snapshots"] = (d.groupby(key, dropna=False, observed=True)
                           .size().reset_index(drop=True))
    counts = (d.groupby(key, dropna=False, observed=True).size()
                .rename("n_snapshots").reset_index())
    out = out.drop(columns=["n_snapshots"]).merge(counts, on=key, how="left")
    return out


def build(events: pd.DataFrame, client: OddsClient | None = None
          ) -> pd.DataFrame:
    hist = featured_history(events, client=client)
    if hist.empty:
        return pd.DataFrame()
    return movement_features(consensus_history(hist))


MOVE_COLS = tuple(f"move_{cp}m" for cp in CHECKPOINTS) + ("n_snapshots",)
