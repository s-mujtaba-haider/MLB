"""Backfill driver for The Odds API.

Shape of the pull:

* **Event discovery** -- one historical `/events` call per day (1 credit). Each
  call returns every *upcoming* event at that instant, so a game appears in the
  listings of several prior days; we union and dedupe by event id. That
  redundancy is what makes the discovery robust to odd start times.

* **Props** -- the per-event endpoint, one call per (event, region). The
  decision snapshot is taken at exactly first pitch minus
  `DECISION_OFFSET_MIN`; the closing snapshot at minus `CLOSING_OFFSET_MIN`.
  Closing prices exist only to measure CLV and never feed a decision.

* **Featured** -- the whole-slate endpoint returns every event at once, so
  paying per event would be waste. Events are bucketed into 30-minute
  commence-time groups and one snapshot is taken per bucket, 30 minutes before
  the *earliest* first pitch in it. Every quote therefore lands between 30 and
  60 minutes before its own game starts; the realised lead time is recorded on
  each row so it can be filtered or modelled.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import pandas as pd

from . import config as C
from .oddsapi import LEDGER, OddsClient, run_pool

UTC = dt.timezone.utc


def _iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def season_dates(season: int) -> list[dt.date]:
    lo, hi = C.SEASONS[season]
    d0 = dt.date.fromisoformat(lo)
    d1 = dt.date.fromisoformat(hi)
    return [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]


# ---------------------------------------------------------------------------
# Event discovery
# ---------------------------------------------------------------------------

def discover_events(season: int, client: OddsClient | None = None,
                    workers: int = 8) -> pd.DataFrame:
    """Union of every event seen in that season's daily listings."""
    client = client or OddsClient()
    out = C.CURATED / f"events_{season}.parquet"
    dates = season_dates(season)

    def one(d: dt.date):
        ts = _iso(dt.datetime(d.year, d.month, d.day, 11, 0, tzinfo=UTC))
        r = client.historical_events(ts)
        return r.data or []

    print(f"[events {season}] {len(dates)} daily listings")
    chunks = run_pool([(d,) for d in dates], one, workers=workers,
                      desc=f"events {season}")

    seen: dict[str, dict] = {}
    for ch in chunks:
        for e in ch:
            if e.get("id") and e["id"] not in seen:
                seen[e["id"]] = {
                    "event_id": e["id"],
                    "sport_key": e.get("sport_key"),
                    "commence_time": e.get("commence_time"),
                    "home_team": e.get("home_team"),
                    "away_team": e.get("away_team"),
                }
    df = pd.DataFrame(sorted(seen.values(), key=lambda r: r["commence_time"] or ""))
    if df.empty:
        return df
    df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True, format="ISO8601")
    lo, hi = C.SEASONS[season]
    # Keep events whose local (US Eastern) game date falls inside the regular
    # season window. Eastern, not UTC: a 7:05pm ET game is 23:05Z the same day
    # but a 10:10pm ET game is 02:10Z the *next* UTC day.
    eastern = df["commence_dt"].dt.tz_convert("America/New_York")
    df["game_date"] = eastern.dt.date.astype(str)
    df = df[(df["game_date"] >= lo) & (df["game_date"] <= hi)].copy()
    df["season"] = season
    df.to_parquet(out, index=False)
    print(f"[events {season}] {len(df)} events -> {out.name}")
    return df


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------

def fetch_props(events: pd.DataFrame, tag: str = "decision",
                client: OddsClient | None = None, workers: int = 10,
                regions: tuple[str, ...] | None = None) -> int:
    """Fetch every (event, region) prop snapshot for `tag`.

    Returns the number of (event, region) pairs fetched or already cached.
    Idempotent: cached pairs cost nothing.
    """
    client = client or OddsClient()
    offset = C.DECISION_OFFSET_MIN if tag == "decision" else C.CLOSING_OFFSET_MIN
    plan = {r: mk for r, mk in C.PROP_REQUEST_PLAN.items()
            if regions is None or r in regions}

    jobs = []
    for row in events.itertuples():
        snap = _iso(_parse(row.commence_time) - dt.timedelta(minutes=offset))
        for region, markets in plan.items():
            jobs.append((row.event_id, snap, region, markets, tag))

    def one(event_id, snap, region, markets, tag):
        client.historical_event_odds(event_id, snap, region, markets, tag)
        return 1

    print(f"[props {tag}] {len(jobs)} (event,region) requests over "
          f"{len(events)} events")
    got = run_pool(jobs, one, workers=workers, desc=f"props {tag}")
    print(f"[props {tag}] done. {LEDGER.summary()}")
    return len(got)


def fetch_alt(events: pd.DataFrame, tag: str = "alt",
              client: OddsClient | None = None, workers: int = 10,
              regions: tuple[str, ...] | None = None) -> int:
    """Alternate run lines and totals, from the per-event endpoint.

    Same decision instant as everything else. Cached under its own tag so it
    is additive to an existing backfill rather than invalidating it.
    """
    client = client or OddsClient()
    plan = {r: mk for r, mk in C.ALT_REQUEST_PLAN.items()
            if regions is None or r in regions}
    jobs = []
    for row in events.itertuples():
        snap = _iso(_parse(row.commence_time)
                    - dt.timedelta(minutes=C.DECISION_OFFSET_MIN))
        for region, markets in plan.items():
            jobs.append((row.event_id, snap, region, markets, tag))

    def one(event_id, snap, region, markets, tag):
        client.historical_event_odds(event_id, snap, region, markets, tag)
        return 1

    print(f"[alt {tag}] {len(jobs)} (event,region) requests over "
          f"{len(events)} events")
    got = run_pool(jobs, one, workers=workers, desc=f"alt {tag}")
    print(f"[alt {tag}] done. {LEDGER.summary()}")
    return len(got)


# ---------------------------------------------------------------------------
# Featured
# ---------------------------------------------------------------------------

def featured_buckets(events: pd.DataFrame, bucket_min: int = 30
                     ) -> dict[str, list[str]]:
    """Group events into commence-time buckets -> snapshot timestamp.

    Returns {snapshot_iso: [event_id, ...]}.
    """
    buckets: dict[dt.datetime, list[tuple[dt.datetime, str]]] = defaultdict(list)
    for row in events.itertuples():
        ct = _parse(row.commence_time)
        floor = ct.replace(minute=(ct.minute // bucket_min) * bucket_min,
                           second=0, microsecond=0)
        buckets[floor].append((ct, row.event_id))

    out: dict[str, list[str]] = {}
    for floor, items in buckets.items():
        earliest = min(c for c, _ in items)
        snap = earliest - dt.timedelta(minutes=C.DECISION_OFFSET_MIN)
        out.setdefault(_iso(snap), []).extend(eid for _, eid in items)
    return out


def fetch_featured(events: pd.DataFrame, client: OddsClient | None = None,
                   workers: int = 8, regions: tuple[str, ...] | None = None,
                   tag: str = "decision") -> int:
    client = client or OddsClient()
    plan = {r: mk for r, mk in C.FEATURED_REQUEST_PLAN.items()
            if regions is None or r in regions}
    if tag == "closing":
        buckets = {}
        for row in events.itertuples():
            snap = _iso(_parse(row.commence_time)
                        - dt.timedelta(minutes=C.CLOSING_OFFSET_MIN))
            buckets.setdefault(snap, []).append(row.event_id)
    else:
        buckets = featured_buckets(events)

    jobs = [(snap, region, markets)
            for snap in buckets
            for region, markets in plan.items()]

    def one(snap, region, markets):
        client.historical_featured(snap, region, markets)
        return 1

    print(f"[featured {tag}] {len(buckets)} snapshots x {len(plan)} regions "
          f"= {len(jobs)} requests")
    got = run_pool(jobs, one, workers=workers, desc=f"featured {tag}")
    print(f"[featured {tag}] done. {LEDGER.summary()}")
    return len(got)
