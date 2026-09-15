"""Raw Odds API snapshots -> one tidy quote table.

Every row is a single takeable price: one book, one side, one line, at one
instant. All eleven markets share a schema so that downstream code -- de-vig,
consensus, EV, grading, the gate -- is written once rather than eleven times.

Schema
------
event_id      Odds API event id
market        our canonical market name (config.Market.name)
subject       player name for props; 'game' for featured markets
side          'over'/'under' for O/U markets, 'home'/'away' for h2h & spreads
line          the handicap from the over/home perspective (NaN for h2h)
price         American odds
book          bookmaker key
region        Odds API region the quote came from
snapshot_ts   the instant the snapshot was taken (the decision time)
commence_time first pitch
lead_min      minutes between snapshot and first pitch -- always > 0
tag           'decision' or 'closing'

A proposition (the thing that gets de-vigged and graded) is identified by
(event_id, market, subject, line); the two sides of it are the `side` values.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from .oddsapi import OddsClient

UTC = dt.timezone.utc

# Outcome names that mean "over"/"under" across the props and totals markets.
_OVER = {"over", "yes"}
_UNDER = {"under", "no"}


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class _ChunkedRows:
    """Accumulate quote dicts, converting to a compact frame every so often.

    A season is roughly four million quotes. Held as Python dicts until the
    end, that is several gigabytes of interpreter objects before pandas even
    starts; converted and compacted in chunks, it stays in the low hundreds of
    megabytes.
    """

    def __init__(self, chunk: int = 400_000):
        self.chunk = chunk
        self._rows: list[dict] = []
        self._frames: list[pd.DataFrame] = []

    def extend(self, rows) -> None:
        self._rows.extend(rows)
        if len(self._rows) >= self.chunk:
            self.flush()

    def append(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.chunk:
            self.flush()

    def flush(self) -> None:
        if self._rows:
            self._frames.append(compact(_finish(pd.DataFrame(self._rows))))
            self._rows = []

    def frame(self) -> pd.DataFrame:
        self.flush()
        if not self._frames:
            return pd.DataFrame()
        if len(self._frames) == 1:
            return self._frames[0]
        return compact(pd.concat(self._frames, ignore_index=True))


def _lead_min(snapshot_ts: str | None, commence: str) -> float:
    if not snapshot_ts:
        return np.nan
    return (_parse(commence) - _parse(snapshot_ts)).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------

def parse_props(events: pd.DataFrame, tag: str = "decision",
                client: OddsClient | None = None) -> pd.DataFrame:
    client = client or OddsClient(cache_only=True)
    offset = C.DECISION_OFFSET_MIN if tag == "decision" else C.CLOSING_OFFSET_MIN
    rows = _ChunkedRows()

    for ev in events.itertuples():
        want = _iso(_parse(ev.commence_time) - dt.timedelta(minutes=offset))
        for region, markets in C.PROP_REQUEST_PLAN.items():
            r = client.historical_event_odds(ev.event_id, want, region,
                                             markets, tag)
            data = r.data
            if not data:
                continue
            snap = r.snapshot_ts
            lead = _lead_min(snap, ev.commence_time)
            for bk in data.get("bookmakers", []):
                book = bk.get("key")
                for mk in bk.get("markets", []):
                    m = C.BY_API_KEY.get(mk.get("key"))
                    if m is None:
                        continue
                    for o in mk.get("outcomes", []):
                        nm = (o.get("name") or "").strip().lower()
                        if nm in _OVER:
                            side = "over"
                        elif nm in _UNDER:
                            side = "under"
                        else:
                            continue
                        subject = (o.get("description") or "").strip()
                        if not subject:
                            continue
                        rows.append({
                            "event_id": ev.event_id,
                            "market": m.name,
                            "subject": subject,
                            "side": side,
                            "line": o.get("point"),
                            "price": o.get("price"),
                            "book": book,
                            "region": region,
                            "snapshot_ts": snap,
                            "commence_time": ev.commence_time,
                            "lead_min": lead,
                            "tag": tag,
                        })
    return rows.frame()


def parse_alt(events: pd.DataFrame, tag: str = "alt",
              client: OddsClient | None = None) -> pd.DataFrame:
    """Alternate spread/total rungs from the per-event endpoint.

    They normalise into the same `spreads`/`totals` markets at a different
    `line`, so they grade, de-vig and price through exactly the same path as
    the main number.
    """
    client = client or OddsClient(cache_only=True)
    rows = _ChunkedRows()
    for ev in events.itertuples():
        want = _iso(_parse(ev.commence_time)
                    - dt.timedelta(minutes=C.DECISION_OFFSET_MIN))
        for region, markets in C.ALT_REQUEST_PLAN.items():
            r = client.historical_event_odds(ev.event_id, want, region,
                                             markets, tag)
            if not r.data:
                continue
            snap = r.snapshot_ts
            lead = _lead_min(snap, ev.commence_time)
            if np.isfinite(lead) and lead <= 0:
                continue
            for bk in r.data.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    m = C.BY_API_KEY.get(mk.get("key"))
                    if m is None:
                        continue
                    rows.extend(_featured_outcomes(
                        mk, m, ev.event_id, bk.get("key"), region, snap,
                        ev.commence_time, lead, ev.home_team, ev.away_team,
                        "decision"))
    return rows.frame()


# ---------------------------------------------------------------------------
# Featured
# ---------------------------------------------------------------------------

def parse_featured(events: pd.DataFrame, tag: str = "decision",
                   client: OddsClient | None = None) -> pd.DataFrame:
    """Parse whole-slate snapshots, keeping only the events we asked about.

    A featured snapshot carries every upcoming game, so the same file is read
    once per bucket and filtered down to that bucket's events.
    """
    from .ingest_odds import featured_buckets

    client = client or OddsClient(cache_only=True)
    meta = {e.event_id: (e.commence_time, e.home_team, e.away_team)
            for e in events.itertuples()}

    if tag == "closing":
        buckets: dict[str, list[str]] = {}
        for e in events.itertuples():
            snap = _iso(_parse(e.commence_time)
                        - dt.timedelta(minutes=C.CLOSING_OFFSET_MIN))
            buckets.setdefault(snap, []).append(e.event_id)
    else:
        buckets = featured_buckets(events)

    rows = _ChunkedRows()
    for want_ts, event_ids in buckets.items():
        wanted = set(event_ids)
        for region, markets in C.FEATURED_REQUEST_PLAN.items():
            r = client.historical_featured(want_ts, region, markets)
            if not r.data:
                continue
            snap = r.snapshot_ts
            for ev in r.data:
                eid = ev.get("id")
                if eid not in wanted or eid not in meta:
                    continue
                commence, home, away = meta[eid]
                lead = _lead_min(snap, commence)
                if not (lead is None or np.isnan(lead)) and lead <= 0:
                    continue  # never keep a quote from after first pitch
                for bk in ev.get("bookmakers", []):
                    book = bk.get("key")
                    for mk in bk.get("markets", []):
                        m = C.BY_API_KEY.get(mk.get("key"))
                        if m is None:
                            continue
                        rows.extend(_featured_outcomes(
                            mk, m, eid, book, region, snap, commence, lead,
                            home, away, tag))
    return rows.frame()


def _featured_outcomes(mk, m, eid, book, region, snap, commence, lead,
                       home, away, tag) -> list[dict]:
    out = []
    for o in mk.get("outcomes", []):
        nm = (o.get("name") or "").strip()
        low = nm.lower()
        if m.name == "totals":
            if low in _OVER:
                side, line = "over", o.get("point")
            elif low in _UNDER:
                side, line = "under", o.get("point")
            else:
                continue
        elif m.name in ("h2h", "spreads"):
            if nm == home:
                side = "home"
            elif nm == away:
                side = "away"
            else:
                continue
            # Store the handicap from the home perspective on both rows, so a
            # proposition key is shared by its two sides.
            pt = o.get("point")
            if m.name == "h2h":
                line = np.nan
            else:
                line = pt if side == "home" else (-pt if pt is not None else None)
        else:
            continue
        out.append({
            "event_id": eid, "market": m.name, "subject": "game", "side": side,
            "line": line, "price": o.get("price"), "book": book,
            "region": region, "snapshot_ts": snap, "commence_time": commence,
            "lead_min": lead, "tag": tag,
        })
    return out


# ---------------------------------------------------------------------------

#: Columns worth storing as categoricals. Three seasons is ~17M quote rows;
#: left as Python objects that is tens of gigabytes, and as categories it is
#: well under one.
_CATEGORICAL = ("event_id", "market", "subject", "side", "book", "region",
                "tag", "snapshot_ts", "commence_time", "espn_id",
                "athlete_id", "match_how", "result", "game_date")


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink a quote frame in place-ish, without changing any value."""
    if df.empty:
        return df
    for c in _CATEGORICAL:
        if c in df.columns and not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype("category")
    for c in ("line", "lead_min", "stat_value", "p_cons", "p_book_fair",
              "hold_book", "n_books_cons"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("int16")
    return df


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df[df["price"].notna()]
    # A price inside (-100, 100) is not a valid American quote.
    df = df[(df["price"] <= -100) | (df["price"] >= 100)]
    df["tier"] = df["book"].map(C.book_tier).astype("int8")
    # The same book can appear in two regions (betonlineag is in us and eu);
    # keep one row per takeable price.
    df = df.drop_duplicates(
        subset=["event_id", "market", "subject", "side", "line", "book", "tag"],
        keep="first")
    return df.reset_index(drop=True)
