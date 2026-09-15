"""ESPN public API client -- the results/grading source.

MLB's own statsapi.mlb.com does not resolve from this network, so ESPN supplies
the settlement data. It carries everything the eleven markets need:

  * batting line  : AB, R, H, RBI, HR, BB, K per player
  * pitching line : IP (as "full.part" outs), K per player
  * play-by-play  : distinguishes Single / Double / Triple / Home Run, which is
                    the only way to get total bases -- ESPN's batting line has
                    no 2B/3B column
  * game          : final score, venue, status, start time

Raw summaries are ~280 KB each; we cache a trimmed projection of the payload
(header, boxscore, gameInfo, and only the plays that carry a batting result),
which gzips to roughly 20 KB and keeps a full-season backfill on disk cheap.
"""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

from . import config as C

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"
_TIMEOUT = 45
_MAX_TRIES = 4

# Play types that resolve a plate appearance into a batting outcome we care
# about. ESPN's `type.text` is stable across seasons.
HIT_TYPES = {
    "Single": 1,
    "Double": 2,
    "Triple": 3,
    "Home Run": 4,
}


class EspnClient:
    def __init__(self) -> None:
        self._local = threading.local()
        self.root = C.RAW / "espn"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            # Leave the default requests User-Agent alone: ESPN's CDN 403s on
            # some custom UA strings.
            self._local.s = s
        return s

    # -- cache -----------------------------------------------------------
    def _path(self, key: str) -> Path:
        p = self.root / (key + ".json.gz")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _read(self, key: str) -> Any | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, EOFError, json.JSONDecodeError):
            p.unlink(missing_ok=True)
            return None

    def _write(self, key: str, obj: Any) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(obj, fh)
        tmp.replace(p)

    def _fetch(self, url: str, params: dict) -> Any:
        last: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                r = self.session.get(url, params=params, timeout=_TIMEOUT)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 404:
                    return None
                last = RuntimeError(f"{r.status_code} {r.text[:150]}")
            except requests.RequestException as e:
                last = e
            time.sleep(min(20, 2 ** attempt) + random.random())
        raise RuntimeError(f"espn fetch failed {url} {params}: {last}")

    # -- endpoints -------------------------------------------------------
    def scoreboard(self, date: str) -> list[dict]:
        """All MLB games on a date. `date` is YYYY-MM-DD."""
        key = f"scoreboard/{date}"
        cached = self._read(key)
        if cached is None:
            raw = self._fetch(SCOREBOARD,
                              {"dates": date.replace("-", ""), "limit": 200})
            cached = _trim_scoreboard(raw)
            # Only persist finished slates; an in-progress day must be refetched.
            if cached is not None and all(
                    g["status"] in ("STATUS_FINAL", "STATUS_POSTPONED",
                                    "STATUS_CANCELED")
                    for g in cached):
                self._write(key, cached)
        return cached or []

    def summary(self, event_id: str) -> dict | None:
        key = f"summary/{str(event_id)[:5]}/{event_id}"
        cached = self._read(key)
        if cached is not None:
            return cached or None
        raw = self._fetch(SUMMARY, {"event": event_id})
        if raw is None:
            return None
        trimmed = _trim_summary(raw)
        if trimmed and trimmed.get("status") == "STATUS_FINAL":
            self._write(key, trimmed)
        return trimmed


# ---------------------------------------------------------------------------
# Payload trimming
# ---------------------------------------------------------------------------

def _trim_scoreboard(raw: dict | None) -> list[dict] | None:
    if not raw:
        return None
    out = []
    for ev in raw.get("events", []):
        comps = ev.get("competitions") or [{}]
        comp = comps[0]
        teams = {}
        for c in comp.get("competitors", []):
            teams[c.get("homeAway")] = {
                "id": c.get("team", {}).get("id"),
                "name": c.get("team", {}).get("displayName"),
                "abbr": c.get("team", {}).get("abbreviation"),
                "score": c.get("score"),
            }
        out.append({
            "event_id": ev.get("id"),
            "date": ev.get("date"),
            "status": (comp.get("status", {}).get("type", {}).get("name")
                       or ev.get("status", {}).get("type", {}).get("name")),
            "venue": comp.get("venue", {}).get("fullName"),
            "home": teams.get("home"),
            "away": teams.get("away"),
            "doubleheader": comp.get("doubleheader"),
        })
    return out


def _trim_summary(raw: dict) -> dict | None:
    hdr = raw.get("header") or {}
    comps = hdr.get("competitions") or [{}]
    comp = comps[0]
    status = comp.get("status", {}).get("type", {}).get("name")

    teams = {}
    for c in comp.get("competitors", []):
        teams[c.get("homeAway")] = {
            "id": c.get("team", {}).get("id"),
            "name": c.get("team", {}).get("displayName"),
            "abbr": c.get("team", {}).get("abbreviation"),
            "score": c.get("score"),
        }

    # Batting / pitching lines, keyed by ESPN athlete id.
    players: list[dict] = []
    for block in (raw.get("boxscore") or {}).get("players", []):
        team = (block.get("team") or {}).get("abbreviation")
        team_id = (block.get("team") or {}).get("id")
        for grp in block.get("statistics", []):
            keys = grp.get("keys") or []
            if "atBats" in keys:
                group = "batting"
            elif "fullInnings.partInnings" in keys:
                group = "pitching"
            else:
                continue
            for a in grp.get("athletes", []):
                ath = a.get("athlete") or {}
                stats = a.get("stats") or []
                if not stats:
                    continue
                players.append({
                    "group": group,
                    "team": team,
                    "team_id": team_id,
                    "athlete_id": ath.get("id"),
                    "name": ath.get("displayName"),
                    "short": ath.get("shortName"),
                    "keys": keys,
                    "stats": stats,
                    "starter": a.get("starter"),
                })

    # Only plays that resolve a batting outcome, so total bases and the
    # per-batter hit breakdown can be reconstructed exactly.
    plays: list[dict] = []
    for p in raw.get("plays", []):
        ttext = (p.get("type") or {}).get("text")
        if ttext not in HIT_TYPES:
            continue
        batter = None
        for part in p.get("participants") or []:
            if part.get("type") == "batter":
                batter = (part.get("athlete") or {}).get("id")
        plays.append({
            "type": ttext,
            "batter": batter,
            "atBatId": p.get("atBatId"),
            "batOrder": p.get("batOrder"),
            "team": (p.get("team") or {}).get("id"),
        })

    return {
        "event_id": hdr.get("id"),
        "date": comp.get("date"),
        "status": status,
        "venue": (raw.get("gameInfo") or {}).get("venue", {}).get("fullName"),
        "attendance": (raw.get("gameInfo") or {}).get("attendance"),
        "home": teams.get("home"),
        "away": teams.get("away"),
        "players": players,
        "plays": plays,
    }


# ---------------------------------------------------------------------------
# Stat extraction
# ---------------------------------------------------------------------------

def _num(v: str | None) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def ip_to_outs(ip: str | None) -> int:
    """'5.2' innings pitched -> 17 outs. ESPN encodes partial innings in the
    decimal place as thirds, so this is not a float multiplication."""
    if not ip:
        return 0
    s = str(ip)
    if "." in s:
        full, part = s.split(".", 1)
    else:
        full, part = s, "0"
    try:
        return int(full) * 3 + int(part[0])
    except ValueError:
        return 0


def player_stats(summary: dict) -> list[dict]:
    """Flatten a trimmed summary into one row per player-game-role, with the
    graded statistics the markets settle against."""
    # Hit-type counts per batter, from play-by-play.
    tb: dict[str, int] = {}
    hit_counts: dict[str, dict[str, int]] = {}
    for p in summary.get("plays", []):
        b = p.get("batter")
        if not b:
            continue
        bases = HIT_TYPES.get(p["type"], 0)
        tb[b] = tb.get(b, 0) + bases
        hit_counts.setdefault(b, {}).setdefault(p["type"], 0)
        hit_counts[b][p["type"]] += 1

    rows = []
    for pl in summary.get("players", []):
        keys, stats = pl.get("keys") or [], pl.get("stats") or []
        kv = dict(zip(keys, stats))
        aid = pl.get("athlete_id")
        base = {
            "event_id": summary.get("event_id"),
            "athlete_id": aid,
            "name": pl.get("name"),
            "team": pl.get("team"),
            "team_id": pl.get("team_id"),
            "group": pl["group"],
            "starter": pl.get("starter"),
        }
        if pl["group"] == "batting":
            ab = _num(kv.get("atBats"))
            base.update({
                "at_bats": ab,
                "runs": _num(kv.get("runs")),
                "hits": _num(kv.get("hits")),
                "rbi": _num(kv.get("RBIs")),
                "home_runs": _num(kv.get("homeRuns")),
                "walks": _num(kv.get("walks")),
                "strike_outs": _num(kv.get("strikeouts")),
                "pitches_seen": _num(kv.get("pitches")),
                "total_bases": float(tb.get(aid, 0)),
                "doubles": float(hit_counts.get(aid, {}).get("Double", 0)),
                "triples": float(hit_counts.get(aid, {}).get("Triple", 0)),
                "played": 1.0 if (ab > 0 or _num(kv.get("walks")) > 0) else 0.0,
            })
        else:
            base.update({
                "outs": float(ip_to_outs(kv.get("fullInnings.partInnings"))),
                "strike_outs": _num(kv.get("strikeouts")),
                "hits_allowed": _num(kv.get("hits")),
                "earned_runs": _num(kv.get("earnedRuns")),
                "walks": _num(kv.get("walks")),
                "home_runs_allowed": _num(kv.get("homeRuns")),
                "pitch_count": _num(kv.get("pitches")),
                "played": 1.0,
            })
        rows.append(base)
    return rows
