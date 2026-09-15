"""Daily scheduler.

Reliability requirements, in the order they bite in practice:

1. **It must not die.** Every cycle is wrapped: a bad API response, a missing
   boxscore or a malformed line fails that cycle and the loop continues. A
   scheduler that exits on the first exception is worse than no scheduler,
   because it looks fine until the day it silently stopped.
2. **It must be idempotent.** Cycles overlap games, and a restart re-runs the
   day. Picks are keyed by (date, market, subject, line, side, book) and
   re-running replaces rather than duplicates.
3. **It must be observable.** A heartbeat file records the last successful
   cycle, what it produced, and the last error, so "is it still firing?" is
   answerable without reading logs.
4. **It must fire at the right time.** Prop markets are thin until lineups
   post. The loop sweeps every `interval_min` and picks up games in a lead-time
   window, so each game is priced once its market is liquid and well before
   first pitch.

Run continuously, or once per invocation under cron / Task Scheduler:

    python scripts/daily.py --loop            # long-running
    python scripts/daily.py --once            # single sweep
"""
from __future__ import annotations

import datetime as dt
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from . import config as C
from . import picks as PK
from . import pipeline as P
from . import production as PR
from .oddsapi import OddsClient

HEARTBEAT = C.DATA / "scheduler_heartbeat.json"
LOGFILE = C.REPORTS / "scheduler.log"

# Games are priced once they are inside this window before first pitch. The
# upper bound keeps us from quoting a market before lineups exist; the lower
# bound leaves time to actually place the bet.
LEAD_MAX_MIN = 240
LEAD_MIN_MIN = 20


def _log(msg: str) -> None:
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOGFILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _beat(**kw) -> None:
    prev = {}
    if HEARTBEAT.exists():
        try:
            prev = json.loads(HEARTBEAT.read_text())
        except json.JSONDecodeError:
            prev = {}
    prev.update(kw)
    prev["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds")
    HEARTBEAT.write_text(json.dumps(prev, indent=2))


class Scheduler:
    def __init__(self, seasons: tuple[int, ...] = (2024, 2025, 2026),
                 interval_min: int = 15):
        self.seasons = seasons
        self.interval = interval_min * 60
        self.bundles: dict[str, PR.Bundle] = {}
        self.players: pd.DataFrame | None = None
        self.games: pd.DataFrame | None = None
        self._loaded_on: str | None = None

    # -- state ------------------------------------------------------------
    def ensure_loaded(self) -> None:
        """Load models and history once per day; results grow daily."""
        today = dt.date.today().isoformat()
        if self._loaded_on == today and self.bundles:
            return
        self.bundles = PR.load()
        frames_g, frames_p = [], []
        for s in self.seasons:
            try:
                g, p = P.build_results(s)
                frames_g.append(g)
                frames_p.append(p)
            except Exception as e:  # noqa: BLE001
                _log(f"WARN could not load results for {s}: {e!r}")
        if not frames_g:
            raise RuntimeError("no historical results available")
        self.games = pd.concat(frames_g, ignore_index=True).drop_duplicates("espn_id")
        self.players = pd.concat(frames_p, ignore_index=True)
        self._loaded_on = today
        firing = [m for m, b in self.bundles.items() if b.fires]
        _log(f"loaded {len(self.bundles)} models ({len(firing)} firing: "
             f"{', '.join(sorted(firing))}); "
             f"{len(self.games)} games / {len(self.players)} player-games")

    def refresh_results(self) -> None:
        """Pull yesterday's and today's finished games so form stays current."""
        try:
            from . import ingest_results as IR
            today = dt.date.today()
            lo = (today - dt.timedelta(days=3)).isoformat()
            g, p = IR.run(lo, today.isoformat(), workers=8)
            if not g.empty and self.games is not None:
                self.games = (pd.concat([self.games, g], ignore_index=True)
                                .drop_duplicates("espn_id", keep="last"))
                self.players = (pd.concat([self.players, p], ignore_index=True)
                                  .drop_duplicates(
                                      subset=["event_id", "athlete_id", "group"],
                                      keep="last"))
                _log(f"refreshed results: +{len(g)} games")
        except Exception as e:  # noqa: BLE001
            _log(f"WARN result refresh failed: {e!r}")

    # -- one sweep --------------------------------------------------------
    def cycle(self) -> int:
        self.ensure_loaded()
        client = OddsClient()
        picks = PK.generate(self.bundles, self.players, self.games,
                            client=client,
                            within_hours=LEAD_MAX_MIN / 60.0,
                            min_lead_min=LEAD_MIN_MIN)
        if picks.empty:
            _beat(last_cycle_picks=0, last_error=None)
            return 0
        PK.save_picks(picks)
        by_mkt = picks["market"].value_counts().to_dict()
        _log(f"generated {len(picks)} picks: {by_mkt}")
        _beat(last_cycle_picks=int(len(picks)), by_market=by_mkt,
              last_error=None,
              credits_remaining=client.ledger.remaining)
        return len(picks)

    def run_once(self) -> int:
        try:
            n = self.cycle()
            _beat(last_success=dt.datetime.now(dt.timezone.utc)
                  .isoformat(timespec="seconds"))
            return n
        except Exception as e:  # noqa: BLE001 -- the loop must survive anything
            _log(f"ERROR cycle failed: {e!r}\n{traceback.format_exc()}")
            _beat(last_error=repr(e)[:400])
            return -1

    def loop(self) -> None:
        _log(f"scheduler starting, interval {self.interval // 60} min")
        last_refresh = None
        while True:
            today = dt.date.today().isoformat()
            if last_refresh != today:
                self.refresh_results()
                last_refresh = today
            self.run_once()
            time.sleep(self.interval)
