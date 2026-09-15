"""Pull ESPN scoreboards and boxscores, and flatten them into two tables.

games        one row per completed game: teams, score, venue, start time
player_games one row per player-game-role, carrying every statistic the eleven
             markets settle against

Postponed, cancelled and suspended games are dropped rather than graded. A
postponed game whose props were already quoted is a *void*, not a loss, and
quietly grading it as a loss is one of the classic ways a backfill turns into a
fantasy.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as C
from .espn import EspnClient, player_stats
from .oddsapi import run_pool

FINAL = "STATUS_FINAL"


def date_range(d0: str, d1: str) -> list[str]:
    a = dt.date.fromisoformat(d0)
    b = dt.date.fromisoformat(d1)
    return [(a + dt.timedelta(days=i)).isoformat()
            for i in range((b - a).days + 1)]


def fetch_scoreboards(dates: list[str], client: EspnClient | None = None,
                      workers: int = 8) -> pd.DataFrame:
    client = client or EspnClient()

    def one(d):
        return [dict(g, game_date=d) for g in client.scoreboard(d)]

    print(f"[espn scoreboards] {len(dates)} dates")
    chunks = run_pool([(d,) for d in dates], one, workers=workers,
                      desc="scoreboards", progress_every=40)
    rows = [g for ch in chunks for g in ch]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)


def fetch_boxscores(scoreboard: pd.DataFrame, client: EspnClient | None = None,
                    workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = client or EspnClient()
    finals = scoreboard[scoreboard["status"] == FINAL]
    ids = finals["event_id"].tolist()

    def one(eid):
        return client.summary(eid)

    print(f"[espn boxscores] {len(ids)} final games "
          f"({len(scoreboard) - len(finals)} non-final skipped)")
    summaries = run_pool([(i,) for i in ids], one, workers=workers,
                         desc="boxscores", progress_every=200)

    game_rows, player_rows = [], []
    for s in summaries:
        if not s or s.get("status") != FINAL:
            continue
        home, away = s.get("home") or {}, s.get("away") or {}
        try:
            hs, as_ = int(home.get("score")), int(away.get("score"))
        except (TypeError, ValueError):
            continue
        game_rows.append({
            "espn_id": s["event_id"],
            "date": s.get("date"),
            "venue": s.get("venue"),
            "home_team": home.get("name"),
            "away_team": away.get("name"),
            "home_abbr": home.get("abbr"),
            "away_abbr": away.get("abbr"),
            "home_score": hs,
            "away_score": as_,
            "total_runs": hs + as_,
            "home_margin": hs - as_,
        })
        for r in player_stats(s):
            player_rows.append(r)

    games = pd.DataFrame(game_rows)
    players = pd.DataFrame(player_rows)
    if not games.empty:
        games["commence_dt"] = pd.to_datetime(games["date"], utc=True,
                                              format="ISO8601")
        games["game_date"] = (games["commence_dt"]
                              .dt.tz_convert("America/New_York").dt.date
                              .astype(str))
    return games, players


def run(d0: str, d1: str, workers: int = 8, save_prefix: str | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = date_range(d0, d1)
    sb = fetch_scoreboards(dates, workers=workers)
    if sb.empty:
        raise RuntimeError(f"no ESPN games found for {d0}..{d1}")
    games, players = fetch_boxscores(sb, workers=workers)
    print(f"[espn] {len(games)} games, {len(players)} player-game rows")
    if save_prefix:
        games.to_parquet(C.CURATED / f"{save_prefix}_games.parquet", index=False)
        players.to_parquet(C.CURATED / f"{save_prefix}_players.parquet",
                           index=False)
    return games, players
