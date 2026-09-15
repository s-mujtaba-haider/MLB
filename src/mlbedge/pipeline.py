"""End-to-end assembly: cached raw snapshots -> graded quotes -> gate input.

Each stage writes a parquet checkpoint so the expensive parts (parsing ~30,000
gzipped snapshot files, grading thirteen million quotes) happen once. Nothing
here touches the network: the Odds API client runs cache-only, so a rebuild can
never silently spend credits.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import config as C
from . import grade as G
from . import ingest_results as IR
from . import match as M
from . import normalize as N
from .oddsapi import OddsClient


def _season_window(season: int) -> tuple[str, str]:
    lo, hi = C.SEASONS[season]
    cutoff = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    return lo, min(hi, cutoff)


def build_quotes(season: int, tags: tuple[str, ...] = ("decision",),
                 force: bool = False) -> pd.DataFrame:
    """Parse cached snapshots for a season into one tidy quote table."""
    out = C.CURATED / f"quotes_{season}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    ev_path = C.CURATED / f"events_{season}.parquet"
    if not ev_path.exists():
        raise FileNotFoundError(f"run the backfill first: {ev_path} missing")
    lo, hi = _season_window(season)
    events = pd.read_parquet(ev_path)
    events = events[events["game_date"].between(lo, hi)].copy()

    client = OddsClient(cache_only=True)
    frames = []
    for tag in tags:
        print(f"[quotes {season}] parsing props ({tag}) for {len(events)} events",
              flush=True)
        frames.append(N.parse_props(events, tag=tag, client=client))
        print(f"[quotes {season}] parsing featured ({tag})", flush=True)
        frames.append(N.parse_featured(events, tag=tag, client=client))
    q = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    q["season"] = season
    q.to_parquet(out, index=False)
    print(f"[quotes {season}] {len(q):,} rows -> {out.name}", flush=True)
    return q


def build_results(season: int, force: bool = False
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    gp = C.CURATED / f"s{season}_games.parquet"
    pp = C.CURATED / f"s{season}_players.parquet"
    if gp.exists() and pp.exists() and not force:
        return pd.read_parquet(gp), pd.read_parquet(pp)
    lo, hi = _season_window(season)
    return IR.run(lo, hi, workers=12, save_prefix=f"s{season}")


def build_graded(season: int, tags: tuple[str, ...] = ("decision",),
                 force: bool = False) -> pd.DataFrame:
    """Quotes joined to results and settled."""
    out = C.CURATED / f"graded_{season}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    q = build_quotes(season, tags=tags)
    games, players = build_results(season)
    events = pd.read_parquet(C.CURATED / f"events_{season}.parquet")

    matched = M.match_games(events, games)
    e2e = dict(zip(matched["event_id"], matched["espn_id"]))
    n_ok = sum(1 for v in e2e.values() if v is not None and v == v)
    print(f"[graded {season}] {n_ok}/{len(e2e)} events matched to ESPN games",
          flush=True)

    q = M.attach_players(q, players, e2e)
    gr = G.grade(q, games, players)
    gr = gr.merge(games[["espn_id", "game_date"]], on="espn_id", how="left")
    gr.to_parquet(out, index=False)
    print(f"[graded {season}] {len(gr):,} rows -> {out.name}", flush=True)
    return gr


def load_all(seasons: tuple[int, ...], tags: tuple[str, ...] = ("decision",)
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Graded quotes, games and players across seasons."""
    gr, gm, pl = [], [], []
    for s in seasons:
        gr.append(build_graded(s, tags=tags))
        g, p = build_results(s)
        gm.append(g)
        pl.append(p)
    return (pd.concat(gr, ignore_index=True),
            pd.concat(gm, ignore_index=True).drop_duplicates("espn_id"),
            pd.concat(pl, ignore_index=True))
