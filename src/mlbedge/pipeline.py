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
    print(f"[quotes {season}] parsing alternate ladders", flush=True)
    frames.append(N.parse_alt(events, client=client))
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
    gr = N.compact(gr)
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


def build_season_frames(season: int, devig_method: str = "shin",
                        tags: tuple[str, ...] = ("decision", "closing")
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Consensus, propositions and bet candidates for one season.

    Built per season and thrown away: the quote frame is by far the largest
    object in the pipeline (millions of rows per season), while the frames it
    produces are two orders of magnitude smaller. Holding three seasons of
    quotes in memory at once is what turns this from a laptop job into a
    cluster job, for no benefit -- nothing downstream of the consensus needs
    to see two seasons at the same time.
    """
    import hashlib
    import json as _json

    from . import calibrate as CAL
    from . import config as C
    from . import dataset as D
    from . import ladder as LAD
    from .devig import attach_consensus

    # Cache keyed on the inputs that change the answer: the de-vig model, and
    # the fitted tables. A new ladder or new anchor weights must invalidate,
    # or a re-run silently reports the previous configuration's numbers.
    tbl = LAD.load()
    weights = CAL.load()
    sig = hashlib.sha1(
        _json.dumps([devig_method, sorted(tags), tbl, weights],
                    sort_keys=True).encode()).hexdigest()[:10]
    pp = C.CURATED / f"props_{season}_{sig}.parquet"
    cc = C.CURATED / f"cand_{season}_{sig}.parquet"
    kk = C.CURATED / f"close_{season}_{sig}.parquet"
    if pp.exists() and cc.exists():
        return (pd.read_parquet(pp), pd.read_parquet(cc),
                pd.read_parquet(kk) if kk.exists() else pd.DataFrame())

    graded = build_graded(season, tags=tags)
    games, players = build_results(season)
    cons = attach_consensus(graded, method=devig_method, min_books=2,
                            self_anchor_markets=C.SELF_ANCHOR_MARKETS,
                            fitted_weights=weights)
    props = D.build_props(graded, games, players, cons=cons, ladder_table=tbl)
    cand = D.build_candidates(graded, cons=cons, tag="decision", props=props,
                              ladder_table=tbl)
    if not cand.empty:
        cand = cand.merge(games[["espn_id", "game_date"]], on="espn_id",
                          how="left", suffixes=("", "_g"))
        if "game_date_g" in cand:
            cand["game_date"] = cand["game_date"].astype(object).fillna(
                cand["game_date_g"])
            cand = cand.drop(columns=["game_date_g"])
    closing = cons[cons["tag"] == "closing"][
        D_CLOSING_COLS].copy() if "tag" in cons else pd.DataFrame()
    del graded, cons
    props.to_parquet(pp, index=False)
    cand.to_parquet(cc, index=False)
    if not closing.empty:
        closing.to_parquet(kk, index=False)
    return props, cand, closing


D_CLOSING_COLS = ["event_id", "market", "subject", "line", "tag", "side",
                  "book", "price"]
