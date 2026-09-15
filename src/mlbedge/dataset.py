"""Assemble the modelling and betting frames.

Two frames come out of here, at deliberately different grains:

**props** -- one row per proposition (event, market, subject, line). This is
the modelling grain: ~1.5M rows carrying the full point-in-time feature set,
the all-book consensus, and the settled outcome of the over/home side. A
proposition quoted by twelve books must contribute one training row, not
twelve, or the fit silently reweights itself toward whatever the books like
quoting.

**candidates** -- one row per (proposition, side): the best price available at
a bettable book, with the leave-one-out consensus for *that* book. This is the
betting grain, and it mirrors what a bettor actually does -- shop the number,
take the best price, bet once. Evaluating all thirteen million quotes would
also be pretending you can bet the same proposition twelve times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .devig import attach_consensus, logit
from .grade import LOSS, PUSH, VOID, WIN

PROP_KEY = ["event_id", "market", "subject", "line"]

# Model features. These are all *proposition-level* quantities, deliberately
# named so they cannot collide with the per-book columns on the candidate
# frame -- a feature that means "median hold across books" at training time
# and "this book's hold" at serving time is a silent train/serve skew.
MARKET_STRUCT = ["n_books_prop", "hold_med", "lead_min", "line",
                 "book_spread", "n_quotes", "book_std",
                 "proj_mean", "line_minus_proj", "line_over_proj",
                 # ladder context: where this line sits relative to the one the
                 # market actually prices, and how far the two disagree
                 "primary_line", "p_primary", "n_books_primary",
                 "ladder_minus_cons", "is_primary_line"]


def proposition_outcome(graded: pd.DataFrame) -> pd.DataFrame:
    """Settle each proposition once, from the over/home perspective."""
    over_mask = graded["side"].isin(["over", "home"])
    g = graded[over_mask]
    if g.empty:
        return pd.DataFrame(columns=PROP_KEY + ["won", "stat_value"])
    agg = (g.groupby(PROP_KEY, dropna=False, observed=True)
             .agg(result=("result", "first"),
                  stat_value=("stat_value", "first")).reset_index())
    agg["won"] = np.where(agg["result"] == WIN, 1.0,
                          np.where(agg["result"] == LOSS, 0.0, np.nan))
    return agg[agg["result"].isin([WIN, LOSS])][
        PROP_KEY + ["won", "stat_value"]]


def proposition_structure(dec: pd.DataFrame) -> pd.DataFrame:
    """Per-proposition market-structure summary.

    Shared by the training frame and the live pick path. These are model
    features, so they have to be computed the same way in both places -- the
    live path previously left several of them null, which silently fed the
    model a different world at serving time than it was fitted on.
    """
    disp = (dec.dropna(subset=["p_book_fair"])
               .groupby(PROP_KEY, dropna=False, observed=True)["p_book_fair"]
               .agg(["std", "min", "max", "size"]))
    disp.columns = ["book_std", "book_min", "book_max", "n_two_sided"]
    disp["book_spread"] = disp["book_max"] - disp["book_min"]

    agg = {"p_cons_all": ("p_cons", "median"),
           "n_books_prop": ("n_books_prop", "max"),
           "hold_med": ("hold_book", "median"),
           "lead_min": ("lead_min", "first"),
           "n_quotes": ("price", "size"),
           "commence_time": ("commence_time", "first")}
    for opt in ("espn_id", "athlete_id"):
        if opt in dec.columns:
            agg[opt] = (opt, "first")
    return (dec.groupby(PROP_KEY, dropna=False, observed=True)
               .agg(**agg).reset_index().join(disp, on=PROP_KEY))


def build_props(graded: pd.DataFrame, games: pd.DataFrame,
                players: pd.DataFrame, cons: pd.DataFrame | None = None,
                ladder_table: dict | None = None) -> pd.DataFrame:
    """Proposition-grain frame with consensus, outcome and features."""
    from . import ladder as LAD

    q = cons if cons is not None else attach_consensus(graded)
    dec = q[q["tag"] == "decision"]
    base = proposition_structure(dec)

    out = base.merge(proposition_outcome(dec), on=PROP_KEY, how="inner")

    # Price the rest of the ladder off the line the market prices best. Without
    # this, a proposition quoted at an alternate line by fewer than two books
    # has no consensus at all and is discarded -- which throws away precisely
    # the lines books are laziest about.
    tbl = ladder_table if ladder_table is not None else LAD.load()
    out = LAD.attach(out, tbl)
    out["anchor_src"] = np.where(out["p_cons_all"].notna(), "direct", "ladder")
    out["p_anchor"] = out["p_cons_all"].fillna(out["p_ladder"])

    out = out.dropna(subset=["p_anchor"])
    out["logit_cons"] = logit(out["p_anchor"].to_numpy())
    out = attach_features(out, games, players)
    return out


def attach_features(props: pd.DataFrame, games: pd.DataFrame,
                    players: pd.DataFrame) -> pd.DataFrame:
    """Join every point-in-time form table onto the proposition frame."""
    g = games[["espn_id", "game_date", "venue", "home_team", "away_team"]].copy()
    g["game_date"] = g["game_date"].astype(str)
    out = props.merge(g, on="espn_id", how="left")

    bat = F.batter_form(players, games)
    pit = F.pitcher_form(players, games)
    tmf = F.team_form(players, games)
    prk = F.park_form(games)
    gcx = F.game_context(games)

    bat_cols = [c for c in bat.columns if c.startswith("bat_")]
    pit_cols = [c for c in pit.columns if c.startswith("pit_")]
    tm_cols = [c for c in tmf.columns if c.startswith("tm_")]
    prk_cols = [c for c in prk.columns if c.startswith("park_")]
    tg_cols = [c for c in gcx.columns if c.startswith("tg_")]

    side = out["market"].map({m.name: m.side for m in C.MARKETS})

    # Batter form for batting markets; pitcher form for pitching markets.
    out = out.merge(bat[["event_id", "athlete_id", "team"] + bat_cols]
                    .rename(columns={"event_id": "espn_id",
                                     "team": "player_team"}),
                    on=["espn_id", "athlete_id"], how="left")
    out = out.merge(pit[["event_id", "athlete_id"] + pit_cols]
                    .rename(columns={"event_id": "espn_id"}),
                    on=["espn_id", "athlete_id"], how="left")
    out.loc[side != "batting", bat_cols] = np.nan
    out.loc[side != "pitching", pit_cols] = np.nan

    # Player's own team offensive form, and the park.
    out = out.merge(tmf[["event_id", "team"] + tm_cols]
                    .rename(columns={"event_id": "espn_id",
                                     "team": "player_team"}),
                    on=["espn_id", "player_team"], how="left")
    out = out.merge(prk[["espn_id"] + prk_cols], on="espn_id", how="left")

    # Home and away team season form, for the game-level markets.
    home = (gcx[gcx["is_home"] == 1][["espn_id"] + tg_cols]
            .rename(columns={c: f"home_{c}" for c in tg_cols}))
    away = (gcx[gcx["is_home"] == 0][["espn_id"] + tg_cols]
            .rename(columns={c: f"away_{c}" for c in tg_cols}))
    out = out.merge(home, on="espn_id", how="left")
    out = out.merge(away, on="espn_id", how="left")

    out["is_home_player"] = (out["player_team"].notna() &
                             (out["player_team"] == out["home_team"])
                             ).astype(float)
    return add_projection(out)


# Which rolling rate column carries each market's statistic.
_BAT_RATE = {
    "batter_hits": "bat_hits_ppa",
    "batter_rbis": "bat_rbi_ppa",
    "batter_total_bases": "bat_total_bases_ppa",
    "batter_home_runs": "bat_home_runs_ppa",
    "runs_scored": "bat_runs_ppa",
    "batter_strikeouts": "bat_strike_outs_ppa",
}
_PIT_RATE = {
    "pitcher_strikeouts": "pit_strike_outs",
    "pitcher_outs": "pit_outs",
}


def add_projection(out: pd.DataFrame) -> pd.DataFrame:
    """An explicit baseball projection, and the line relative to it.

    A gradient booster can in principle recover "expected plate appearances
    times per-PA rate" from the raw rolling columns, but only by spending
    splits on it, and only for the markets where it happens to find the
    interaction. Handing it the product directly -- and, more importantly, the
    *line minus the projection* -- gives it the one quantity that actually
    decides a prop, in the units the prop is written in.
    """
    n = len(out)
    proj = np.full(n, np.nan)
    mkt = out["market"].to_numpy()

    for market, rate_col in _BAT_RATE.items():
        m = mkt == market
        if not m.any():
            continue
        # Blend recent form with the longer window: 30 games is stable but
        # slow, 10 is responsive but noisy.
        r10 = out.get(f"{rate_col}_r10")
        r30 = out.get(f"{rate_col}_r30")
        exp = out.get(f"{rate_col}_exp")
        rate = _blend(r10, r30, exp)
        pa = _blend(out.get("bat_pa_r10"), out.get("bat_pa_r30"),
                    out.get("bat_pa_exp"))
        if rate is not None and pa is not None:
            proj = np.where(m, rate * pa, proj)

    for market, col in _PIT_RATE.items():
        m = mkt == market
        if not m.any():
            continue
        v = _blend(out.get(f"{col}_r3"), out.get(f"{col}_r8"),
                   out.get(f"{col}_exp"))
        if v is not None:
            proj = np.where(m, v, proj)

    out["proj_mean"] = proj
    line = out["line"].to_numpy(dtype=float)
    out["line_minus_proj"] = line - proj
    with np.errstate(divide="ignore", invalid="ignore"):
        out["line_over_proj"] = np.where(proj > 0, line / proj, np.nan)
    return out


def _blend(*series, weights=(0.3, 0.5, 0.2)):
    """Weighted blend of rolling windows, skipping any that are absent."""
    cols = [s for s in series if s is not None]
    if not cols:
        return None
    arrs = [pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            for s in cols]
    w = np.array(weights[:len(arrs)], dtype=float)
    stack = np.vstack(arrs)
    mask = np.isfinite(stack)
    wmat = np.where(mask, w[:, None], 0.0)
    tot = wmat.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(tot > 0,
                       np.nansum(np.where(mask, stack, 0.0) * wmat, axis=0) / tot,
                       np.nan)
    return out


def feature_columns(props: pd.DataFrame, market: str) -> list[str]:
    """Features appropriate to one market, dropping all-null columns."""
    m = C.BY_NAME.get(market)
    cols = list(MARKET_STRUCT) + ["is_home_player", "logit_cons"]
    if m is None:
        return [c for c in cols if c in props.columns]
    if m.side == "batting":
        cols += [c for c in props.columns if c.startswith(("bat_", "tm_"))]
        cols += [c for c in props.columns if c.startswith("park_")]
    elif m.side == "pitching":
        cols += [c for c in props.columns if c.startswith(("pit_", "park_"))]
        cols += [c for c in props.columns if c.startswith(("home_tg_", "away_tg_"))]
    else:
        cols += [c for c in props.columns
                 if c.startswith(("home_tg_", "away_tg_", "park_"))]
    cols = [c for c in dict.fromkeys(cols) if c in props.columns]
    sub = props[props["market"] == market]
    keep = [c for c in cols
            if sub[c].notna().any() and sub[c].nunique(dropna=True) > 1]
    return keep


# ---------------------------------------------------------------------------
# Betting grain
# ---------------------------------------------------------------------------

def book_primary_anchor(dec: pd.DataFrame, primary: pd.DataFrame
                        ) -> pd.DataFrame:
    """Each book's *leave-one-out* consensus at its group's primary line.

    When a proposition is priced off a different line, the anchor must still
    exclude the book being judged -- otherwise a book that is an outlier at the
    primary line quietly sets the benchmark for its own alternate-line price.
    Books that do not quote the primary line contributed nothing to it, so the
    all-book value is already leave-one-out for them.
    """
    p = primary[["event_id", "market", "subject", "primary_line"]]
    at_primary = dec.merge(p, on=["event_id", "market", "subject"], how="inner")
    at_primary = at_primary[at_primary["line"] == at_primary["primary_line"]]
    if at_primary.empty:
        return pd.DataFrame(columns=["event_id", "market", "subject", "book",
                                     "p_primary_loo"])
    return (at_primary.groupby(["event_id", "market", "subject", "book"],
                               dropna=False, observed=True)["p_cons"]
            .median().rename("p_primary_loo").reset_index())


def build_candidates(graded: pd.DataFrame, cons: pd.DataFrame | None = None,
                     tag: str = "decision", props: pd.DataFrame | None = None,
                     ladder_table: dict | None = None) -> pd.DataFrame:
    """Best bettable price per (proposition, side), with that book's LOO
    consensus attached."""
    from . import ladder as LAD

    q = cons if cons is not None else attach_consensus(graded)
    d = q[(q["tag"] == tag) & q["book"].isin(C.BETTABLE_BOOKS)]
    d = d[~d["book"].isin(C.DFS_BOOKS)]

    # Fill a missing same-line consensus from the ladder, anchored on this
    # book's own leave-one-out view of the primary line.
    tbl = ladder_table if ladder_table is not None else LAD.load()
    if tbl and props is not None and not props.empty and not d.empty:
        pcols = [c for c in ("primary_line", "p_primary", "n_books_primary")
                 if c in props.columns]
        prim = props.drop_duplicates(subset=LAD.GROUP_KEY)[LAD.GROUP_KEY + pcols]
        loo = book_primary_anchor(q[q["tag"] == tag], prim)
        d = d.merge(prim, on=LAD.GROUP_KEY, how="left")
        d = d.merge(loo, on=["event_id", "market", "subject", "book"],
                    how="left")
        d["p_primary_eff"] = d["p_primary_loo"].fillna(d["p_primary"])
        need = d["p_cons"].isna() & d["p_primary_eff"].notna()
        if need.any():
            filled = LAD.apply(d.loc[need], tbl, p_primary_col="p_primary_eff")
            d.loc[need, "p_cons"] = filled
            # A ladder-derived anchor is exactly as well supported as the
            # primary line it was derived from -- not more. Claiming a fixed
            # minimum here would smuggle unsupported propositions past the
            # book-count filter.
            if "n_books_primary" in d.columns:
                d.loc[need, "n_books_cons"] = d.loc[need, "n_books_primary"]
            else:
                d.loc[need, "n_books_cons"] = np.nan
        d["anchor_src"] = np.where(need, "ladder", "direct")

    d = d.dropna(subset=["p_cons"])
    if d.empty:
        return d
    # Best price = the one that pays most. American odds are not monotonic as
    # integers across the sign boundary, so rank on decimal payout.
    from .odds import profit_per_unit
    d = d.assign(_payout=profit_per_unit(d["price"].to_numpy(dtype=float)))
    best = (d.sort_values("_payout", ascending=False)
              .drop_duplicates(subset=PROP_KEY + ["side"], keep="first")
              .drop(columns=["_payout"]))
    return best.reset_index(drop=True)
