"""Settle every quote against the boxscore.

Four outcomes, and the distinction between the last two is what keeps a
backtest honest:

  win   the side cashed
  loss  the side lost
  push  the statistic landed exactly on the line -- stake returned
  void  no action: the game did not complete, or the player never took the
        field. The book returns the stake; grading these as losses is a
        silent, systematic pessimism, and grading them as wins is fraud.

A prop is voided when the named player has no boxscore row for that game, or
(for batters) took no plate appearance. That is how US books actually settle
them, and it is common: roughly 2% of quoted batter props in a sample slate
belong to players who never appeared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

WIN, LOSS, PUSH, VOID = "win", "loss", "push", "void"

# Market.stat -> the column on the player-game row that settles it.
_BATTING_STAT = {
    "hits": "hits",
    "rbi": "rbi",
    "total_bases": "total_bases",
    "home_runs": "home_runs",
    "runs": "runs",
    "strike_outs": "strike_outs",
}
_PITCHING_STAT = {
    "strike_outs": "strike_outs",
    "outs": "outs",
}


def _player_lookup(players: pd.DataFrame) -> dict:
    """(event_id, athlete_id, group) -> stat row."""
    out = {}
    for r in players.to_dict("records"):
        out[(r["event_id"], r["athlete_id"], r["group"])] = r
    return out


def grade(quotes: pd.DataFrame, games: pd.DataFrame,
          players: pd.DataFrame) -> pd.DataFrame:
    """Attach `stat_value` and `result` to every quote row."""
    q = quotes.copy()
    gidx = games.set_index("espn_id")[
        ["home_score", "away_score", "total_runs", "home_margin"]].to_dict("index")
    pidx = _player_lookup(players)

    stat_value = np.full(len(q), np.nan)
    result = np.array([VOID] * len(q), dtype=object)

    markets = q["market"].to_numpy()
    sides = q["side"].to_numpy()
    lines = q["line"].to_numpy(dtype=float)
    espn = q["espn_id"].to_numpy()
    ath = q["athlete_id"].to_numpy() if "athlete_id" in q else np.full(len(q), None)

    for i in range(len(q)):
        eid = espn[i]
        if eid is None or (isinstance(eid, float) and np.isnan(eid)):
            continue
        g = gidx.get(eid)
        if g is None:
            continue  # game never completed -> void
        m = C.BY_NAME.get(markets[i])
        if m is None:
            continue

        side, line = sides[i], lines[i]

        if m.kind == "featured":
            if m.name == "h2h":
                margin = g["home_margin"]
                if margin == 0:
                    result[i] = PUSH  # MLB has no ties, but never assume
                    continue
                won = (margin > 0) if side == "home" else (margin < 0)
                stat_value[i] = margin
                result[i] = WIN if won else LOSS
            elif m.name == "spreads":
                if np.isnan(line):
                    continue
                # `line` is stored from the home perspective on both rows.
                adj = g["home_margin"] + line
                stat_value[i] = g["home_margin"]
                if adj == 0:
                    result[i] = PUSH
                else:
                    home_covers = adj > 0
                    won = home_covers if side == "home" else not home_covers
                    result[i] = WIN if won else LOSS
            elif m.name == "totals":
                if np.isnan(line):
                    continue
                tot = g["total_runs"]
                stat_value[i] = tot
                if tot == line:
                    result[i] = PUSH
                else:
                    over = tot > line
                    won = over if side == "over" else not over
                    result[i] = WIN if won else LOSS
            continue

        # --- player props --------------------------------------------------
        aid = ath[i]
        if aid is None or (isinstance(aid, float) and np.isnan(aid)):
            continue  # player never appeared -> void
        group = "batting" if m.side == "batting" else "pitching"
        row = pidx.get((eid, aid, group))
        if row is None:
            continue
        if group == "batting":
            # No plate appearance -> no action, exactly as the books settle it.
            if float(row.get("pitches_seen") or 0) <= 0 and \
               float(row.get("at_bats") or 0) <= 0 and \
               float(row.get("walks") or 0) <= 0:
                continue
            col = _BATTING_STAT.get(m.stat)
        else:
            col = _PITCHING_STAT.get(m.stat)
        if col is None:
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        val = float(val)
        stat_value[i] = val
        if np.isnan(line):
            continue
        if val == line:
            result[i] = PUSH
        else:
            over = val > line
            won = over if side == "over" else not over
            result[i] = WIN if won else LOSS

    q["stat_value"] = stat_value
    q["result"] = result
    q["graded"] = q["result"].isin([WIN, LOSS, PUSH])
    return q


def settle_profit(result: pd.Series, price: pd.Series) -> pd.Series:
    """Profit per 1-unit stake. Push and void both return zero profit."""
    from .odds import profit_per_unit
    win = pd.Series(profit_per_unit(price.to_numpy(dtype=float)),
                    index=price.index)
    out = pd.Series(0.0, index=result.index)
    out[result == WIN] = win[result == WIN]
    out[result == LOSS] = -1.0
    return out


def grading_report(graded: pd.DataFrame) -> pd.DataFrame:
    """Per-market settlement audit: how much of the quoted universe actually
    graded, and how the rest broke down."""
    g = graded.copy()
    tot = g.groupby("market").size().rename("quotes")
    piv = (g.pivot_table(index="market", columns="result", values="price",
                         aggfunc="size", fill_value=0))
    out = piv.join(tot)
    for c in (WIN, LOSS, PUSH, VOID):
        if c not in out:
            out[c] = 0
    out["graded"] = out[WIN] + out[LOSS] + out[PUSH]
    out["graded_pct"] = (out["graded"] / out["quotes"] * 100).round(2)
    out["void_pct"] = (out[VOID] / out["quotes"] * 100).round(2)
    out["push_pct"] = (out[PUSH] / out["quotes"] * 100).round(2)
    return out[[ "quotes", "graded", "graded_pct", WIN, LOSS, PUSH, VOID,
                 "void_pct", "push_pct"]].sort_index()
