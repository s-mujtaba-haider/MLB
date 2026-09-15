"""Point-in-time features.

The decision instant is first pitch minus 30 minutes. Everything here must be
knowable then, and the enforcement is structural rather than a promise: every
rolling aggregate is computed on a frame sorted by game date, grouped by
entity, and **shifted by one game** before any window is applied. The current
game can therefore never enter its own features.

Same-day games are excluded too, not just the current one. A team can play a
doubleheader, and game one finishes before game two starts; using a date-level
cutoff rather than a timestamp cutoff gives up a sliver of real information in
exchange for an airtight guarantee. That trade is the right way round.

What is deliberately *not* used:

* **Actual batting order.** It is public at the decision time, but our only
  source for it is the play-by-play, which exists only for players who
  actually batted. Keying on it would leak the fact that the player appeared
  at all. A rolling average of recent batting order is used instead.
* **Anything from the boxscore of the game being predicted**, obviously,
  including whether the player appeared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (10, 30, 90)


def _roll(df: pd.DataFrame, by: str, cols: list[str], windows=WINDOWS,
          prefix: str = "") -> pd.DataFrame:
    """Lagged rolling means: for each entity, the mean of prior games only."""
    out = {}
    g = df.groupby(by, observed=True, sort=False)
    for c in cols:
        prior = g[c].shift(1)          # <- the leak guard
        pg = prior.groupby(df[by], observed=True, sort=False)
        for w in windows:
            out[f"{prefix}{c}_r{w}"] = pg.transform(
                lambda s, w=w: s.rolling(w, min_periods=max(2, w // 5)).mean())
        out[f"{prefix}{c}_exp"] = pg.transform(
            lambda s: s.expanding(min_periods=3).mean())
    return pd.DataFrame(out, index=df.index)


def _game_frame(games: pd.DataFrame) -> pd.DataFrame:
    g = games[["espn_id", "game_date", "venue", "home_team", "away_team",
               "home_score", "away_score", "total_runs", "home_margin"]].copy()
    g["game_date"] = g["game_date"].astype(str)
    return g


# ---------------------------------------------------------------------------
# Batters
# ---------------------------------------------------------------------------

BAT_RATE_COLS = ["hits", "total_bases", "home_runs", "rbi", "runs",
                 "strike_outs", "walks"]


def batter_form(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (athlete, game) with that batter's prior-games form."""
    g = _game_frame(games)
    b = players[players["group"] == "batting"].merge(
        g[["espn_id", "game_date", "venue", "home_team", "away_team"]],
        left_on="event_id", right_on="espn_id", how="inner")
    b = b.sort_values(["athlete_id", "game_date", "event_id"]).reset_index(drop=True)

    b["pa"] = b["at_bats"].fillna(0) + b["walks"].fillna(0)
    b["pa"] = b["pa"].clip(lower=0)
    # Per-PA rates are the stable quantity; per-game totals confound form with
    # playing time, which the market already knows about separately.
    for c in BAT_RATE_COLS:
        b[f"{c}_ppa"] = np.where(b["pa"] > 0, b[c].fillna(0) / b["pa"], np.nan)

    rate_cols = [f"{c}_ppa" for c in BAT_RATE_COLS]
    feats = _roll(b, "athlete_id", rate_cols + ["pa"], prefix="bat_")

    # Games of prior experience -- a proxy for how trustworthy the form is.
    feats["bat_prior_games"] = (b.groupby("athlete_id", observed=True)
                                 .cumcount())
    # Days of rest since the player's previous appearance.
    d = pd.to_datetime(b["game_date"])
    feats["bat_days_rest"] = (d - d.groupby(b["athlete_id"]).shift(1)).dt.days
    out = pd.concat([b[["event_id", "athlete_id", "game_date", "team"]], feats],
                    axis=1)
    return out


# ---------------------------------------------------------------------------
# Pitchers
# ---------------------------------------------------------------------------

PIT_COLS = ["outs", "strike_outs", "hits_allowed", "earned_runs", "walks",
            "home_runs_allowed", "pitch_count"]


def pitcher_form(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    g = _game_frame(games)
    p = players[players["group"] == "pitching"].merge(
        g[["espn_id", "game_date"]], left_on="event_id", right_on="espn_id",
        how="inner")
    p = p.sort_values(["athlete_id", "game_date", "event_id"]).reset_index(drop=True)

    # A start is an outing of real length; relief cameos distort per-start form.
    p["is_start"] = (p["outs"] >= 9).astype(float)
    p["k_per_out"] = np.where(p["outs"] > 0, p["strike_outs"] / p["outs"], np.nan)
    p["pitch_per_out"] = np.where(p["outs"] > 0,
                                  p["pitch_count"] / p["outs"], np.nan)

    cols = PIT_COLS + ["k_per_out", "pitch_per_out", "is_start"]
    feats = _roll(p, "athlete_id", cols, windows=(3, 8, 20), prefix="pit_")
    feats["pit_prior_games"] = p.groupby("athlete_id", observed=True).cumcount()
    d = pd.to_datetime(p["game_date"])
    feats["pit_days_rest"] = (d - d.groupby(p["athlete_id"]).shift(1)).dt.days
    return pd.concat([p[["event_id", "athlete_id", "game_date", "team"]], feats],
                     axis=1)


# ---------------------------------------------------------------------------
# Teams, opponents and parks
# ---------------------------------------------------------------------------

def team_form(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Team-level offensive rates from prior games (per plate appearance)."""
    g = _game_frame(games)
    b = players[players["group"] == "batting"].merge(
        g[["espn_id", "game_date"]], left_on="event_id", right_on="espn_id",
        how="inner")
    agg = (b.groupby(["team", "event_id", "game_date"], observed=True)
             .agg(**{c: (c, "sum") for c in BAT_RATE_COLS},
                  at_bats=("at_bats", "sum"), pa_sum=("at_bats", "sum"))
             .reset_index())
    agg["pa_sum"] = agg["pa_sum"] + agg["walks"]
    for c in BAT_RATE_COLS:
        agg[f"tm_{c}_ppa"] = np.where(agg["pa_sum"] > 0,
                                      agg[c] / agg["pa_sum"], np.nan)
    agg = agg.sort_values(["team", "game_date", "event_id"]).reset_index(drop=True)
    feats = _roll(agg, "team", [f"tm_{c}_ppa" for c in BAT_RATE_COLS],
                  windows=(10, 30), prefix="")
    return pd.concat([agg[["team", "event_id", "game_date"]], feats], axis=1)


def park_form(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling runs environment by venue, from prior games at that venue."""
    g = _game_frame(games).sort_values(["venue", "game_date", "espn_id"])
    g = g.reset_index(drop=True)
    feats = _roll(g, "venue", ["total_runs"], windows=(30, 120), prefix="park_")
    return pd.concat([g[["espn_id", "venue", "game_date"]], feats], axis=1)


def game_context(games: pd.DataFrame) -> pd.DataFrame:
    """Team-level rolling run scoring/allowing, used by the featured markets."""
    g = _game_frame(games)
    long = pd.concat([
        g.assign(team=g["home_team"], opp=g["away_team"], is_home=1.0,
                 runs_for=g["home_score"], runs_against=g["away_score"]),
        g.assign(team=g["away_team"], opp=g["home_team"], is_home=0.0,
                 runs_for=g["away_score"], runs_against=g["home_score"]),
    ], ignore_index=True)
    long["won"] = (long["runs_for"] > long["runs_against"]).astype(float)
    long = long.sort_values(["team", "game_date", "espn_id"]).reset_index(drop=True)
    feats = _roll(long, "team", ["runs_for", "runs_against", "won"],
                  windows=(10, 30, 90), prefix="tg_")
    return pd.concat([long[["espn_id", "team", "opp", "is_home", "game_date"]],
                      feats], axis=1)


# ---------------------------------------------------------------------------
# Starting pitcher identification (point-in-time)
# ---------------------------------------------------------------------------

def current_form(players: pd.DataFrame, games: pd.DataFrame,
                 as_of_date: str, event_id: str = "__PENDING__"
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form for a game that has not happened yet.

    A placeholder appearance dated `as_of_date` is appended for every player,
    then the ordinary (shifted) form pipeline is run and those placeholder rows
    are returned. Reusing the exact backtest code path rather than writing a
    parallel "live" version is deliberate: a separate implementation is how
    train/serve skew gets in, and skew here would silently change what the
    model's inputs mean between validation and production.
    """
    stub_game = pd.DataFrame([{
        "espn_id": event_id, "game_date": as_of_date, "venue": None,
        "home_team": None, "away_team": None, "home_score": np.nan,
        "away_score": np.nan, "total_runs": np.nan, "home_margin": np.nan,
        "date": f"{as_of_date}T00:00:00Z",
    }])
    games2 = pd.concat([games, stub_game], ignore_index=True)

    ath = players[["athlete_id", "group", "team"]].drop_duplicates(
        subset=["athlete_id", "group"])
    stub = ath.assign(event_id=event_id)
    for c in players.columns:
        if c not in stub.columns:
            stub[c] = np.nan
    players2 = pd.concat([players, stub[players.columns]], ignore_index=True)

    bat = batter_form(players2, games2)
    pit = pitcher_form(players2, games2)
    return (bat[bat["event_id"] == event_id].drop(columns=["event_id"]),
            pit[pit["event_id"] == event_id].drop(columns=["event_id"]))


def starters_from_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Who is starting, inferred from the market itself.

    Books quote `pitcher_strikeouts` and `pitcher_outs` only for the announced
    starters, so the presence of those props at the decision snapshot *is* the
    point-in-time announcement. This avoids reading the starter out of the
    boxscore, which would be reading it out of the future.
    """
    p = quotes[(quotes["market"].isin(["pitcher_strikeouts", "pitcher_outs"]))
               & quotes["athlete_id"].notna()]
    if p.empty:
        return pd.DataFrame(columns=["event_id", "athlete_id", "n_quotes"])
    return (p.groupby(["event_id", "athlete_id"], observed=True)
              .size().rename("n_quotes").reset_index()
              .sort_values(["event_id", "n_quotes"], ascending=[True, False]))
