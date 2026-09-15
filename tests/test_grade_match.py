"""Settlement and joining -- the two places a backfill corrupts itself quietly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlbedge import grade as G
from mlbedge import match as M


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def test_accents_and_suffixes_normalise():
    assert M.norm_player("José Ramírez") == "jose ramirez"
    assert M.norm_player("Ronald Acuña Jr.") == "ronald acuna"
    assert M.norm_player("Vladimir Guerrero Jr") == "vladimir guerrero"
    assert M.norm_player("Lourdes Gurriel Jr.") == "lourdes gurriel"


def test_birth_year_qualifier_is_stripped():
    """Books disambiguate two Max Muncys with a birth year; the match is
    already scoped to one game, which resolves it."""
    assert M.norm_player("Max Muncy (2002)") == "max muncy"
    assert M.norm_player("Max Muncy") == "max muncy"


def test_punctuation_and_initials():
    assert M.norm_player("J.T. Realmuto") == "jt realmuto"
    assert M.norm_player("Logan O'Hoppe") == "logan o'hoppe"
    assert M.initial_last("Gunnar Henderson") == "g henderson"


def test_franchise_renames_and_relocations_unify():
    assert M.norm_team("Oakland Athletics") == M.norm_team("Sacramento Athletics")
    assert M.norm_team("Cleveland Indians") == M.norm_team("Cleveland Guardians")
    assert M.norm_team("Florida Marlins") == M.norm_team("Miami Marlins")


# ---------------------------------------------------------------------------
# Doubleheaders
# ---------------------------------------------------------------------------

def test_doubleheader_halves_do_not_collapse_onto_one_boxscore():
    """Two games, same date, same teams. Each odds event must claim its own
    ESPN game -- letting both match the same boxscore double-grades one game
    and silently drops the other."""
    events = pd.DataFrame([
        {"event_id": "g1", "commence_time": "2025-07-04T17:05:00Z",
         "home_team": "Detroit Tigers", "away_team": "Chicago White Sox"},
        {"event_id": "g2", "commence_time": "2025-07-04T23:10:00Z",
         "home_team": "Detroit Tigers", "away_team": "Chicago White Sox"},
    ])
    games = pd.DataFrame([
        {"espn_id": "E1", "date": "2025-07-04T17:10:00Z",
         "home_team": "Detroit Tigers", "away_team": "Chicago White Sox"},
        {"espn_id": "E2", "date": "2025-07-04T23:15:00Z",
         "home_team": "Detroit Tigers", "away_team": "Chicago White Sox"},
    ])
    out = M.match_games(events, games)
    got = dict(zip(out["event_id"], out["espn_id"]))
    assert got["g1"] == "E1"
    assert got["g2"] == "E2"
    assert len(set(got.values())) == 2


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows)


GAMES = _frame([{"espn_id": "E1", "home_score": 5, "away_score": 3,
                 "total_runs": 8, "home_margin": 2}])
PLAYERS = _frame([
    {"event_id": "E1", "athlete_id": "A", "group": "batting", "hits": 2.0,
     "total_bases": 5.0, "home_runs": 1.0, "rbi": 3.0, "runs": 1.0,
     "strike_outs": 1.0, "at_bats": 4.0, "walks": 0.0, "pitches_seen": 15.0},
    {"event_id": "E1", "athlete_id": "B", "group": "batting", "hits": 0.0,
     "total_bases": 0.0, "home_runs": 0.0, "rbi": 0.0, "runs": 0.0,
     "strike_outs": 0.0, "at_bats": 0.0, "walks": 0.0, "pitches_seen": 0.0},
    {"event_id": "E1", "athlete_id": "P", "group": "pitching",
     "strike_outs": 7.0, "outs": 18.0},
])


def _quote(market, side, line, athlete=None, espn="E1"):
    return {"event_id": "x", "espn_id": espn, "market": market, "side": side,
            "line": line, "price": -110, "athlete_id": athlete,
            "subject": athlete or "game", "book": "dk", "tag": "decision"}


def test_totals_push_on_the_exact_number():
    q = _frame([_quote("totals", "over", 8.0), _quote("totals", "under", 8.0),
                _quote("totals", "over", 7.5), _quote("totals", "under", 8.5)])
    r = G.grade(q, GAMES, PLAYERS)["result"].tolist()
    assert r == [G.PUSH, G.PUSH, G.WIN, G.WIN]


def test_spread_is_settled_from_the_home_perspective():
    """`line` is stored as the home handicap on both rows, so the two sides
    share a proposition key. Home won by 2."""
    q = _frame([_quote("spreads", "home", -1.5), _quote("spreads", "away", -1.5),
                _quote("spreads", "home", -2.0), _quote("spreads", "away", -2.0),
                _quote("spreads", "home", -2.5)])
    r = G.grade(q, GAMES, PLAYERS)["result"].tolist()
    assert r == [G.WIN, G.LOSS, G.PUSH, G.PUSH, G.LOSS]


def test_moneyline_has_no_push_when_someone_won():
    q = _frame([_quote("h2h", "home", np.nan), _quote("h2h", "away", np.nan)])
    assert G.grade(q, GAMES, PLAYERS)["result"].tolist() == [G.WIN, G.LOSS]


def test_player_props_settle_against_the_boxscore():
    q = _frame([
        _quote("batter_hits", "over", 1.5, "A"),
        _quote("batter_total_bases", "over", 4.5, "A"),
        _quote("batter_home_runs", "over", 0.5, "A"),
        _quote("batter_rbis", "under", 2.5, "A"),
        _quote("pitcher_strikeouts", "over", 6.5, "P"),
        _quote("pitcher_outs", "under", 17.5, "P"),
    ])
    r = G.grade(q, GAMES, PLAYERS)["result"].tolist()
    assert r == [G.WIN, G.WIN, G.WIN, G.LOSS, G.WIN, G.LOSS]


def test_prop_on_an_integer_line_pushes():
    q = _frame([_quote("pitcher_strikeouts", "over", 7.0, "P"),
                _quote("pitcher_outs", "over", 18.0, "P")])
    assert G.grade(q, GAMES, PLAYERS)["result"].tolist() == [G.PUSH, G.PUSH]


def test_absent_player_voids_rather_than_loses():
    """The single most important settlement rule. A player who never appeared
    has his props returned by the book; grading them as losses is a silent,
    systematic pessimism that makes every backtest look worse than reality --
    and grading them as wins is fraud."""
    q = _frame([_quote("batter_hits", "over", 0.5, "GHOST")])
    g = G.grade(q, GAMES, PLAYERS)
    assert g["result"].iloc[0] == G.VOID
    assert not g["graded"].iloc[0]


def test_player_with_no_plate_appearance_voids():
    """Player B is in the boxscore (defensive replacement) but never batted."""
    q = _frame([_quote("batter_hits", "under", 0.5, "B")])
    assert G.grade(q, GAMES, PLAYERS)["result"].iloc[0] == G.VOID


def test_game_that_never_completed_voids_every_quote():
    q = _frame([_quote("totals", "over", 8.0, espn="MISSING"),
                _quote("batter_hits", "over", 0.5, "A", espn="MISSING")])
    assert set(G.grade(q, GAMES, PLAYERS)["result"]) == {G.VOID}


def test_void_and_push_both_return_the_stake():
    res = pd.Series([G.WIN, G.LOSS, G.PUSH, G.VOID])
    price = pd.Series([100.0, 100.0, 100.0, 100.0])
    assert G.settle_profit(res, price).tolist() == [1.0, -1.0, 0.0, 0.0]


def test_grading_report_accounts_for_every_quote():
    q = _frame([_quote("totals", "over", 8.0), _quote("batter_hits", "over", 0.5, "A"),
                _quote("batter_hits", "over", 0.5, "GHOST")])
    rep = G.grading_report(G.grade(q, GAMES, PLAYERS))
    assert (rep["graded"] + rep[G.VOID] == rep["quotes"]).all()
