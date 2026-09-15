"""Central configuration: markets, books, regions, seasons, paths.

Everything the rest of the pipeline keys off lives here, so that a market
definition exists in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CURATED = DATA / "curated"
REPORTS = ROOT / "reports"
for _p in (RAW, CURATED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)


def api_key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ODDS_API_KEY not set (env var or .env file)")


SPORT = "baseball_mlb"


@dataclass(frozen=True)
class Market:
    """One tradeable market.

    name     -- our canonical name (what the client asked for)
    api_key  -- The Odds API market key
    kind     -- 'featured' (whole-slate endpoint) or 'prop' (per-event endpoint)
    stat     -- the graded statistic, resolved by grade.py against boxscores
    side     -- 'batting' | 'pitching' | 'team'
    regions  -- Odds API regions worth pulling for this market
    """

    name: str
    api_key: str
    kind: str
    stat: str
    side: str
    regions: tuple[str, ...]
    label: str = ""


MARKETS: tuple[Market, ...] = (
    # featured / game level
    Market("h2h", "h2h", "featured", "moneyline", "team",
           ("us", "eu", "us_ex"), "Moneyline"),
    Market("spreads", "spreads", "featured", "run_line", "team",
           ("us", "eu", "us_ex"), "Run line"),
    Market("totals", "totals", "featured", "total_runs", "team",
           ("us", "eu", "us_ex"), "Game total"),
    # batter props
    Market("batter_hits", "batter_hits", "prop", "hits", "batting",
           ("us", "eu", "us_ex", "us2"), "Batter hits"),
    Market("batter_rbis", "batter_rbis", "prop", "rbi", "batting",
           ("us", "eu", "us_ex", "us2"), "Batter RBIs"),
    Market("batter_total_bases", "batter_total_bases", "prop", "total_bases", "batting",
           ("us", "eu", "us_ex", "us2"), "Batter total bases"),
    Market("batter_home_runs", "batter_home_runs", "prop", "home_runs", "batting",
           ("us", "eu", "us_ex", "us2"), "Batter home runs"),
    Market("runs_scored", "batter_runs_scored", "prop", "runs", "batting",
           ("us", "eu", "us_ex", "us2"), "Batter runs scored"),
    # Thinnest market by far, and its coverage migrates between seasons: Fliff
    # (us2) quoted it through 2025, by 2026 it lives on the DFS books and the
    # exchanges. We sweep every region for it -- the API only bills for regions
    # that actually return data, so the empty ones are free.
    Market("batter_strikeouts", "batter_strikeouts", "prop", "strike_outs", "batting",
           ("us2", "us_dfs", "us_ex", "us", "eu"), "Batter strikeouts"),
    # pitcher props
    Market("pitcher_strikeouts", "pitcher_strikeouts", "prop", "strike_outs", "pitching",
           ("us", "eu", "us_ex", "us2"), "Pitcher strikeouts"),
    Market("pitcher_outs", "pitcher_outs", "prop", "outs", "pitching",
           ("us", "eu", "us_ex", "us2"), "Pitcher outs"),
)

BY_NAME = {m.name: m for m in MARKETS}
BY_API_KEY = {m.api_key: m for m in MARKETS}
PROP_MARKETS = tuple(m for m in MARKETS if m.kind == "prop")
FEATURED_MARKETS = tuple(m for m in MARKETS if m.kind == "featured")
ALL_MARKET_NAMES = tuple(m.name for m in MARKETS)

# Which API market keys to request per region on the per-event props endpoint.
# The API only bills for markets that actually return data, so an over-broad
# request costs latency rather than credits -- but we keep these tight anyway.
PROP_REQUEST_PLAN: dict[str, tuple[str, ...]] = {
    "us": ("batter_hits", "batter_rbis", "batter_total_bases", "batter_home_runs",
           "batter_runs_scored", "pitcher_strikeouts", "pitcher_outs"),
    "eu": ("batter_hits", "batter_rbis", "batter_total_bases", "batter_home_runs",
           "batter_runs_scored", "pitcher_strikeouts", "pitcher_outs"),
    "us_ex": ("batter_hits", "batter_rbis", "batter_total_bases", "batter_home_runs",
              "batter_runs_scored", "pitcher_strikeouts", "pitcher_outs"),
    "us2": ("batter_strikeouts",),
    "us_dfs": ("batter_strikeouts",),
}

# DFS pick'em operators quote a line but not a genuinely tradeable two-way
# price. Their lines are a useful consensus signal for a starved market; they
# are never treated as a book we can bet into.
DFS_BOOKS = {"prizepicks", "underdog", "pick6", "betr_us_dfs", "dabble_au"}
FEATURED_REQUEST_PLAN: dict[str, tuple[str, ...]] = {
    "us": ("h2h", "spreads", "totals"),
    "eu": ("h2h", "spreads", "totals"),
    "us_ex": ("h2h", "spreads", "totals"),
}

# Alternate run lines and totals. Only on the per-event endpoint, never the
# whole-slate one, so they cost per event rather than per snapshot. Worth it:
# DraftKings posts roughly thirty alternate spread and thirty alternate total
# outcomes per game against two on the main line, and the alternate rungs are
# the ones books leave alone once they are up.
# Measured coverage per region on a sample game (outcomes returned, 20 credits
# each): us 286 (DraftKings 64, Caesars 66), us2 114 across the soft books
# where mispricing concentrates, eu 58 including Pinnacle as a sharp anchor,
# us_ex only 26. The first three earn their cost; us_ex does not.
ALT_REQUEST_PLAN: dict[str, tuple[str, ...]] = {
    "us": ("alternate_spreads", "alternate_totals"),
    "us2": ("alternate_spreads", "alternate_totals"),
    "eu": ("alternate_spreads", "alternate_totals"),
}
ALT_API_KEYS = {"alternate_spreads": "spreads", "alternate_totals": "totals"}
# An alternate rung is the same market at a different number, so it maps onto
# the same Market object and flows through the identical normalise/grade path.
for _alt, _canon in ALT_API_KEYS.items():
    BY_API_KEY[_alt] = BY_NAME[_canon]

# --- bookmakers -----------------------------------------------------------
# Sharpness tier drives consensus weighting and which books may anchor a price.
#   0 : zero/low-vig exchanges -- closest thing to a true probability
#   1 : sharp books -- Pinnacle and the reduced-juice shops
#   2 : mainstream US retail -- deep liquidity, some bias
#   3 : soft / offshore / promotional -- where mispricing tends to live
EXCHANGES = {"novig", "prophetx", "betfair_ex_eu", "matchbook", "betopenly"}
SHARP = {"pinnacle", "lowvig", "betonlineag", "betus", "mybookieag"}
RETAIL = {"draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers",
          "espnbet", "fanatics", "hardrockbet", "bovada"}


def book_tier(key: str) -> int:
    if key in EXCHANGES:
        return 0
    if key in SHARP:
        return 1
    if key in RETAIL:
        return 2
    return 3


# Books a bet may actually be recorded against: real, reachable sportsbooks
# with meaningful limits. Consensus is built from everything; bets are only
# ever priced at one of these.
BETTABLE_BOOKS = {
    "draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "espnbet",
    "fanatics", "hardrockbet", "bovada", "betonlineag", "lowvig", "betus",
    "mybookieag", "novig", "prophetx", "fliff", "ballybet", "betparx",
    "rebet", "windcreek", "pinnacle",
}

# Relative precision of each book as a probability anchor. Used by
# devig.consensus(); these are priors, refit on train folds by calibrate.py.
ANCHOR_WEIGHTS = {
    "pinnacle": 1.00,
    "novig": 0.85,
    "prophetx": 0.80,
    "betfair_ex_eu": 0.75,
    "matchbook": 0.60,
    "lowvig": 0.55,
    "betonlineag": 0.50,
    "draftkings": 0.45,
    "fanduel": 0.45,
    "betmgm": 0.40,
    "williamhill_us": 0.30,
    "espnbet": 0.25,
    "betrivers": 0.25,
    "fanatics": 0.25,
    "hardrockbet": 0.20,
    "bovada": 0.20,
}
DEFAULT_ANCHOR_WEIGHT = 0.10

# Markets where, for most of their history, exactly one bookmaker in the world
# quotes the line. There is no cross-book benchmark to build, so that book's
# own vig-free price anchors and the edge has to come from the projection
# rather than from shopping the number. See devig.attach_consensus.
SELF_ANCHOR_MARKETS = frozenset({"batter_strikeouts"})

# How many *other* books must stand behind a price before it can be bet.
# Two, not three: the selection calibrator now conditions on the book count and
# discounts a thin consensus on its own, which is a better instrument than a
# hard cutoff. A blanket minimum of three threw away real volume in exchange
# for a judgement the correction already makes more precisely.
DEFAULT_MIN_BOOKS = 2
MIN_BOOKS = {"batter_strikeouts": 1}


def min_books_for(market: str) -> int:
    return MIN_BOOKS.get(market, DEFAULT_MIN_BOOKS)


# The books a typical US bettor can actually reach and hold a balance at.
# Reported separately because "best price across twenty books" overstates what
# is executable: an edge that survives only at obscure or offshore shops is a
# different product from one that is available at DraftKings.
MAJOR_BOOKS = {"draftkings", "fanduel", "betmgm", "williamhill_us",
               "espnbet", "betrivers", "fanatics"}

# --- seasons & snapshot timing -------------------------------------------
# Regular-season windows, inclusive. Postseason is excluded: different roster,
# bullpen and lineup dynamics, and it contaminates rolling form features.
SEASONS: dict[int, tuple[str, str]] = {
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-30"),
    2025: ("2025-03-27", "2025-09-28"),
    2026: ("2026-03-26", "2026-09-27"),
}

# The decision moment: 30 minutes before first pitch. Starting lineups are
# public, prop markets are at their most liquid, and a human could actually
# place the bet. Every model input must be as-of at or before this instant --
# enforced by leakage.py.
DECISION_OFFSET_MIN = 30
# Closing snapshot, used only to measure CLV. Never an input to a decision.
CLOSING_OFFSET_MIN = 2
