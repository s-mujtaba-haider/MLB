"""Join the odds universe to the results universe.

Two joins, both of which are quiet data-corruption hazards if done naively:

**Games.** Odds API events carry team display names and a first-pitch time;
ESPN games carry the same. Matching on (Eastern game date, home team) is nearly
always right, but doubleheaders put two games on one date with identical team
names -- those are split by first-pitch proximity. Franchise relocations and
renames (the Athletics moved Oakland -> Sacramento; Cleveland Indians ->
Guardians) are handled by normalising to a stable franchise key.

**Players.** The Odds API gives a free-text `description` ("Gunnar Henderson");
ESPN gives a `displayName`. Matching is scoped to the two rosters that actually
played in that game, which turns an open-ended name-matching problem into a
choice among ~40 candidates. Accents, punctuation and generational suffixes are
normalised away; a bounded fuzzy fallback catches the rest. Anything still
unmatched is reported rather than silently dropped -- an unmatched player is an
ungraded bet, and ungraded bets are how a backtest flatters itself.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import pandas as pd

# Franchise aliases -> canonical key. Only entries that actually differ between
# the two feeds, or across seasons, need to appear here.
_TEAM_ALIASES = {
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "las vegas athletics": "athletics",
    "athletics": "athletics",
    "cleveland indians": "cleveland guardians",
    "tampa bay devil rays": "tampa bay rays",
    "florida marlins": "miami marlins",
    "washington nationals": "washington nationals",
    "st louis cardinals": "st. louis cardinals",
    "st. louis cardinals": "st. louis cardinals",
    "arizona diamondbacks": "arizona diamondbacks",
}

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm_team(name: str | None) -> str:
    if not name:
        return ""
    s = strip_accents(str(name)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return _TEAM_ALIASES.get(s, s)


def norm_player(name: str | None) -> str:
    """Aggressive normalisation for name comparison.

    'José Ramírez Jr.' -> 'jose ramirez'.  'Luis L. Ortiz' -> 'luis l ortiz'.
    """
    if not name:
        return ""
    s = strip_accents(str(name)).lower()
    s = s.replace("&apos;", "'").replace("`", "'")
    # Books disambiguate same-named players with a birth year: the Athletics'
    # "Max Muncy (2002)" vs the Dodgers' Max Muncy. Drop the qualifier; the
    # match is already scoped to one game, which resolves the ambiguity.
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9'. ]+", " ", s)
    parts = [p for p in re.split(r"\s+", s) if p]
    while parts and parts[-1].strip(".") in _SUFFIXES:
        parts.pop()
    return " ".join(parts).replace(".", "").strip()


def initial_last(name: str) -> str:
    """'gunnar henderson' -> 'g henderson'. Books often abbreviate."""
    p = norm_player(name).split()
    if len(p) < 2:
        return norm_player(name)
    return f"{p[0][0]} {p[-1]}"


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------

def match_games(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Return events with `espn_id` attached (NaN where unmatched)."""
    ev = events.copy()
    gm = games.copy()
    ev["_home"] = ev["home_team"].map(norm_team)
    ev["_away"] = ev["away_team"].map(norm_team)
    ev["_dt"] = pd.to_datetime(ev["commence_time"], utc=True, format="ISO8601")
    ev["_date"] = ev["_dt"].dt.tz_convert("America/New_York").dt.date.astype(str)

    gm["_home"] = gm["home_team"].map(norm_team)
    gm["_away"] = gm["away_team"].map(norm_team)
    gm["_dt"] = pd.to_datetime(gm["date"], utc=True, format="ISO8601")
    gm["_date"] = gm["_dt"].dt.tz_convert("America/New_York").dt.date.astype(str)

    cand = ev.merge(
        gm[["espn_id", "_home", "_away", "_date", "_dt"]],
        on=["_home", "_away", "_date"], how="left", suffixes=("", "_g"))

    # Doubleheaders produce duplicate candidates; keep the closest first pitch.
    cand["_gap"] = (cand["_dt_g"] - cand["_dt"]).abs()
    cand = (cand.sort_values("_gap")
                .drop_duplicates(subset=["event_id"], keep="first")
                .sort_index())

    # An ESPN game may only be claimed once -- otherwise both halves of a
    # doubleheader can collapse onto the same boxscore.
    claimed = cand[cand["espn_id"].notna()].copy()
    claimed = (claimed.sort_values("_gap")
                      .drop_duplicates(subset=["espn_id"], keep="first"))
    keep = set(claimed["event_id"])
    cand.loc[~cand["event_id"].isin(keep), "espn_id"] = None

    out = cand.drop(columns=[c for c in cand.columns
                             if c.startswith("_") or c.endswith("_g")])
    out["espn_id"] = cand["espn_id"]
    return out


# ---------------------------------------------------------------------------
# Player matching
# ---------------------------------------------------------------------------

class PlayerMatcher:
    """Match book player names to ESPN athletes, scoped to one game."""

    def __init__(self, players: pd.DataFrame):
        # espn_id -> {normalised name: athlete_id}
        self._by_game: dict[str, dict[str, str]] = {}
        self._alt: dict[str, dict[str, str]] = {}
        self._name: dict[str, str] = {}
        for eid, sub in players.groupby("event_id"):
            exact, alt = {}, {}
            for r in sub.itertuples():
                n = norm_player(r.name)
                if not n:
                    continue
                exact.setdefault(n, r.athlete_id)
                alt.setdefault(initial_last(r.name), r.athlete_id)
                # Last name alone, only when unambiguous within the game.
                last = n.split()[-1]
                if last in alt and alt[last] != r.athlete_id:
                    alt[last] = "__AMBIG__"
                else:
                    alt.setdefault(last, r.athlete_id)
                self._name[r.athlete_id] = r.name
            self._by_game[eid] = exact
            self._alt[eid] = alt

    def match(self, espn_id: str, book_name: str) -> tuple[str | None, str]:
        """Return (athlete_id, how). `how` records which rule fired, so the
        match-quality report can show what is carrying the join."""
        exact = self._by_game.get(espn_id)
        if not exact:
            return None, "no_game"
        n = norm_player(book_name)
        if n in exact:
            return exact[n], "exact"
        alt = self._alt.get(espn_id, {})
        il = initial_last(book_name)
        if il in alt and alt[il] != "__AMBIG__":
            return alt[il], "initial_last"
        last = n.split()[-1] if n else ""
        if last and alt.get(last) not in (None, "__AMBIG__"):
            return alt[last], "last_name"
        close = difflib.get_close_matches(n, list(exact), n=1, cutoff=0.88)
        if close:
            return exact[close[0]], "fuzzy"
        return None, "unmatched"


def attach_players(quotes: pd.DataFrame, players: pd.DataFrame,
                   event_to_espn: dict[str, str]) -> pd.DataFrame:
    """Attach athlete_id to every prop quote. Featured rows pass through."""
    m = PlayerMatcher(players)
    q = quotes.copy()
    q["espn_id"] = q["event_id"].map(event_to_espn)

    is_prop = q["subject"].ne("game")
    pairs = (q.loc[is_prop, ["espn_id", "subject"]]
               .drop_duplicates())
    resolved = {}
    for r in pairs.itertuples(index=False):
        resolved[(r.espn_id, r.subject)] = m.match(r.espn_id, r.subject)

    q["athlete_id"] = None
    q["match_how"] = "featured"
    idx = q.index[is_prop]
    keys = list(zip(q.loc[idx, "espn_id"], q.loc[idx, "subject"]))
    q.loc[idx, "athlete_id"] = [resolved[k][0] for k in keys]
    q.loc[idx, "match_how"] = [resolved[k][1] for k in keys]
    return q


def match_report(quotes: pd.DataFrame) -> pd.DataFrame:
    """Per-market view of how the player join was achieved."""
    p = quotes[quotes["subject"].ne("game")]
    if p.empty:
        return pd.DataFrame()
    g = (p.groupby(["market", "match_how"])
           .agg(rows=("price", "size"),
                players=("subject", "nunique")).reset_index())
    tot = g.groupby("market")["rows"].transform("sum")
    g["pct"] = (g["rows"] / tot * 100).round(2)
    return g.sort_values(["market", "rows"], ascending=[True, False])
