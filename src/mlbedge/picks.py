"""Daily pick generation against live odds.

Mirrors the backtest path exactly -- same normaliser, same consensus, same
feature builder, same models -- with one unavoidable difference: at decision
time the game has no boxscore, so players cannot be matched against the roster
that played. They are matched instead against recent appearances for the two
teams involved, which is the same problem scoped the same way.

Picks are written to a parquet ledger keyed by (date, market, subject, line,
side, book) so that re-running a day is idempotent and so that every pick
surfaced on the dashboard can later be graded against what actually happened.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from . import normalize as N
from .devig import attach_consensus, logit
from .match import norm_player
from .model import side_probability
from .odds import american_to_prob, ev_per_unit, kelly_fraction
from .oddsapi import OddsClient, run_pool
from .production import Bundle

PICKS_PATH = C.DATA / "picks.parquet"
UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Live odds
# ---------------------------------------------------------------------------

def todays_events(client: OddsClient, within_hours: float = 14.0
                  ) -> pd.DataFrame:
    """Upcoming MLB events inside the next `within_hours`."""
    r = client.live_events()
    rows = r.data or []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{ "event_id": e["id"],
                         "commence_time": e["commence_time"],
                         "home_team": e.get("home_team"),
                         "away_team": e.get("away_team")} for e in rows])
    df["commence_dt"] = pd.to_datetime(df["commence_time"], utc=True,
                                       format="ISO8601")
    now = pd.Timestamp.now(tz="UTC")
    df = df[(df["commence_dt"] > now) &
            (df["commence_dt"] <= now + pd.Timedelta(hours=within_hours))]
    df["game_date"] = (df["commence_dt"].dt.tz_convert("America/New_York")
                       .dt.date.astype(str))
    return df.sort_values("commence_dt").reset_index(drop=True)


def fetch_live_odds(events: pd.DataFrame, client: OddsClient,
                    markets: list[str], workers: int = 8) -> pd.DataFrame:
    """Pull current prices for the requested markets and normalise them."""
    want_api = {C.BY_NAME[m].api_key for m in markets if m in C.BY_NAME}
    rows: list[dict] = []

    feat = [m for m in markets if C.BY_NAME[m].kind == "featured"]
    if feat:
        for region, mks in C.FEATURED_REQUEST_PLAN.items():
            mks = [m for m in mks if m in want_api]
            if not mks:
                continue
            r = client.live_featured(region, mks)
            for ev in (r.data or []):
                rows.extend(_featured_rows(ev, region, events))

    prop = [m for m in markets if C.BY_NAME[m].kind == "prop"]
    if prop:
        jobs = []
        for e in events.itertuples():
            for region, mks in C.PROP_REQUEST_PLAN.items():
                mks = [m for m in mks if m in want_api]
                if mks:
                    jobs.append((e.event_id, region, tuple(mks),
                                 e.commence_time))

        def one(eid, region, mks, commence):
            r = client.live_event_odds(eid, region, list(mks))
            return _prop_rows(r.data, eid, region, commence) if r.data else []

        for chunk in run_pool(jobs, one, workers=workers, desc="live props",
                              progress_every=0):
            rows.extend(chunk)

    q = pd.DataFrame(rows)
    return N._finish(q) if not q.empty else q


def _lead(commence: str) -> float:
    ct = dt.datetime.fromisoformat(commence.replace("Z", "+00:00"))
    return (ct - dt.datetime.now(UTC)).total_seconds() / 60.0


def _prop_rows(data: dict, eid: str, region: str, commence: str) -> list[dict]:
    out = []
    now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lead = _lead(commence)
    for bk in data.get("bookmakers", []):
        for mk in bk.get("markets", []):
            m = C.BY_API_KEY.get(mk.get("key"))
            if m is None:
                continue
            for o in mk.get("outcomes", []):
                nm = (o.get("name") or "").strip().lower()
                side = "over" if nm in ("over", "yes") else (
                    "under" if nm in ("under", "no") else None)
                subj = (o.get("description") or "").strip()
                if side is None or not subj:
                    continue
                out.append({"event_id": eid, "market": m.name, "subject": subj,
                            "side": side, "line": o.get("point"),
                            "price": o.get("price"), "book": bk.get("key"),
                            "region": region, "snapshot_ts": now,
                            "commence_time": commence, "lead_min": lead,
                            "tag": "live"})
    return out


def _featured_rows(ev: dict, region: str, events: pd.DataFrame) -> list[dict]:
    meta = events.set_index("event_id")
    eid = ev.get("id")
    if eid not in meta.index:
        return []
    home = meta.at[eid, "home_team"]
    away = meta.at[eid, "away_team"]
    commence = meta.at[eid, "commence_time"]
    now = dt.datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lead = _lead(commence)
    out = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            m = C.BY_API_KEY.get(mk.get("key"))
            if m is None:
                continue
            out.extend(N._featured_outcomes(mk, m, eid, bk.get("key"), region,
                                            now, commence, lead, home, away,
                                            "live"))
    return out


# ---------------------------------------------------------------------------
# Live player matching
# ---------------------------------------------------------------------------

class LiveMatcher:
    """Name -> athlete_id using recent appearances.

    The game has not been played, so there is no roster to scope against.
    Recency is the substitute: a player who appeared in the last few weeks is a
    far better candidate than a namesake who last played in April.
    """

    def __init__(self, players: pd.DataFrame, games: pd.DataFrame,
                 lookback_days: int = 45):
        g = games[["espn_id", "game_date"]]
        p = players.merge(g, left_on="event_id", right_on="espn_id", how="left")
        cutoff = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
        recent = p[p["game_date"] >= cutoff]
        if recent.empty:
            recent = p
        recent = recent.sort_values("game_date")
        self._idx: dict[str, str] = {}
        self._team: dict[str, str] = {}
        for r in recent.itertuples():
            n = norm_player(r.name)
            if n:
                self._idx[n] = r.athlete_id     # later rows win: most recent
                self._team[r.athlete_id] = r.team

    def match(self, name: str) -> str | None:
        return self._idx.get(norm_player(name))

    def team(self, athlete_id: str) -> str | None:
        return self._team.get(athlete_id)


# ---------------------------------------------------------------------------
# Pick generation
# ---------------------------------------------------------------------------

def generate(bundles: dict[str, Bundle], players: pd.DataFrame,
             games: pd.DataFrame, client: OddsClient | None = None,
             within_hours: float = 14.0, min_lead_min: float = 20.0,
             kelly_cap: float = 0.02) -> pd.DataFrame:
    """Produce today's picks for every market the gate authorised to fire."""
    client = client or OddsClient()
    firing = {m: b for m, b in bundles.items() if b.fires}
    if not firing:
        print("no markets authorised to fire")
        return pd.DataFrame()

    events = todays_events(client, within_hours=within_hours)
    if events.empty:
        print("no upcoming games in window")
        return pd.DataFrame()
    print(f"{len(events)} upcoming games; markets firing: "
          f"{', '.join(sorted(firing))}")

    quotes = fetch_live_odds(events, client, list(firing))
    if quotes.empty:
        print("no live quotes returned")
        return pd.DataFrame()
    quotes = quotes[quotes["lead_min"] >= min_lead_min]
    if quotes.empty:
        print("all games are inside the minimum lead time")
        return pd.DataFrame()

    cons = attach_consensus(quotes, min_books=2,
                            self_anchor_markets=C.SELF_ANCHOR_MARKETS)

    # Resolve players and attach form.
    lm = LiveMatcher(players, games)
    is_prop = cons["subject"].ne("game")
    cons["athlete_id"] = None
    cons.loc[is_prop, "athlete_id"] = [lm.match(s) for s in
                                       cons.loc[is_prop, "subject"]]

    as_of = dt.date.today().isoformat()
    bat, pit = F.current_form(players, games, as_of)
    tm_latest = _latest_team_form(players, games)
    park = _latest_park(games)

    out = []
    for mkt, b in firing.items():
        sub = cons[(cons["market"] == mkt) &
                   cons["book"].isin(C.BETTABLE_BOOKS) &
                   ~cons["book"].isin(C.DFS_BOOKS)].copy()
        sub = sub.dropna(subset=["p_cons"])
        sub = sub[sub["n_books_cons"] >= b.min_books]
        if sub.empty:
            continue
        sub = _attach_live_features(sub, bat, pit, tm_latest, park,
                                    events, lm, cons)
        sub["logit_cons"] = logit(sub["p_cons"].to_numpy(dtype=float))
        for f in b.features:
            if f not in sub.columns:
                sub[f] = np.nan
        p_over = b.model.predict_over(sub)
        sub["p_model"] = side_probability(p_over, sub["side"])
        price = sub["price"].to_numpy(dtype=float)
        sub["p_raw"] = american_to_prob(price)
        sub["ev"] = ev_per_unit(sub["p_model"].to_numpy(), price)
        sub["kelly"] = np.minimum(
            kelly_fraction(sub["p_model"].to_numpy(), price), kelly_cap)
        sub["threshold"] = b.effective_threshold
        sub["deployment"] = b.deployment
        sel = sub[sub["ev"] >= b.effective_threshold].copy()
        if sel.empty:
            continue
        # One pick per proposition+side: the best price only.
        sel = (sel.sort_values("ev", ascending=False)
                  .drop_duplicates(subset=["event_id", "market", "subject",
                                           "line", "side"], keep="first"))
        out.append(sel)

    if not out:
        print("no picks cleared their thresholds")
        return pd.DataFrame()

    picks = pd.concat(out, ignore_index=True)
    picks["confidence"] = confidence(picks)
    picks["generated_at"] = dt.datetime.now(UTC).isoformat(timespec="seconds")
    picks["game_date"] = picks["event_id"].map(
        dict(zip(events["event_id"], events["game_date"])))
    cols = ["generated_at", "game_date", "commence_time", "event_id", "market",
            "subject", "side", "line", "price", "book", "p_model", "p_cons",
            "p_raw", "ev", "kelly", "confidence", "n_books_cons", "threshold",
            "deployment", "lead_min"]
    picks = picks[[c for c in cols if c in picks.columns]]
    return picks.sort_values(["ev"], ascending=False).reset_index(drop=True)


def confidence(picks: pd.DataFrame) -> np.ndarray:
    """A 0-100 score combining edge size with how well-supported the price is.

    Edge alone is a bad confidence signal: the largest edges tend to appear on
    the thinnest propositions, where the consensus is one book's opinion and
    the "edge" is really disagreement about a number nobody has priced
    carefully.
    """
    ev = picks["ev"].to_numpy(dtype=float)
    nb = picks["n_books_cons"].to_numpy(dtype=float)
    lead = picks["lead_min"].to_numpy(dtype=float)
    edge_score = np.clip(ev / 0.12, 0, 1)
    book_score = np.clip((nb - 2) / 8.0, 0, 1)
    # Prices very far from first pitch are less reliable than mature ones.
    time_score = np.clip(1.2 - np.abs(lead - 60) / 240.0, 0.3, 1.0)
    raw = 0.55 * edge_score + 0.32 * book_score + 0.13 * time_score
    return np.round(100 * np.clip(raw, 0, 1), 1)


# ---------------------------------------------------------------------------

def _latest_team_form(players: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    tf = F.team_form(players, games)
    return (tf.sort_values("game_date").groupby("team", as_index=False)
              .tail(1).drop(columns=["event_id", "game_date"]))


def _latest_park(games: pd.DataFrame) -> pd.DataFrame:
    pf = F.park_form(games)
    return (pf.sort_values("game_date").groupby("venue", as_index=False)
              .tail(1).drop(columns=["espn_id", "game_date"]))


def _attach_live_features(sub: pd.DataFrame, bat: pd.DataFrame,
                          pit: pd.DataFrame, tm: pd.DataFrame,
                          park: pd.DataFrame, events: pd.DataFrame,
                          lm: LiveMatcher, all_quotes: pd.DataFrame
                          ) -> pd.DataFrame:
    """Build exactly the feature set the model was fitted on.

    `all_quotes` is the full consensus frame, not the bettable subset: the
    market-structure features describe how many books priced the proposition
    and how far apart they were, which must be measured over every book that
    quoted it, the same way build_props measures it.
    """
    from .dataset import PROP_KEY, add_projection, proposition_structure

    struct = proposition_structure(all_quotes)
    keep = [c for c in ("n_books_prop", "hold_med", "n_quotes", "book_std",
                        "book_spread") if c in struct.columns]
    sub = sub.drop(columns=[c for c in keep if c in sub.columns], errors="ignore")
    sub = sub.merge(struct[PROP_KEY + keep], on=PROP_KEY, how="left")

    bat_cols = [c for c in bat.columns if c.startswith("bat_")]
    pit_cols = [c for c in pit.columns if c.startswith("pit_")]
    sub = sub.merge(bat[["athlete_id"] + bat_cols], on="athlete_id", how="left")
    sub = sub.merge(pit[["athlete_id"] + pit_cols], on="athlete_id", how="left")

    sub["player_team"] = sub["athlete_id"].map(lm.team)
    tm_cols = [c for c in tm.columns if c.startswith("tm_")]
    sub = sub.merge(tm.rename(columns={"team": "player_team"})
                    [["player_team"] + tm_cols], on="player_team", how="left")

    home = dict(zip(events["event_id"], events["home_team"]))
    sub["home_team"] = sub["event_id"].map(home)
    sub["is_home_player"] = (sub["player_team"].notna() &
                             (sub["player_team"] == sub["home_team"])
                             ).astype(float)
    sub = _attach_live_opponent(sub, pit, all_quotes, lm)
    return add_projection(sub)


def _attach_live_opponent(sub: pd.DataFrame, pit: pd.DataFrame,
                          all_quotes: pd.DataFrame, lm: LiveMatcher
                          ) -> pd.DataFrame:
    """Opposing starter's form, resolved from tonight's own market.

    Books quote pitcher props only for announced starters, so the presence of
    those props in the live snapshot identifies the two starters -- the same
    point-in-time source the backtest uses.
    """
    pit_cols = [c for c in pit.columns if c.startswith("pit_")]
    sub["has_opp_starter"] = 0.0
    for c in pit_cols:
        sub[f"opp_{c}"] = np.nan
    if not pit_cols:
        return sub

    starters = (all_quotes[all_quotes["market"]
                           .isin(["pitcher_strikeouts", "pitcher_outs"])]
                .dropna(subset=["athlete_id"])[["event_id", "athlete_id"]]
                .drop_duplicates())
    if starters.empty:
        return sub
    starters["pitcher_team"] = starters["athlete_id"].map(lm.team)
    starters = starters.dropna(subset=["pitcher_team"])
    if starters.empty:
        return sub

    rows = sub[["event_id", "player_team"]].reset_index()
    pair = rows.merge(starters, on="event_id", how="left")
    pair = pair[pair["pitcher_team"].notna() & pair["player_team"].notna()
                & (pair["pitcher_team"] != pair["player_team"])]
    pair = pair.drop_duplicates(subset=["index"], keep="first")
    if pair.empty:
        return sub

    opp = pit[["athlete_id"] + pit_cols].rename(
        columns={"athlete_id": "opp_athlete_id",
                 **{c: f"opp_{c}" for c in pit_cols}})
    pair = pair.rename(columns={"athlete_id": "opp_athlete_id"})
    pair["has_opp_starter"] = 1.0
    pair = pair.merge(opp, on="opp_athlete_id", how="left")

    keep = ["index", "has_opp_starter"] + [f"opp_{c}" for c in pit_cols]
    sub = sub.drop(columns=[c for c in keep if c != "index" and c in sub.columns])
    sub = sub.merge(pair[keep].set_index("index"), left_index=True,
                    right_index=True, how="left")
    sub["has_opp_starter"] = sub["has_opp_starter"].fillna(0.0)
    return sub


def save_picks(picks: pd.DataFrame, path=PICKS_PATH) -> None:
    """Append idempotently: re-running a day replaces that day's picks."""
    if picks.empty:
        return
    key = ["game_date", "market", "subject", "line", "side", "book"]
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["game_date"].isin(picks["game_date"].unique())]
        allp = pd.concat([old, picks], ignore_index=True)
    else:
        allp = picks
    allp = allp.drop_duplicates(subset=key, keep="last")
    allp.to_parquet(path, index=False)
    print(f"saved {len(picks)} picks ({len(allp)} total) -> {path}")
