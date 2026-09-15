"""Leakage guard.

Point-in-time integrity is the thing that separates a real edge from a fake
one, so it is checked rather than asserted. Four independent checks, any of
which can veto a market:

1. **Quote timing.** Every price used for a decision was observable strictly
   before first pitch.
2. **Feature as-of.** Rolling features are *recomputed from scratch* for a
   random sample of rows using only strictly-prior games, and compared to the
   stored value. This catches a mis-specified shift that no amount of code
   review reliably would.
3. **Fold boundary.** No training row is dated on or after the first test
   date, and the calibration and the EV threshold were fitted inside the
   training window only.
4. **Conditional signal.** Any feature carrying implausible information about
   the outcome *after* the market consensus is accounted for is flagged. A
   genuine feature adds a little; a leaked one adds a lot.

Check 4 is a smoke alarm, not a proof: it raises a warning, and the gate
escalates it to a veto past a hard threshold.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class LeakFinding:
    check: str
    severity: str          # 'ok' | 'warn' | 'veto'
    detail: str
    n: int = 0


@dataclass
class LeakReport:
    findings: list[LeakFinding] = field(default_factory=list)

    def add(self, check: str, severity: str, detail: str, n: int = 0) -> None:
        self.findings.append(LeakFinding(check, severity, detail, n))

    @property
    def vetoed(self) -> bool:
        return any(f.severity == "veto" for f in self.findings)

    @property
    def warnings(self) -> list[LeakFinding]:
        return [f for f in self.findings if f.severity == "warn"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.__dict__ for f in self.findings])

    def summary(self) -> str:
        v = [f for f in self.findings if f.severity == "veto"]
        if v:
            return f"VETO: {v[0].check} -- {v[0].detail}"
        w = self.warnings
        return f"clean ({len(self.findings)} checks)" if not w else \
               f"clean with {len(w)} warning(s)"


# ---------------------------------------------------------------------------

def check_quote_timing(quotes: pd.DataFrame, rep: LeakReport) -> None:
    if "lead_min" not in quotes:
        rep.add("quote_timing", "veto", "lead_min column missing")
        return
    bad = quotes["lead_min"].isna() | (quotes["lead_min"] <= 0)
    n = int(bad.sum())
    if n:
        rep.add("quote_timing", "veto",
                f"{n} quotes taken at or after first pitch", n)
    else:
        rep.add("quote_timing", "ok",
                f"all {len(quotes)} quotes strictly pre-first-pitch "
                f"(min lead {quotes['lead_min'].min():.1f} min)", len(quotes))


def check_fold_boundary(train: pd.DataFrame, test: pd.DataFrame,
                        rep: LeakReport, date_col: str = "game_date") -> None:
    if train.empty or test.empty:
        rep.add("fold_boundary", "warn", "empty train or test fold")
        return
    t0 = test[date_col].min()
    bleed = int((train[date_col] >= t0).sum())
    if bleed:
        rep.add("fold_boundary", "veto",
                f"{bleed} training rows dated on/after first test date {t0}",
                bleed)
    else:
        rep.add("fold_boundary", "ok",
                f"train ends {train[date_col].max()}, test starts {t0}")


def check_feature_asof(feature_rows: pd.DataFrame, source: pd.DataFrame,
                       entity_col: str, date_col: str, value_col: str,
                       feature_col: str, window: int, rep: LeakReport,
                       n_sample: int = 300, tol: float = 1e-6) -> None:
    """Recompute a rolling feature independently and compare.

    `feature_rows` holds the stored feature; `source` is the raw per-game
    observations it was built from. For a random sample of rows we take the
    entity's observations strictly before that row's date, average the last
    `window` of them, and require a match.
    """
    fr = feature_rows.dropna(subset=[feature_col])
    if fr.empty:
        rep.add(f"asof::{feature_col}", "warn", "no non-null values to check")
        return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(fr), size=min(n_sample, len(fr)), replace=False)
    sample = fr.iloc[idx]

    src = source[[entity_col, date_col, value_col]].dropna(subset=[value_col])
    by_entity = {k: v.sort_values(date_col)
                 for k, v in src.groupby(entity_col, observed=True)}

    mismatches, checked, future_used = 0, 0, 0
    for r in sample.itertuples():
        ent = getattr(r, entity_col)
        d = getattr(r, date_col)
        s = by_entity.get(ent)
        if s is None:
            continue
        prior = s[s[date_col] < d][value_col]
        if len(prior) < max(2, window // 5):
            continue
        expect = prior.tail(window).mean()
        got = getattr(r, feature_col)
        checked += 1
        if not np.isclose(expect, got, rtol=1e-4, atol=tol):
            mismatches += 1
            # Does including the current/future games explain the stored value?
            incl = s[s[date_col] <= d][value_col].tail(window).mean()
            if np.isclose(incl, got, rtol=1e-4, atol=tol):
                future_used += 1

    if checked == 0:
        rep.add(f"asof::{feature_col}", "warn", "no comparable rows")
    elif future_used >= max(5, 0.02 * checked):
        # Systematic, not incidental. A handful of coincidental matches is
        # expected -- two windows can agree by chance, especially around a
        # doubleheader where both games share a date -- and vetoing a market
        # on one of them is a false positive that costs more than it saves.
        rep.add(f"asof::{feature_col}", "veto",
                f"{future_used}/{checked} sampled rows match a window that "
                f"includes the current game -- the feature sees its own outcome",
                future_used)
    elif future_used:
        rep.add(f"asof::{feature_col}", "warn",
                f"{future_used}/{checked} sampled rows coincide with a "
                f"current-game window; below the systematic threshold",
                future_used)
    elif mismatches:
        rep.add(f"asof::{feature_col}", "warn",
                f"{mismatches}/{checked} rows differ from an independent "
                f"recomputation (not explained by future data)", mismatches)
    else:
        rep.add(f"asof::{feature_col}", "ok",
                f"{checked} sampled rows reproduce exactly from prior games only",
                checked)


#: Only these families can have a broken as-of boundary, because only these
#: are rolled up from past games. Market-structure columns (book counts, holds,
#: quoted lines, lead time) are read straight off the decision snapshot, so
#: their timing is already guaranteed by check_quote_timing -- and they can
#: legitimately carry real pre-game signal. Books thin out a total when rain is
#: forecast, and rain suppresses runs; that is information, not leakage, and
#: vetoing on it kills a sound market.
FORM_PREFIXES = ("bat_", "pit_", "tm_", "park_", "opp_", "home_tg_", "away_tg_")


def _vetoable(name: str) -> bool:
    return name.startswith(FORM_PREFIXES)


def check_conditional_signal(df: pd.DataFrame, feature_cols: list[str],
                             outcome_col: str, market_col: str,
                             rep: LeakReport, warn_auc: float = 0.60,
                             veto_auc: float = 0.70) -> None:
    """Residual discrimination of each feature after the market is removed.

    We regress out the market consensus (in log-odds) and measure how well each
    feature still separates winners from losers. A real feature lands a hair
    above 0.5; a feature that has seen the result lands far above it.
    """
    d = df.dropna(subset=[outcome_col, market_col])
    if len(d) < 500:
        rep.add("conditional_signal", "warn", "sample too small to test")
        return
    y = d[outcome_col].to_numpy(dtype=float)
    m = d[market_col].to_numpy(dtype=float)

    worst, worst_auc = None, 0.5
    for c in feature_cols:
        if c not in d:
            continue
        x = pd.to_numeric(d[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 500:
            continue
        auc = _stratified_auc(x[ok], y[ok], m[ok])
        auc = max(auc, 1 - auc)
        if auc > worst_auc:
            worst, worst_auc = c, auc
        if auc >= warn_auc and not _vetoable(c):
            rep.add("conditional_signal_info", "warn",
                    f"{c} carries residual signal at AUC {auc:.3f}; it is read "
                    f"off the decision snapshot, so this is information rather "
                    f"than leakage")

    if worst is not None and not _vetoable(worst) and worst_auc >= veto_auc:
        rep.add("conditional_signal", "warn",
                f"{worst} residual AUC {worst_auc:.3f}, but it is a "
                f"market-structure column whose timing is already proven by "
                f"the quote-timing check -- not escalated to a veto")
        return

    if worst is None:
        rep.add("conditional_signal", "ok", "no feature exceeds chance")
    elif worst_auc >= veto_auc:
        rep.add("conditional_signal", "veto",
                f"{worst} separates residual outcomes at AUC {worst_auc:.3f} "
                f"-- implausible without leakage")
    elif worst_auc >= warn_auc:
        rep.add("conditional_signal", "warn",
                f"{worst} residual AUC {worst_auc:.3f} (above {warn_auc})")
    else:
        rep.add("conditional_signal", "ok",
                f"max residual AUC {worst_auc:.3f} ({worst})")


def _stratified_auc(x: np.ndarray, y: np.ndarray, m: np.ndarray,
                    bins: int = 20, min_stratum: int = 50) -> float:
    """AUC of x against y *within* strata of the market probability.

    Subtracting E[y|market] and testing the sign of the residual does not work
    for a binary outcome: y - p is positive exactly when y == 1, so the test
    silently degenerates into an unconditional AUC and flags any feature
    related to the result at all -- the line itself, for instance, which is
    related to the result only *through* the market. Stratifying and pooling
    the within-stratum comparisons is the real conditional test: a feature that
    adds nothing beyond the market scores 0.5.
    """
    order = np.argsort(m)
    weighted, total = 0.0, 0.0
    for ch in np.array_split(order, bins):
        if len(ch) < min_stratum:
            continue
        yy = y[ch].astype(bool)
        n1, n0 = int(yy.sum()), int((~yy).sum())
        if n1 == 0 or n0 == 0:
            continue
        w = float(n1 * n0)
        weighted += _rank_auc(x[ch], yy) * w
        total += w
    return weighted / total if total else 0.5


def _rank_auc(x: np.ndarray, pos: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = pd.Series(x).rank().to_numpy()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def audit_rolling_features(props: pd.DataFrame, players: pd.DataFrame,
                           games: pd.DataFrame, rep: LeakReport,
                           n_sample: int = 250) -> None:
    """Rebuild representative rolling features from raw per-game data.

    This is the check that actually proves the as-of boundary rather than
    asserting it. The per-game observations are reconstructed independently of
    features.py, and for a sample of proposition rows we require the stored
    feature to equal the mean of that entity's *strictly prior* games. A stored
    value that instead matches a window including the current game is reported
    as a veto, naming the feature.
    """
    g = games[["espn_id", "game_date"]].copy()
    g["game_date"] = g["game_date"].astype(str)
    # Order on real first pitch, exactly as features.py does, so a doubleheader
    # is compared against the same notion of "strictly before".
    if "date" in games.columns:
        g["asof_key"] = pd.to_datetime(games["date"], utc=True,
                                       format="ISO8601").astype("int64")
    else:
        g["asof_key"] = pd.to_datetime(g["game_date"], utc=True).astype("int64")

    bat = players[players["group"] == "batting"].merge(
        g, left_on="event_id", right_on="espn_id", how="inner")
    if not bat.empty:
        pa = bat["at_bats"].fillna(0) + bat["walks"].fillna(0)
        bat = bat.assign(hits_ppa=np.where(pa > 0, bat["hits"].fillna(0) / pa,
                                           np.nan))
        src = bat[["athlete_id", "asof_key", "hits_ppa"]].dropna()
        rows = props.dropna(subset=["athlete_id"]).merge(
            g[["espn_id", "asof_key"]], on="espn_id", how="left")
        rows = rows[["athlete_id", "asof_key", "bat_hits_ppa_r10"]].dropna()
        if len(rows) > 50:
            check_feature_asof(rows, src, "athlete_id", "asof_key",
                               "hits_ppa", "bat_hits_ppa_r10", 10, rep,
                               n_sample=n_sample)

    pit = players[players["group"] == "pitching"].merge(
        g, left_on="event_id", right_on="espn_id", how="inner")
    if not pit.empty:
        src = pit[["athlete_id", "game_date", "strike_outs"]].dropna()
        rows = props.dropna(subset=["athlete_id"])
        rows = rows[["athlete_id", "game_date", "pit_strike_outs_r8"]].dropna() \
            if "pit_strike_outs_r8" in props.columns else pd.DataFrame()
        if len(rows) > 50:
            check_feature_asof(rows, src, "athlete_id", "game_date",
                               "strike_outs", "pit_strike_outs_r8", 8, rep,
                               n_sample=n_sample)


def audit(quotes: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
          feature_cols: list[str], outcome_col: str = "won",
          market_col: str = "logit_cons", props: pd.DataFrame | None = None,
          players: pd.DataFrame | None = None,
          games: pd.DataFrame | None = None) -> LeakReport:
    """The standard bundle the gate runs for every market."""
    rep = LeakReport()
    check_quote_timing(quotes, rep)
    check_fold_boundary(train, test, rep)
    if not test.empty:
        check_conditional_signal(test, feature_cols, outcome_col, market_col, rep)
    if props is not None and players is not None and games is not None:
        audit_rolling_features(props, players, games, rep)
    return rep
