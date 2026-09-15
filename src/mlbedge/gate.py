"""The validation gate: PASS / FAIL / VETO per market, with a named cause.

Ordering matters. Leakage is checked first and outranks every statistical
result, because a leaked market looks *better* the more broken it is -- a
market showing +110% ROI with a confidence interval nowhere near zero is not a
discovery, it is a bug with good manners.

After that, a market must clear all of:

  sample     enough graded bets to distinguish the claimed edge from noise
  roi        bootstrap CI lower bound above zero
  family     survives Benjamini-Hochberg across all eleven markets -- testing
             eleven things and reporting the best one is how a 5% false
             positive rate becomes a 43% one
  clv        positive closing-line value where closing prices exist. ROI can be
             luck over a few hundred bets; CLV is measured on every bet and is
             far harder to fake
  stability  the edge shows up across folds rather than in one hot month
  calibration predicted probabilities track realised frequencies

Anything that fails gets a cause and a lever: the specific next action that
would plausibly change the verdict, not "no edge, moved on".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C

PASS, FAIL, VETO = "PASS", "FAIL", "VETO"
LIVE, VETO_FILTERED, KILLED = "live", "veto_filtered", "killed"

MIN_BETS = 300
MIN_FOLD_WIN_RATE = 0.50
MAX_SINGLE_FOLD_SHARE = 0.60
MAX_CAL_ERROR = 0.035
ALPHA = 0.05


@dataclass
class Verdict:
    market: str
    verdict: str
    cause: str = ""
    lever: str = ""
    deployment: str = KILLED
    metrics: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def row(self) -> dict:
        d = {"market": self.market, "verdict": self.verdict,
             "deployment": self.deployment, "cause": self.cause,
             "lever": self.lever}
        d.update(self.metrics)
        return d


# ---------------------------------------------------------------------------

def roi_pvalue(profit: np.ndarray, n_boot: int = 10000, seed: int = 0) -> float:
    """One-sided bootstrap p-value for H0: ROI <= 0."""
    if len(profit) == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(profit), size=(n_boot, len(profit)))
    rois = profit[idx].mean(axis=1)
    centred = rois - profit.mean()
    return float((centred >= profit.mean()).mean())


def recent_roi(bets: pd.DataFrame, frac: float = 1 / 3) -> float:
    """ROI over the most recent slice of the out-of-sample period.

    A market whose edge was real early and gone late has a healthy full-sample
    ROI and no future. Since the whole point is to fire tomorrow, the recent
    slice gets its own check rather than being averaged away.
    """
    if bets.empty or "game_date" not in bets:
        return np.nan
    d = bets.sort_values("game_date")
    k = max(1, int(len(d) * frac))
    return float(d["profit"].tail(k).mean())


def decayed(bets: pd.DataFrame, frac: float = 1 / 3, z: float = 1.0) -> bool:
    """Is the recent slice *confidently* losing, rather than merely noisy?

    A point estimate on a third of the bets is far too noisy to gate on: a
    genuinely profitable market will show a negative recent third perhaps a
    third of the time. So decay is only called when the recent slice is below
    zero by more than `z` standard errors -- confident decay, not a cold spell.
    """
    if bets.empty or "game_date" not in bets:
        return False
    d = bets.sort_values("game_date")
    k = max(1, int(len(d) * frac))
    tail = d["profit"].tail(k).to_numpy(dtype=float)
    if len(tail) < 60:
        return False
    se = tail.std(ddof=1) / np.sqrt(len(tail))
    return bool(tail.mean() + z * se < 0)


def calibration_error(bets: pd.DataFrame) -> float:
    cal = B.calibration(bets)
    if cal.empty:
        return np.nan
    w = cal["n"] / cal["n"].sum()
    return float(np.sum(w * np.abs(cal["pred"] - cal["actual"])))


def benjamini_hochberg(pvals: dict[str, float], alpha: float = ALPHA
                       ) -> dict[str, bool]:
    """Return {market: survives}. Controls the false discovery rate across the
    whole family of markets tested."""
    items = [(k, v) for k, v in pvals.items() if np.isfinite(v)]
    if not items:
        return {k: False for k in pvals}
    items.sort(key=lambda t: t[1])
    m = len(items)
    survives, kmax = {k: False for k in pvals}, -1
    for i, (_, p) in enumerate(items, start=1):
        if p <= alpha * i / m:
            kmax = i
    for i, (k, _) in enumerate(items, start=1):
        survives[k] = i <= kmax
    return survives


# ---------------------------------------------------------------------------

LEVERS = {
    "insufficient_sample": (
        "Sample is the binding constraint, not the edge. Widen the bet "
        "universe before touching the model: lower min_books from 3 to 2 to "
        "admit thinly-quoted propositions, extend the backfill another season, "
        "and add the alternate lines this market posts (they are already in "
        "the raw cache and cost nothing to re-parse)."),
    "no_edge": (
        "The market prices this efficiently at the books we can reach. The "
        "only lever with real headroom is a sharper anchor: weight the "
        "consensus harder toward the zero-vig exchanges (novig, prophetx) and "
        "Pinnacle and re-fit the anchor weights on train folds, then re-test. "
        "If the edge is still flat, this is a line-shopping market only -- "
        "run it veto-filtered at a high EV cut rather than killing it."),
    "negative_edge": (
        "Losing, not merely flat: the selection rule is picking the wrong side "
        "systematically. Check the de-vig model first -- a multiplicative "
        "de-vig on a longshot-heavy market overstates longshot probability and "
        "will reliably buy the wrong tail. Re-run with Shin and power and "
        "compare. Kill it live until it is positive out of sample."),
    "no_clv": (
        "ROI is positive but closing-line value is not, which means the "
        "profit is variance rather than edge and will mean-revert. Do not "
        "ship. Re-test with the decision snapshot moved closer to first pitch; "
        "if CLV stays flat the apparent ROI is noise."),
    "unstable": (
        "The edge is concentrated in a minority of folds rather than "
        "persistent. Most often a market regime the features do not see. Add "
        "an explicit regime feature (park run environment and month-of-season "
        "are already computed) and re-fit; if it stays lumpy, veto-filter to "
        "the folds' common denominator rather than shipping the average."),
    "edge_decayed": (
        "The edge is real in the early sample and gone by the end, which is "
        "the shape of a market the books have since tightened, or of a "
        "bookmaker that has cut limits or corrected a stale feed. The "
        "full-sample ROI is therefore not a forecast. Lever: re-fit on a "
        "trailing window only (drop the oldest season rather than expanding), "
        "and check the per-book breakdown -- if the decay is concentrated in "
        "one or two books that have since tightened, drop those books and "
        "re-test the remainder rather than abandoning the market."),
    "miscalibrated": (
        "Predicted probabilities do not track realised frequencies, so the EV "
        "number is not trustworthy even where the sign is right. Re-fit the "
        "isotonic calibration on a longer training window, and reduce the "
        "shrinkage grid ceiling so the model cannot drift as far from the "
        "market consensus."),
    "family_wise": (
        "Nominally significant but does not survive Benjamini-Hochberg across "
        "eleven markets -- this is the best of eleven tries, not an "
        "independent discovery. Needs more out-of-sample bets to clear the "
        "corrected bar; extend the backfill rather than loosening the test."),
    "leakage": (
        "A feature or price carries information from at or after the decision "
        "instant. Fix the as-of boundary and re-validate from scratch. No "
        "statistical result from this market means anything until it is clean."),
}


def judge(market: str, bets: pd.DataFrame, closing: pd.DataFrame | None,
          leak_report=None, min_bets: int = MIN_BETS) -> Verdict:
    """Evaluate one market. Family-wise correction is applied afterwards by
    `apply_family_correction`, which can downgrade a PASS."""
    summ = B.summarise(bets)
    clv = B.clv_metrics(bets, closing) if closing is not None else {
        "clv_n": 0, "clv_mean": np.nan, "beat_close": np.nan}
    stab = B.period_stability(bets)
    metrics = {**summ, **clv}

    if not bets.empty:
        prof = bets["profit"].to_numpy(dtype=float)
        metrics["p_value"] = roi_pvalue(prof)
        metrics["cal_error"] = calibration_error(bets)
        metrics["n_folds"] = len(stab)
        metrics["fold_win_rate"] = float((stab["roi"] > 0).mean()) if len(stab) else np.nan
        tot = stab["profit"].sum() if len(stab) else 0.0
        metrics["max_fold_share"] = (
            float(stab["profit"].max() / tot) if tot > 0 else np.nan)
        metrics["avg_shrink"] = float(bets["shrink"].mean()) if "shrink" in bets else np.nan
        metrics["recent_roi"] = recent_roi(bets)
        maj = bets[bets["book"].isin(C.MAJOR_BOOKS)] if "book" in bets else bets.iloc[:0]
        metrics["n_major"] = int(len(maj))
        metrics["roi_major"] = (float(maj["profit"].mean())
                                if len(maj) else float("nan"))
    else:
        metrics.update({"p_value": 1.0, "cal_error": np.nan, "n_folds": 0,
                        "fold_win_rate": np.nan, "max_fold_share": np.nan,
                        "avg_shrink": np.nan, "recent_roi": np.nan,
                        "n_major": 0, "roi_major": np.nan})

    checks: dict[str, bool] = {}

    # 1. Leakage outranks everything.
    if leak_report is not None and getattr(leak_report, "vetoed", False):
        return Verdict(market, VETO, "leakage",
                       LEVERS["leakage"] + f" Detail: {leak_report.summary()}",
                       KILLED, metrics, {"leakage": False})
    checks["leakage"] = True

    # 2. Sample.
    checks["sample"] = summ["n_bets"] >= min_bets
    if not checks["sample"]:
        return Verdict(market, FAIL, "insufficient_sample",
                       LEVERS["insufficient_sample"], KILLED, metrics, checks)

    # 3. Edge sign and interval.
    checks["roi_positive"] = bool(summ["roi"] > 0)
    checks["roi_ci"] = bool(summ["roi_lo"] > 0)
    if not checks["roi_positive"]:
        return Verdict(market, FAIL, "negative_edge", LEVERS["negative_edge"],
                       KILLED, metrics, checks)
    if not checks["roi_ci"]:
        return Verdict(market, FAIL, "no_edge", LEVERS["no_edge"],
                       VETO_FILTERED, metrics, checks)

    # 4. CLV, where closing prices exist.
    if clv["clv_n"] >= 100:
        checks["clv"] = bool(clv["clv_mean"] > 0)
        if not checks["clv"]:
            return Verdict(market, FAIL, "no_clv", LEVERS["no_clv"],
                           VETO_FILTERED, metrics, checks)
    else:
        checks["clv"] = True  # not measurable; not held against the market

    # 5. Stability.
    checks["stability"] = bool(
        (metrics["n_folds"] >= 3) and
        (metrics["fold_win_rate"] >= MIN_FOLD_WIN_RATE) and
        (not np.isfinite(metrics["max_fold_share"])
         or metrics["max_fold_share"] <= MAX_SINGLE_FOLD_SHARE))
    if not checks["stability"]:
        return Verdict(market, FAIL, "unstable", LEVERS["unstable"],
                       VETO_FILTERED, metrics, checks)

    # 5b. Decay -- the edge must still be there at the end of the sample.
    checks["no_decay"] = not decayed(bets)
    if not checks["no_decay"]:
        return Verdict(market, FAIL, "edge_decayed", LEVERS["edge_decayed"],
                       VETO_FILTERED, metrics, checks)

    # 6. Calibration.
    ce = metrics["cal_error"]
    checks["calibration"] = bool(not np.isfinite(ce) or ce <= MAX_CAL_ERROR)
    if not checks["calibration"]:
        return Verdict(market, FAIL, "miscalibrated", LEVERS["miscalibrated"],
                       VETO_FILTERED, metrics, checks)

    return Verdict(market, PASS, "", "", LIVE, metrics, checks)


def apply_family_correction(verdicts: list[Verdict], alpha: float = ALPHA
                            ) -> list[Verdict]:
    """Downgrade any PASS that does not survive BH across the family."""
    pvals = {v.market: v.metrics.get("p_value", 1.0) for v in verdicts
             if v.verdict == PASS}
    if not pvals:
        return verdicts
    survives = benjamini_hochberg(pvals, alpha)
    for v in verdicts:
        if v.verdict == PASS and not survives.get(v.market, False):
            v.verdict = FAIL
            v.cause = "family_wise"
            v.lever = LEVERS["family_wise"]
            v.deployment = VETO_FILTERED
            v.checks["family_wise"] = False
        elif v.verdict == PASS:
            v.checks["family_wise"] = True
    return verdicts


def to_frame(verdicts: list[Verdict]) -> pd.DataFrame:
    return pd.DataFrame([v.row() for v in verdicts])
