"""Price the whole line ladder from the line the market actually prices well.

Books do not quote every line with equal care. For batter hits, a dozen books
price Over 0.5 and argue about it all afternoon; Over 2.5 is posted by two of
them and then left alone. The well-attended line is therefore a much better
estimate of the player's distribution than the sparse one -- and the sparse one
is where the mispricing lives.

This module learns, per market, the empirical relationship

    P(Y > target_line)  given  the consensus at the primary line

from training data only, and uses it to put a probability on lines that too few
books quote to build a consensus from directly. The relationship is a lookup
table rather than a parametric count distribution: a Negative Binomial fitted
to pitcher outs is simply wrong (outs arrive three at a time and pile up at 15,
18 and 21), and a family that misfits the tails invents edge exactly where the
alternate lines sit.

The table is estimated on the earliest season only. That season lies entirely
inside every walk-forward fold's training window, so nothing here is fitted on
data a fold is scored against.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .devig import expit, logit

TABLE_PATH = C.DATA / "ladder_table.json"

N_BUCKETS = 20          # strength buckets on the primary-line consensus
MIN_CELL = 60           # observations before a cell is trusted
GROUP_KEY = ["event_id", "market", "subject"]


def strength_bucket(p: np.ndarray, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Map a primary-line probability to a bucket index.

    Buckets are cut on the logit scale so the extremes -- where home-run and
    alternate-line props live -- get resolution instead of being crushed into
    one bin.
    """
    l = np.clip(logit(np.asarray(p, dtype=float)), -5.0, 5.0)
    idx = np.floor((l + 5.0) / 10.0 * n_buckets).astype("float")
    return np.clip(idx, 0, n_buckets - 1)


def primary_lines(props: pd.DataFrame) -> pd.DataFrame:
    """For each (event, market, subject), the line the books price best.

    'Best' is the most weighted books quoting it two-sided, tie-broken by total
    quote count -- i.e. the line the market has actually thought about.
    """
    d = props.dropna(subset=["p_cons_all"]).copy()
    if d.empty:
        return pd.DataFrame(columns=GROUP_KEY + ["primary_line", "p_primary"])
    d["_n"] = d["n_books_prop"].fillna(0)
    d["_q"] = d.get("n_quotes", pd.Series(0, index=d.index)).fillna(0)
    best = (d.sort_values(["_n", "_q"], ascending=False)
              .drop_duplicates(subset=GROUP_KEY, keep="first"))
    return best[GROUP_KEY + ["line", "p_cons_all", "n_books_prop"]].rename(
        columns={"line": "primary_line", "p_cons_all": "p_primary",
                 "n_books_prop": "n_books_primary"})


def fit(props: pd.DataFrame, n_buckets: int = N_BUCKETS,
        min_cell: int = MIN_CELL) -> dict:
    """Learn P(Y > target | market, primary_line, bucket(p_primary)).

    Uses `stat_value`, the realised statistic, so the outcome at *every*
    candidate line is known for every group -- not just the lines a book
    happened to quote.
    """
    need = {"stat_value", "p_cons_all", "market", "line"}
    if not need.issubset(props.columns):
        raise ValueError(f"ladder.fit needs {sorted(need - set(props.columns))}")

    prim = primary_lines(props)
    if prim.empty:
        return {}

    # One row per group, carrying the realised statistic.
    groups = (props.dropna(subset=["stat_value"])
                   .drop_duplicates(subset=GROUP_KEY)[GROUP_KEY + ["stat_value"]]
                   .merge(prim, on=GROUP_KEY, how="inner"))
    if groups.empty:
        return {}
    groups["bucket"] = strength_bucket(groups["p_primary"].to_numpy(), n_buckets)

    table: dict = {}
    for market, g in groups.groupby("market", observed=True):
        targets = sorted(
            float(x) for x in
            pd.unique(props.loc[props["market"] == market, "line"].dropna()))
        # Cap the ladder: a line quoted a handful of times all season is noise.
        counts = (props[props["market"] == market]
                  .groupby("line", observed=True).size())
        targets = [t for t in targets if counts.get(t, 0) >= 200]
        if not targets:
            continue
        mkt_tbl: dict = {}
        for primary, gp in g.groupby("primary_line", observed=True):
            if len(gp) < min_cell * 2:
                continue
            y = gp["stat_value"].to_numpy(dtype=float)
            b = gp["bucket"].to_numpy()
            cells: dict = {}
            for t in targets:
                over = (y > t).astype(float)
                per_bucket = {}
                for bk in range(n_buckets):
                    m = b == bk
                    if m.sum() >= min_cell:
                        per_bucket[str(bk)] = float(over[m].mean())
                if per_bucket:
                    cells[f"{t:g}"] = per_bucket
            if cells:
                mkt_tbl[f"{float(primary):g}"] = cells
        if mkt_tbl:
            table[market] = mkt_tbl
    return table


def _isotonise(per_bucket: dict[str, float]) -> dict[str, float]:
    """Force the probability to be non-decreasing in strength bucket."""
    keys = sorted(per_bucket, key=int)
    vals = np.array([per_bucket[k] for k in keys], dtype=float)
    # Pool-adjacent-violators, unweighted.
    for _ in range(len(vals)):
        bad = np.where(np.diff(vals) < 0)[0]
        if not len(bad):
            break
        i = bad[0]
        vals[i:i + 2] = vals[i:i + 2].mean()
    return {k: float(v) for k, v in zip(keys, vals)}


def smooth(table: dict) -> dict:
    """Enforce monotonicity in strength, and in line, across the table."""
    out: dict = {}
    for market, by_primary in table.items():
        out[market] = {}
        for primary, cells in by_primary.items():
            fixed = {t: _isotonise(pb) for t, pb in cells.items()}
            # A higher line can never be more likely to be exceeded.
            order = sorted(fixed, key=float)
            for i in range(1, len(order)):
                lo, hi = fixed[order[i - 1]], fixed[order[i]]
                for bk in hi:
                    if bk in lo and hi[bk] > lo[bk]:
                        hi[bk] = lo[bk]
            out[market][primary] = fixed
    return out


def _bucket_centre(k: np.ndarray, n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Log-odds at the middle of bucket k."""
    return -5.0 + (np.asarray(k, dtype=float) + 0.5) * 10.0 / n_buckets


def apply(df: pd.DataFrame, table: dict, p_primary_col: str = "p_primary",
          primary_line_col: str = "primary_line",
          n_buckets: int = N_BUCKETS) -> np.ndarray:
    """Ladder-implied P(over) for each row, or NaN where the table is silent.

    Interpolated between bucket centres rather than read off as a step
    function: a bucket spans most of a log-odds unit, and taking its mean as
    the answer for every strength inside it puts a visible staircase bias on
    exactly the alternate lines this exists to price. Vectorised per cell --
    a row-by-row lookup over a million-row frame is not viable.
    """
    if not table or df.empty:
        return np.full(len(df), np.nan)

    key = pd.DataFrame({
        "market": df["market"].astype(str).to_numpy(),
        "primary": [f"{v:g}" if np.isfinite(v) else ""
                    for v in df[primary_line_col].to_numpy(dtype=float)],
        "line": [f"{v:g}" if np.isfinite(v) else ""
                 for v in df["line"].to_numpy(dtype=float)],
    })
    strength = np.clip(logit(df[p_primary_col].to_numpy(dtype=float)), -5.0, 5.0)
    out = np.full(len(df), np.nan)

    for (mkt, prim, line), idx in key.groupby(
            ["market", "primary", "line"], sort=False).indices.items():
        cells = table.get(mkt, {}).get(prim, {})
        pb = cells.get(line)
        if not pb:
            continue
        ks = sorted(pb, key=int)
        xs = _bucket_centre(np.array([int(k) for k in ks]), n_buckets)
        ys = np.array([pb[k] for k in ks], dtype=float)
        if len(xs) == 1:
            out[idx] = ys[0]
        else:
            out[idx] = np.interp(strength[idx], xs, ys)
    return np.clip(out, 1e-4, 1 - 1e-4)


def attach(props: pd.DataFrame, table: dict) -> pd.DataFrame:
    """Add primary-line context and the ladder probability to a frame."""
    prim = primary_lines(props)
    out = props.merge(prim, on=GROUP_KEY, how="left")
    out["p_ladder"] = apply(out, table)
    # How far the ladder disagrees with the direct consensus, where both exist.
    with np.errstate(invalid="ignore"):
        out["ladder_minus_cons"] = np.where(
            out["p_ladder"].notna() & out["p_cons_all"].notna(),
            logit(out["p_ladder"].to_numpy()) - logit(out["p_cons_all"].to_numpy()),
            np.nan)
    out["is_primary_line"] = (out["line"] == out["primary_line"]).astype(float)
    return out


def save(table: dict, path=TABLE_PATH) -> None:
    path.write_text(json.dumps(table))
    n = sum(len(c) for m in table.values() for c in m.values())
    print(f"saved ladder table: {len(table)} markets, {n} line cells -> {path}")


def load(path=TABLE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
