"""Train and persist the models that actually fire in production.

The gate decides *which* markets deploy; this module builds the artefacts they
deploy with. One bundle per market, holding the fitted model, the feature list,
the EV threshold, and the gate verdict that authorised it.

Deployment policy, driven by the gate:

  live           the market passed. Fires at its fitted EV threshold.
  veto_filtered  the market failed, but not because it loses money. It fires
                 only where the evidence is strongest -- a much higher EV cut
                 and a minimum book count -- so it cannot bleed while it is
                 being improved.
  killed         the market loses or is leaking. It does not fire at all.

A failed market is never left running at its original threshold. That is the
difference between a market that is being worked on and a market that is
quietly losing money.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as B
from . import config as C
from . import dataset as D
from .gate import KILLED, LIVE, VETO_FILTERED
from .model import MarketModel

MODEL_DIR = C.ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# How much harder a veto-filtered market has to work to fire.
VETO_EV_FLOOR = 0.08
VETO_MIN_BOOKS = 5


@dataclass
class Bundle:
    market: str
    model: MarketModel
    features: list[str]
    threshold: float
    deployment: str
    verdict: str
    cause: str = ""
    trained_at: str = ""
    n_train: int = 0
    metrics: dict = field(default_factory=dict)

    @property
    def effective_threshold(self) -> float:
        if self.deployment == VETO_FILTERED:
            return max(self.threshold, VETO_EV_FLOOR)
        return self.threshold

    @property
    def min_books(self) -> int:
        return VETO_MIN_BOOKS if self.deployment == VETO_FILTERED else 3

    @property
    def fires(self) -> bool:
        return self.deployment in (LIVE, VETO_FILTERED)


def train_all(props: pd.DataFrame, candidates: pd.DataFrame,
              verdicts: pd.DataFrame, min_books: int = 3) -> dict[str, Bundle]:
    """Fit a deployable model per market on the full history."""
    vmap = verdicts.set_index("market").to_dict("index")
    out: dict[str, Bundle] = {}
    for mkt in C.ALL_MARKET_NAMES:
        info = vmap.get(mkt, {})
        deployment = info.get("deployment", KILLED)
        p = props[props["market"] == mkt]
        if p.empty or len(p) < 800:
            print(f"  [{mkt}] too little data to train ({len(p)} rows)")
            continue
        feats = D.feature_columns(p, mkt)
        if "logit_cons" not in feats:
            feats.append("logit_cons")
        mm = MarketModel(mkt)
        try:
            rep = mm.fit(p, feats)
        except ValueError as e:
            print(f"  [{mkt}] fit failed: {e}")
            continue

        c = candidates[candidates["market"] == mkt]
        thr = 0.03
        if not c.empty:
            merged = c.merge(p[D.PROP_KEY + [f for f in feats if f in p.columns]],
                             on=D.PROP_KEY, how="inner", suffixes=("", "_p"))
            if not merged.empty:
                from .devig import logit
                merged["logit_cons"] = logit(merged["p_cons"].to_numpy(dtype=float))
                merged = merged[merged["n_books_cons"] >= min_books]
                scored = B._score(merged, mm)
                thr = B.choose_threshold(scored)

        out[mkt] = Bundle(
            market=mkt, model=mm, features=feats, threshold=thr,
            deployment=deployment, verdict=info.get("verdict", "FAIL"),
            cause=info.get("cause", ""),
            trained_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            n_train=rep.n_train,
            metrics={k: info.get(k) for k in
                     ("n_bets", "roi", "roi_lo", "roi_hi", "p_value",
                      "clv_mean", "hit_rate")})
        print(f"  [{mkt}] trained n={rep.n_train} shrink={rep.shrink:.2f} "
              f"thr={thr:.3f} deploy={deployment}")
    return out


def save(bundles: dict[str, Bundle], path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / "bundles.pkl")
    with path.open("wb") as fh:
        pickle.dump(bundles, fh)
    meta = {m: {"deployment": b.deployment, "verdict": b.verdict,
                "threshold": round(b.effective_threshold, 4),
                "cause": b.cause, "trained_at": b.trained_at,
                "n_train": b.n_train, "shrink": round(b.model.shrink, 3),
                "metrics": {k: (None if v is None or (isinstance(v, float)
                                                      and not np.isfinite(v))
                                else v)
                            for k, v in b.metrics.items()}}
            for m, b in bundles.items()}
    (MODEL_DIR / "manifest.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(bundles)} bundles -> {path}")
    return path


def load(path: Path | None = None) -> dict[str, Bundle]:
    path = path or (MODEL_DIR / "bundles.pkl")
    if not path.exists():
        raise FileNotFoundError(f"no trained models at {path}; "
                                f"run scripts/train_production.py first")
    with path.open("rb") as fh:
        return pickle.load(fh)
