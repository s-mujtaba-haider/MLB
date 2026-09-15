"""The Odds API v4 client.

Three properties matter more than anything else here:

1. **Resumable.** Every successful response is written to a gzipped file on
   disk keyed by its request. A re-run of a backfill costs zero credits for
   anything already fetched, so an interrupted 7,000-event pull picks up
   exactly where it stopped.
2. **Credit-aware.** The API bills per market per region, x10 on historical
   endpoints. Spend is read back out of the response headers and appended to a
   ledger, so the real cost of a backfill is measured rather than estimated.
3. **Polite under concurrency.** A bounded thread pool with backoff on 429/5xx.
"""
from __future__ import annotations

import gzip
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlencode

import requests

from . import config as C

BASE = "https://api.the-odds-api.com/v4"
_TIMEOUT = 60
_MAX_TRIES = 5


class CreditLedger:
    """Thread-safe running total of credits spent and remaining."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (C.DATA / "credit_ledger.jsonl")
        self._lock = threading.Lock()
        self.spent = 0
        self.remaining: int | None = None
        self.calls = 0

    def record(self, tag: str, last: int, remaining: int | None) -> None:
        with self._lock:
            self.spent += last
            self.calls += 1
            if remaining is not None:
                self.remaining = remaining
            rec = {"t": time.time(), "tag": tag, "cost": last,
                   "remaining": remaining}
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def summary(self) -> str:
        return (f"{self.calls} calls, {self.spent:,} credits spent this run, "
                f"{self.remaining:,} remaining" if self.remaining is not None
                else f"{self.calls} calls, {self.spent:,} credits spent")


LEDGER = CreditLedger()


@dataclass
class Response:
    data: Any
    snapshot_ts: str | None
    prev_ts: str | None
    next_ts: str | None
    from_cache: bool
    cost: int = 0


def _cache_path(key: str) -> Path:
    p = C.RAW / (key + ".json.gz")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _read_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError):
        # A torn file from an interrupted write: drop it and refetch.
        p.unlink(missing_ok=True)
        return None


def _write_cache(key: str, payload: dict) -> None:
    p = _cache_path(key)
    tmp = p.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    tmp.replace(p)


class OddsClient:
    def __init__(self, key: str | None = None, ledger: CreditLedger | None = None,
                 cache_only: bool = False):
        self.key = key or C.api_key()
        self.ledger = ledger or LEDGER
        self.cache_only = cache_only
        self._local = threading.local()

    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            self._local.s = s
        return s

    # -- core ------------------------------------------------------------
    def _get(self, path: str, params: dict, cache_key: str,
             allow_empty: bool = True) -> Response:
        cached = _read_cache(cache_key)
        if cached is not None:
            return Response(cached.get("data"), cached.get("timestamp"),
                            cached.get("previous_timestamp"),
                            cached.get("next_timestamp"), True, 0)
        if self.cache_only:
            # Normalisation and backtests run cache-only so that re-parsing can
            # never silently spend credits on a cache miss.
            return Response(None, None, None, None, True, 0)

        q = dict(params)
        q["apiKey"] = self.key
        url = f"{BASE}{path}?{urlencode(q)}"
        last_err: Exception | None = None
        for attempt in range(_MAX_TRIES):
            try:
                r = self.session.get(url, timeout=_TIMEOUT)
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(30, 2 ** attempt) + random.random())
                continue

            if r.status_code == 200:
                body = r.json()
                # Historical endpoints wrap the payload; current ones do not.
                if isinstance(body, dict) and "data" in body:
                    payload = body
                else:
                    payload = {"data": body, "timestamp": None,
                               "previous_timestamp": None, "next_timestamp": None}
                cost = int(r.headers.get("x-requests-last", 0) or 0)
                rem = r.headers.get("x-requests-remaining")
                self.ledger.record(cache_key, cost,
                                   int(float(rem)) if rem else None)
                _write_cache(cache_key, payload)
                return Response(payload.get("data"), payload.get("timestamp"),
                                payload.get("previous_timestamp"),
                                payload.get("next_timestamp"), False, cost)

            if r.status_code in (404, 422):
                # No data for this event/timestamp -- a real answer, not a
                # failure. Cache the empty so we never pay to ask again.
                if allow_empty:
                    payload = {"data": None, "timestamp": None,
                               "previous_timestamp": None, "next_timestamp": None,
                               "_http": r.status_code}
                    _write_cache(cache_key, payload)
                    return Response(None, None, None, None, False, 0)
                raise RuntimeError(f"{r.status_code} {r.text[:200]}")

            if r.status_code == 401:
                raise RuntimeError(f"401 unauthorized -- check API key: {r.text[:200]}")
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** attempt * 2) + random.random() * 2)
                last_err = RuntimeError(f"{r.status_code} {r.text[:200]}")
                continue
            raise RuntimeError(f"{r.status_code} {r.text[:300]}")
        raise RuntimeError(f"giving up on {path}: {last_err}")

    # -- historical endpoints ---------------------------------------------
    def historical_events(self, iso_ts: str, sport: str = C.SPORT) -> Response:
        return self._get(
            f"/historical/sports/{sport}/events",
            {"date": iso_ts},
            f"events/{sport}/{iso_ts.replace(':', '')}",
        )

    def historical_featured(self, iso_ts: str, region: str,
                            markets: Sequence[str], sport: str = C.SPORT) -> Response:
        mk = ",".join(markets)
        return self._get(
            f"/historical/sports/{sport}/odds",
            {"date": iso_ts, "regions": region, "markets": mk,
             "oddsFormat": "american", "dateFormat": "iso"},
            f"featured/{sport}/{region}/{iso_ts.replace(':', '')}",
        )

    def historical_event_odds(self, event_id: str, iso_ts: str, region: str,
                              markets: Sequence[str], tag: str,
                              sport: str = C.SPORT) -> Response:
        mk = ",".join(markets)
        return self._get(
            f"/historical/sports/{sport}/events/{event_id}/odds",
            {"date": iso_ts, "regions": region, "markets": mk,
             "oddsFormat": "american", "dateFormat": "iso"},
            f"props/{event_id[:2]}/{event_id}/{region}_{tag}",
        )

    # -- live endpoints (used by the daily scheduler) ----------------------
    def live_events(self, sport: str = C.SPORT) -> Response:
        return self._get(f"/sports/{sport}/events", {"dateFormat": "iso"},
                         f"live/events/{sport}/{int(time.time() // 300)}")

    def live_event_odds(self, event_id: str, region: str,
                        markets: Sequence[str], sport: str = C.SPORT) -> Response:
        bucket = int(time.time() // 300)
        return self._get(
            f"/sports/{sport}/events/{event_id}/odds",
            {"regions": region, "markets": ",".join(markets),
             "oddsFormat": "american", "dateFormat": "iso"},
            f"live/props/{event_id}/{region}_{bucket}",
        )

    def live_featured(self, region: str, markets: Sequence[str],
                      sport: str = C.SPORT) -> Response:
        bucket = int(time.time() // 300)
        return self._get(
            f"/sports/{sport}/odds",
            {"regions": region, "markets": ",".join(markets),
             "oddsFormat": "american", "dateFormat": "iso"},
            f"live/featured/{sport}/{region}_{bucket}",
        )


def run_pool(jobs: Iterable[tuple], fn: Callable, workers: int = 8,
             desc: str = "", progress_every: int = 200) -> list:
    """Run fn(*job) across a bounded thread pool, tolerating per-job failure.

    Returns the list of successful results. Failures are counted and reported
    rather than raised, because a single bad event should not abort a backfill
    that has already paid for thousands of others.
    """
    jobs = list(jobs)
    out, errs = [], []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
                if r is not None:
                    out.append(r)
            except Exception as e:  # noqa: BLE001 -- deliberate: keep going
                errs.append((futs[f], repr(e)[:200]))
            if progress_every and done % progress_every == 0:
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(jobs) - done) / rate if rate else 0
                print(f"  [{desc}] {done}/{len(jobs)} "
                      f"({rate:.1f}/s, eta {eta/60:.1f}m) "
                      f"{LEDGER.summary()}", flush=True)
    if errs:
        print(f"  [{desc}] {len(errs)} failures; first: {errs[0]}", flush=True)
    return out
