# MLB Edge — Phase 1

Eleven MLB markets, ingested from real odds and real results, validated
through a strict point-in-time gate, and shipped to a live daily pick feed.

```
batter_hits   batter_rbis   batter_total_bases   batter_home_runs
runs_scored   batter_strikeouts   pitcher_strikeouts   pitcher_outs
h2h   spreads   totals
```

`runs_scored` is read as **batter runs scored** (`batter_runs_scored` upstream),
not a team-runs market — it sits alongside the other batter props in the brief.
Say so if the intent was team totals and it is a one-line change in
`config.MARKETS`.

Every market gets an honest **PASS / FAIL / VETO**, with a named cause and a
specific lever for each failure. Markets that pass ship live and fire daily.
Markets that fail are killed or veto-filtered — none are left running as they
were.

## Result

**0 of 11 markets clear the gate.** Three have genuinely positive out-of-sample
returns with confidence intervals that include zero, and run veto-filtered at a
high EV floor; eight are killed.

| | markets |
|---|---|
| `live` | none |
| `veto_filtered` | `totals` (+2.18% ROI), `batter_total_bases` (+1.79%), `spreads` (+0.06%) |
| `killed` | the other eight |

Closing-line value is positive on all eleven (+0.11% to +0.49%), so the price
selection genuinely works — it is simply smaller than the ~2.2-point hold it
has to overcome. The binding constraint is independent sample: intervals are
clustered by game, and a 1-2 point edge needs on the order of 10,000
independent games to clear a 95% bar. Four full seasons is 9,731 games in
total, and each market fires on a fraction of them.

**[docs/FINDINGS.md](docs/FINDINGS.md) is the full analysis** -- the arithmetic,
the two findings that mattered (the winner's curse on shopped prices, and
longshot bias), what was ruled out and why, and what would actually change the
answer.

---

## What the edge actually is

Two distinct sources, and the reports say which one is carrying each market:

1. **Price dispersion.** Twenty-plus books quote the same proposition and they
   disagree. A de-vigged consensus weighted toward the sharp end — Pinnacle,
   and the zero-vig exchanges (novig, prophetx, Betfair, Matchbook) — is a much
   better probability estimate than any one book's price, so a soft book
   quoting well away from it is a takeable edge.

2. **The line ladder.** Books do not quote every line with equal care. A dozen
   of them price Over 0.5 hits and argue about it all afternoon; Over 2.5 is
   posted by two and then left alone. `ladder.py` learns, per market, the
   empirical relationship between the consensus at the well-attended line and
   the probability of clearing every other rung — so the sparse lines become
   priceable instead of being discarded, which is where a lot of the edge is.
   The same applies to alternate run lines and totals, which we ingest
   explicitly: ~30 rungs per game per book against two on the main number.

3. **Projection.** A per-market model over point-in-time features (rolling
   per-plate-appearance form, opposing starter, park run environment, rest)
   adjusts the consensus where the features genuinely add information.

The model is **anchored to the market by construction**:

```
logit(p_final) = logit(p_consensus) + s · (logit(p_model) − logit(p_consensus))
```

`s` is fitted on each training fold by log-loss. If a market is efficient, `s`
is driven toward zero, the model collapses to the consensus, and any edge that
remains is line-shopping — which is real, but a different thing, and the report
says so rather than dressing it up as a projection edge. A booster free to
output any probability will always find structure in a million rows; anchored
and shrunk, it cannot drift far without earning it out of sample.

---

## Point-in-time integrity

The decision instant is **first pitch minus 30 minutes** — lineups are public,
prop markets are liquid, and a human could actually place the bet.

- **Odds** come from a historical snapshot taken at that instant. Closing
  prices are pulled separately and are used *only* to measure CLV; they never
  feed a decision.
- **Features** are computed on a frame sorted by date, grouped by entity, and
  shifted by one game before any window is applied. Same-day games are excluded
  too, so a doubleheader's first game cannot inform its second.
- **Actual batting order is deliberately unused.** It is public at decision
  time, but the only available source for it is the play-by-play, which exists
  only for players who actually batted — keying on it would leak the fact that
  the player appeared at all. A rolling average of recent batting order is used
  instead.
- **Thresholds and calibration are fitted inside the training window.** Picking
  an EV cut on the evaluation data is the quiet leak that flatters most
  backtests.

`leakage.py` does not take any of this on trust. It **recomputes** rolling
features from scratch for a random sample of rows using only strictly-prior
games and requires a match; if a stored value instead matches a window that
includes the current game, the market is vetoed. Leakage outranks every
statistical result — a market showing +110% ROI with a confidence interval
nowhere near zero is not a discovery, it is a bug with good manners.

---

## The gate

A market must clear all of:

| Check | Bar |
|---|---|
| leakage | clean as-of audit — vetoes outright |
| sample | ≥ 300 graded out-of-sample bets |
| roi | bootstrap 95% CI lower bound above zero |
| family | survives Benjamini-Hochberg across all eleven markets |
| clv | positive closing-line value where closing prices exist |
| stability | edge present across folds, not one hot month |
| decay | recent slice not confidently losing |
| calibration | predicted probabilities track realised frequencies |

The family-wise correction matters: testing eleven markets and reporting the
best one turns a 5% false-positive rate into roughly 43%. A market that is
nominally significant but does not survive BH is reported as such, not shipped.

**Deployment follows the verdict.** `live` fires at its fitted threshold;
`veto_filtered` fires only at a much higher EV cut and book count, so a market
that is being improved cannot bleed while it waits; `killed` does not fire.

---

## Data

| | |
|---|---|
| Odds | The Odds API v4 — historical snapshots, regions `us`/`us2`/`eu`/`us_ex`/`us_dfs` |
| Results | ESPN public API — boxscores and play-by-play |
| Seasons | 2024, 2025, 2026 |

Total bases is derived from play-by-play, because ESPN's batting line has no
2B/3B column. MLB's own `statsapi.mlb.com` does not resolve from this network,
which is why ESPN is the results source.

Settlement distinguishes **win / loss / push / void**. A player who was quoted
a prop but never appeared has the bet returned by the book — roughly 2% of
quoted batter props on a typical slate. Grading those as losses is a silent,
systematic pessimism; grading them as wins is fraud. Both are avoided.

---

## Layout

```
src/mlbedge/
  config.py      markets, books, regions, seasons, anchor weights
  oddsapi.py     Odds API client — cached, resumable, credit-metered
  espn.py        results client + stat extraction
  ingest_odds.py backfill driver
  ingest_results.py
  normalize.py   raw snapshots -> one tidy quote table
  match.py       game join (doubleheaders, renames) + player join
  grade.py       settlement: win/loss/push/void
  odds.py        price maths, four de-vig models
  devig.py       leave-one-out cross-book consensus
  features.py    point-in-time form
  dataset.py     proposition and candidate frames
  model.py       per-market shrunk market-adjustment model
  backtest.py    walk-forward, CLV, calibration
  leakage.py     as-of audit
  gate.py        PASS / FAIL / VETO + cause + lever
  production.py  deployable bundles
  picks.py       live pick generation
  scheduler.py   daily firing
api/server.py    pick + market + performance API
web/index.html   dashboard
```

---

## Running it

```bash
pip install -r requirements.txt
echo "ODDS_API_KEY=..." > .env

python scripts/backfill.py --seasons 2024 2025 2026 --closing 2026
python scripts/validate.py --seasons 2024 2025 2026      # the gate
python scripts/train_production.py                        # deployable models

uvicorn api.server:app --port 8000                        # dashboard
python scripts/daily.py --loop                            # fire daily
python scripts/grade_picks.py                             # settle what fired
```

Every ingest stage is cached on disk and idempotent: an interrupted backfill
resumes at zero credit cost, and re-parsing never spends credits — the client
runs cache-only outside of ingestion so it cannot.

```bash
python -m pytest tests/ -q
```

The tests generate each market with a known pathology — real edge, efficient
pricing, too-thin sample, an edge that dies mid-sample, miscalibration, a
planted leak — and assert the gate reaches the right verdict *for the right
reason*.

---

## Reports

| File | Contents |
|---|---|
| `reports/market_report.md` | per-market verdict, metrics, cause and lever |
| `reports/gate_results.csv` | the same, machine-readable |
| `reports/bets.parquet` | every out-of-sample bet the gate judged |
| `reports/leakage.csv` | the as-of audit per market |
| `models/manifest.json` | what is deployed, at what threshold, and why |
