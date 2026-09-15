# MLB Phase 1 — what the data actually says

Four seasons (2023–2026), 9,731 games, ~19M quotes, all eleven markets ingested,
graded and validated. This is the honest read.

## Headline

**0 of 11 markets clear the research bar.** Three have genuinely positive
out-of-sample returns but confidence intervals that include zero:

| market | bets | games | OOS ROI | 95% CI (clustered) | p | CLV |
|---|---:|---:|---:|---|---:|---:|
| `totals` | 7,274 | ~1,200 | **+2.18%** | −4.01% … +8.35% | 0.243 | +0.40% |
| `batter_total_bases` | 3,971 | ~2,300 | **+1.79%** | −1.52% … +5.16% | 0.145 | +0.27% |
| `spreads` | 6,411 | ~4,500 | +0.06% | −6.84% … +6.72% | 0.484 | +0.42% |
| `batter_strikeouts` | 1,562 | ~550 | −0.42% | −4.88% … +4.10% | 0.565 | n/a |
| `pitcher_strikeouts` | 5,158 | ~3,900 | −0.75% | −4.43% … +2.83% | 0.650 | +0.49% |
| `batter_hits` | 7,826 | ~3,200 | −0.84% | −3.91% … +2.33% | 0.697 | +0.41% |
| `h2h` | 2,825 | 2,825 | −0.87% | −5.04% … +3.50% | 0.649 | +0.28% |
| `runs_scored` | 2,443 | ~1,500 | −2.44% | −6.60% … +1.74% | 0.875 | +0.11% |
| `pitcher_outs` | 2,238 | ~2,100 | −2.54% | −7.18% … +2.15% | 0.849 | +0.43% |
| `batter_rbis` | 2,427 | ~1,800 | −5.12% | −9.72% … −0.56% | 0.985 | +0.35% |
| `batter_home_runs` | 3,291 | ~1,850 | −13.13% | −21.64% … −4.71% | 0.999 | +0.22% |

## Why nothing passes: the arithmetic, not the effort

A market passes when `ROI > 1.96 × sigma / sqrt(independent games)`.

The interval is clustered **by game**, because bets inside a game are not
independent observations. Six alternate totals on one game are one observation:
if it goes over 9 it went over 8.5 too. Every batter in a fourteen-run game goes
over together. Resampling bets individually produces an interval roughly twice
too narrow — which is exactly how a market with no edge acquires a confidence
interval that clears zero.

That leaves a hard constraint:

| independent games bet | ROI needed to clear zero |
|---:|---:|
| 1,200 | ~5.7% |
| 3,000 | ~3.6% |
| 10,000 | ~2.0% |

Realistic MLB edges after the hold are one to two points. Clearing a 95% bar at
that edge size needs on the order of **10,000+ independent games per market**.
Four full seasons is 9,731 games *in total*, and each market fires on a fraction
of them. `totals` bet roughly 1,200 distinct games, so it would have needed a
5.7% edge — nearly triple what it produced.

**More features, more tuning and more seasons cannot close a gap of that shape.**
The binding constraint is how many independent baseball games exist.

## Positive CLV on every market, and why it isn't enough

Closing-line value is positive on all eleven markets, +0.11% to +0.49%. We do
consistently take better prices than the market closes at. But:

- beating the close by ~0.4 points of probability,
- while paying a hold of ~2.2 points per side,

is a losing trade. To profit you must beat the close by **more than the hold**,
not merely beat it. The measured CLV is roughly one-fifth of what is required.

Worth noting for the opposite reason: `totals` returns +2.18% where CLV alone
predicts about −3.5%. The model is finding real mispricing the closing line does
not fully correct. The edge is genuine — it is simply small relative to the
noise in 1,200 games.

## The two findings that mattered most

**1. The winner's curse dominates a shopped price.** Bucketing four seasons of
real candidates by apparent edge against the leave-one-out consensus:

| apparent edge | bets | consensus error | price error | ROI |
|---|---:|---:|---:|---:|
| 1–2% | 11,182 | −1.7% | −0.3% | −1.5% |
| 2–3% | 3,034 | −2.7% | −0.3% | +0.3% |
| 3–5% | 618 | −1.5% | **+2.0%** | +1.8% |

Below about three points *the book is right and the consensus is wrong*. Taking
the best of twenty numbers selects on each book's error, so conditional on a
price being the best available it is longer than the truth more often than the
consensus implies. `selection.py` corrects for this, fitted where the selection
happens and cross-fitted out of fold.

**2. The strategy collects on favourites and bleeds on longshots.** Realised win
rate minus the price-implied probability:

| price band | bets | realised − price | ROI |
|---|---:|---:|---:|
| −250 … −150 | 8,334 | **+0.89%** | **+1.41%** |
| worse than −250 | 3,905 | +0.53% | +0.71% |
| +150 … +250 | 13,982 | **−1.60%** | **−4.73%** |
| +400 and longer | 8,271 | −0.81% | −5.45% |

A de-vig that is a shade optimistic about the tail becomes a large apparent edge
at +300 and a negligible one at −200, so apparent edges out there are mostly the
de-vig's own error. Thirty thousand longshot bets were funding eighteen thousand
profitable ones. The bet universe is now segmented by (anchor, price band) and
each fold decides on its own training window which segments may fire.

## What was ruled out

- **Re-weighting the consensus.** Five schemes (prior, sharp-heavy, sharp-only,
  exchange-first, uniform) were compared by consensus log-loss on the fit
  season. Differences were 0.00001–0.0016 — noise. The de-vigged consensus is
  already near-optimal; there is no edge to be had by weighting books better.
- **Betting earlier than T−30.** The obvious idea, since converged markets are
  hardest to beat. Measured on cached snapshots: median hold 4.34% and 20 books
  at 6–12 hours out, versus 4.47% and 19 books inside the last hour. The market
  is fully formed half a day early. Genuinely soft lines exist at *opening*,
  days before, which the historical snapshots do not reach.
- **The line ladder as a betting source.** It improves probability estimates
  (totals ladder log-loss 0.397 against 0.599 for directly-priced lines) but is
  unreliable in the tails where alternate rungs live. Kept as a model feature,
  gated out of betting wherever the training fold says it does not collect.

## Per-market cause and lever

See `reports/market_report.md`, which carries the named cause and the specific
lever for each market. In summary:

- `totals`, `batter_total_bases`, `spreads` — **no_edge**: positive returns,
  interval includes zero. Lever is sample, not modelling. Each needs roughly
  three to four times the independent games at the observed edge. That means
  more seasons (2020–2022 featured markets are available from the same vendor;
  props are not, historically) or firing on a larger share of games at the same
  edge, which the threshold search already optimises against.
- `h2h`, `batter_hits`, `pitcher_strikeouts`, `batter_strikeouts` — **negative
  but within noise**. Slightly negative point estimates with intervals spanning
  zero; they are efficiently priced at the books we can reach.
- `batter_rbis`, `runs_scored`, `pitcher_outs` — **negative beyond noise** on
  the point estimate. Killed.
- `batter_home_runs` — **worst market by a distance** at −13.1%. Structurally a
  longshot market: almost every quote is a +300 to +900 Over, precisely the band
  where the de-vig error is largest. Lever: it needs a calibrated home-run
  model (batted-ball data, park-adjusted barrel rates), not a consensus
  adjustment. Without that it should not be run.

## Deployment

| deployment | markets |
|---|---|
| `live` | none |
| `veto_filtered` | `totals`, `batter_total_bases`, `spreads` |
| `killed` | the other eight |

Veto-filtered markets fire only above an 8% EV floor with at least five books
behind the price, so they cannot bleed while more evidence accumulates. Nothing
runs at its original threshold, and nothing that loses money runs at all.

## What would actually change the answer

1. **More independent games.** 2020–2022 featured-market history is available
   from the same vendor and would roughly double the game count for `h2h`,
   `spreads` and `totals`. Player props do not exist historically before
   May 2023, so the prop markets cannot be extended this way.
2. **A real projection model for the tail markets**, particularly home runs —
   batted-ball and park-adjusted inputs rather than a market adjustment. That is
   a different modelling exercise, not a tuning pass.
3. **A different decision point.** Everything here is priced 30 minutes before
   first pitch against a converged market. Opening lines are where the softness
   is; capturing them needs a live feed running for months, not a historical
   backfill.
4. **Accepting a deployment bar instead of a research bar** — positive ROI,
   positive CLV, stability, no leakage — and sizing by confidence. That is a
   legitimate business decision, and on current numbers it admits three markets,
   not eight. It should be labelled as judgement, not as proof.
