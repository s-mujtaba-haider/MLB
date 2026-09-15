# MLB market validation — Phase 1

Seasons: 2025  
Walk-forward: 5 expanding-window folds, 2025-07-05 → 2025-09-28  
Verdict: **0/11 PASS**, 4 VETO (leakage)

## Summary

| Market | Verdict | Deploy | Bets | ROI | 95% CI | p | CLV | Hit | Shrink |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| `batter_hits` | **VETO** | killed | 1203 | +0.89% | -6.16% … +7.93% | 0.4046 | n/a | 41.1% | 0.22 |
| `batter_home_runs` | **VETO** | killed | 379 | +6.82% | -13.34% … +28.88% | 0.2640 | n/a | 37.2% | 0.24 |
| `batter_rbis` | **FAIL** | killed | 1023 | -0.02% | -7.16% … +7.61% | 0.5082 | n/a | 50.1% | 0.09 |
| `batter_strikeouts` | **VETO** | killed | 5 | -37.91% | -100.00% … +31.72% | 0.9254 | n/a | 40.0% | 0.20 |
| `batter_total_bases` | **VETO** | killed | 2627 | -4.68% | -8.90% … -0.42% | 0.9845 | n/a | 44.2% | 0.27 |
| `h2h` | **FAIL** | veto_filtered | 775 | +6.15% | -1.75% … +14.33% | 0.0727 | +0.0000 | 47.4% | 0.33 |
| `pitcher_outs` | **FAIL** | killed | 1227 | -0.68% | -6.82% … +5.37% | 0.5865 | n/a | 44.9% | 0.36 |
| `pitcher_strikeouts` | **FAIL** | veto_filtered | 1080 | +2.27% | -4.17% … +8.91% | 0.2572 | n/a | 45.3% | 0.22 |
| `runs_scored` | **FAIL** | killed | 9 | -27.39% | -100.00% … +45.23% | 0.8279 | n/a | 33.3% | 0.25 |
| `spreads` | **FAIL** | killed | 866 | -1.07% | -7.64% … +5.26% | 0.6303 | n/a | 52.0% | 0.38 |
| `totals` | **FAIL** | killed | 509 | -7.86% | -16.79% … +1.55% | 0.9558 | -0.0027 | 44.8% | 0.61 |

### How to read this

- **ROI** is flat-stake return per unit risked on out-of-sample bets only; every fold's model, calibration and EV threshold were fitted strictly before that fold began.
- **95% CI** is a percentile bootstrap over bets. A market passes only if the lower bound clears zero — a positive point estimate with an interval straddling zero is not evidence.
- **p** is a one-sided bootstrap p-value, then corrected across all eleven markets with Benjamini-Hochberg. Testing eleven things and reporting the best one turns a 5% false-positive rate into roughly 43%.
- **CLV** is mean closing-line value in probability terms, matched on the same book and the same proposition. It is measured on every bet rather than through outcome noise, so it is the lower-variance evidence that an edge is real.
- **Major-book ROI** restricts to DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, BetRivers and Fanatics. 'Best price across twenty books' overstates what is executable; an edge that survives only at obscure or offshore shops is a different product from one available at DraftKings.
- **Shrink** is how far the model was allowed to move from the market consensus, fitted per fold on training data. Near zero means the edge is line-shopping rather than projection — a real edge, but a different one, and worth knowing which.

## Per-market detail

### `batter_hits` — Batter hits — **VETO**

- Bets: **1203** (495W / 708L / 0P)
- ROI: **+0.89%** (95% CI -6.16% … +7.93%), p = 0.4046
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 60% profitable, largest fold = 137% of profit
- Calibration error: 0.0482; average price +114
- Executable at major US books only: 461 bets at -0.25%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 1/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 337 | +4.35% |
| 2025-07-26 | 424 | -3.98% |
| 2025-08-16 | 181 | +7.52% |
| 2025-09-06 | 255 | +0.53% |
| 2025-09-27 | 6 | -33.49% |

</details>

### `batter_home_runs` — Batter home runs — **VETO**

- Bets: **379** (141W / 238L / 0P)
- ROI: **+6.82%** (95% CI -13.34% … +28.88%), p = 0.2640
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 80% profitable, largest fold = 44% of profit
- Calibration error: 0.0489; average price +311
- Executable at major US books only: 165 bets at +10.27%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 1/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 45 | +14.03% |
| 2025-07-26 | 196 | +2.72% |
| 2025-08-16 | 79 | +14.32% |
| 2025-09-06 | 54 | -9.18% |
| 2025-09-27 | 5 | +157.00% |

</details>

### `batter_rbis` — Batter RBIs — **FAIL**

- Bets: **1023** (513W / 510L / 0P)
- ROI: **-0.02%** (95% CI -7.16% … +7.61%), p = 0.5082
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 4 folds, 50% profitable, largest fold = n/a of profit
- Calibration error: 0.0467; average price -0
- Executable at major US books only: 520 bets at -5.32%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-26 | 123 | +13.81% |
| 2025-08-16 | 732 | -1.57% |
| 2025-09-06 | 125 | +6.05% |
| 2025-09-27 | 43 | -30.70% |

</details>

### `batter_strikeouts` — Batter strikeouts — **VETO**

- Bets: **5** (2W / 3L / 0P)
- ROI: **-37.91%** (95% CI -100.00% … +31.72%), p = 0.9254
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 2 folds, 50% profitable, largest fold = n/a of profit
- Calibration error: 0.6664; average price -243
- Executable at major US books only: 0 bets at n/a

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 2/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

### `batter_total_bases` — Batter total bases — **VETO**

- Bets: **2627** (1162W / 1465L / 0P)
- ROI: **-4.68%** (95% CI -8.90% … -0.42%), p = 0.9845
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 20% profitable, largest fold = n/a of profit
- Calibration error: 0.0364; average price +70
- Executable at major US books only: 893 bets at +0.06%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 2/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 240 | -7.37% |
| 2025-07-26 | 1504 | -1.85% |
| 2025-08-16 | 743 | -11.29% |
| 2025-09-06 | 129 | +7.27% |
| 2025-09-27 | 11 | -26.15% |

</details>

### `h2h` — Moneyline — **FAIL**

- Bets: **775** (367W / 408L / 0P)
- ROI: **+6.15%** (95% CI -1.75% … +14.33%), p = 0.0727
- CLV: +0.0000 on 2 matched bets; beat the close 0.0%
- Stability: 5 folds, 60% profitable, largest fold = 51% of profit
- Calibration error: 0.0480; average price +76
- Executable at major US books only: 30 bets at -12.47%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 140 | +17.45% |
| 2025-07-26 | 286 | +8.35% |
| 2025-08-16 | 108 | +6.15% |
| 2025-09-06 | 234 | -1.83% |
| 2025-09-27 | 7 | -42.94% |

</details>

### `pitcher_outs` — Pitcher outs — **FAIL**

- Bets: **1227** (551W / 676L / 0P)
- ROI: **-0.68%** (95% CI -6.82% … +5.37%), p = 0.5865
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 60% profitable, largest fold = n/a of profit
- Calibration error: 0.0445; average price +79
- Executable at major US books only: 230 bets at -2.22%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 356 | +3.67% |
| 2025-07-26 | 546 | -6.10% |
| 2025-08-16 | 208 | +6.11% |
| 2025-09-06 | 95 | -1.80% |
| 2025-09-27 | 22 | +4.26% |

</details>

### `pitcher_strikeouts` — Pitcher strikeouts — **FAIL**

- Bets: **1080** (489W / 591L / 0P)
- ROI: **+2.27%** (95% CI -4.17% … +8.91%), p = 0.2572
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 80% profitable, largest fold = 39% of profit
- Calibration error: 0.0333; average price +110
- Executable at major US books only: 140 bets at -10.52%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 351 | +1.30% |
| 2025-07-26 | 312 | +3.11% |
| 2025-08-16 | 235 | +0.91% |
| 2025-09-06 | 165 | -0.05% |
| 2025-09-27 | 17 | +48.47% |

</details>

### `runs_scored` — Batter runs scored — **FAIL**

- Bets: **9** (3W / 6L / 0P)
- ROI: **-27.39%** (95% CI -100.00% … +45.23%), p = 0.8279
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 3 folds, 67% profitable, largest fold = n/a of profit
- Calibration error: 0.3913; average price +150
- Executable at major US books only: 6 bets at -100.00%

**Cause: `insufficient_sample`**

**Lever.** Sample is the binding constraint, not the edge. Widen the bet universe before touching the model: lower min_books from 3 to 2 to admit thinly-quoted propositions, extend the backfill another season, and add the alternate lines this market posts (they are already in the raw cache and cost nothing to re-parse).

### `spreads` — Run line — **FAIL**

- Bets: **866** (450W / 416L / 0P)
- ROI: **-1.07%** (95% CI -7.64% … +5.26%), p = 0.6303
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.0438; average price -36
- Executable at major US books only: 62 bets at -14.23%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 58 | +22.48% |
| 2025-07-26 | 248 | -1.57% |
| 2025-08-16 | 272 | +2.05% |
| 2025-09-06 | 263 | -7.41% |
| 2025-09-27 | 25 | -18.09% |

</details>

### `totals` — Game total — **FAIL**

- Bets: **509** (228W / 281L / 0P)
- ROI: **-7.86%** (95% CI -16.79% … +1.55%), p = 0.9558
- CLV: -0.0027 on 3 matched bets; beat the close 33.3%
- Stability: 5 folds, 20% profitable, largest fold = n/a of profit
- Calibration error: 0.0712; average price +64
- Executable at major US books only: 29 bets at -38.05%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 220 | -5.43% |
| 2025-07-26 | 159 | -11.21% |
| 2025-08-16 | 61 | -14.14% |
| 2025-09-06 | 64 | +1.92% |
| 2025-09-27 | 5 | -57.00% |

</details>
