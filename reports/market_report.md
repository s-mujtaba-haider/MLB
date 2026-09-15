# MLB market validation -- Phase 1

Seasons: 2023, 2024, 2025, 2026  
Walk-forward: 20 expanding-window folds, 2024-05-03 -> 2026-09-14  
Verdict: **0/11 PASS**

## Summary

| Market | Verdict | Deploy | Bets | Games | ROI | 95% CI | p | CLV | Hit | Shrink |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `batter_hits` | **FAIL** | killed | 5827 | 3155 | -0.12% | -3.89% ... +3.95% | 0.5256 | +0.0036 | 37.3% | 0.46 |
| `batter_home_runs` | **FAIL** | killed | 2907 | 1851 | -4.22% | -12.90% ... +4.81% | 0.8291 | +0.0034 | 22.1% | 0.52 |
| `batter_rbis` | **FAIL** | killed | 9833 | 3961 | -2.81% | -4.96% ... -0.61% | 0.9944 | +0.0022 | 51.6% | 0.31 |
| `batter_strikeouts` | **FAIL** | killed | 698 | 550 | -1.82% | -9.72% ... +6.07% | 0.6707 | +0.0000 | 48.3% | 0.44 |
| `batter_total_bases` | **FAIL** | veto_filtered | 3672 | 2311 | +2.09% | -1.35% ... +5.70% | 0.1305 | +0.0027 | 51.1% | 0.67 |
| `h2h` | **FAIL** | veto_filtered | 2939 | 2939 | +1.35% | -3.07% ... +5.57% | 0.2772 | +0.0026 | 43.1% | 0.28 |
| `pitcher_outs` | **FAIL** | killed | 5193 | 3547 | -2.41% | -5.27% ... +0.54% | 0.9480 | +0.0039 | 47.0% | 0.26 |
| `pitcher_strikeouts` | **FAIL** | killed | 6478 | 3947 | -1.34% | -4.59% ... +1.96% | 0.7897 | +0.0049 | 44.3% | 0.41 |
| `runs_scored` | **FAIL** | killed | 6433 | 2967 | -6.05% | -8.64% ... -3.43% | 1.0000 | +0.0013 | 52.1% | 0.41 |
| `spreads` | **FAIL** | killed | 13310 | 4492 | -2.17% | -6.86% ... +2.58% | 0.8105 | +0.0027 | 25.7% | 0.07 |
| `totals` | **FAIL** | veto_filtered | 3594 | 1241 | +1.26% | -6.92% ... +9.72% | 0.3787 | +0.0056 | 43.4% | 0.24 |

### How to read this

- **ROI** is flat-stake return per unit risked on out-of-sample bets only; every fold's model, calibration and EV threshold were fitted strictly before that fold began.
- **95% CI** is a percentile bootstrap over **games**, not bets. A market passes only if the lower bound clears zero -- a positive point estimate with an interval straddling zero is not evidence.
- **p** is a one-sided bootstrap p-value, then corrected across all eleven markets with Benjamini-Hochberg. Testing eleven things and reporting the best one turns a 5% false-positive rate into roughly 43%.
- **CLV** is mean closing-line value in probability terms, matched on the same book and the same proposition. It is measured on every bet rather than through outcome noise, so it is the lower-variance evidence that an edge is real.
- **Major-book ROI** restricts to DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, BetRivers and Fanatics. 'Best price across twenty books' overstates what is executable; an edge that survives only at obscure or offshore shops is a different product from one available at DraftKings.
- **Shrink** is how far the model was allowed to move from the market consensus, fitted per fold on training data. Near zero means the edge is line-shopping rather than projection -- a real edge, but a different one, and worth knowing which.

## Per-market detail

### `batter_hits` -- Batter hits -- **FAIL**

- Bets: **5827** (2175W / 3652L / 0P) across **3155** distinct games
- ROI: **-0.12%** (95% CI -3.89% ... +3.95%), p = 0.5256
- CLV: +0.0036 on 1663 matched bets; beat the close 18.3%
- Stability: 14 folds, 50% profitable, largest fold = n/a of profit
- Calibration error: 0.0223; average price +183
- Executable at major US books only: 4145 bets at -0.91%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 905 | -1.87% |
| 2024-06-17 | 812 | -2.99% |
| 2024-08-01 | 518 | -2.39% |
| 2024-09-15 | 24 | +12.71% |
| 2025-03-14 | 97 | -12.51% |
| 2025-04-28 | 334 | -7.96% |
| 2025-06-12 | 634 | +3.44% |
| 2025-07-27 | 366 | +17.90% |
| 2025-09-10 | 430 | +1.13% |
| 2026-03-09 | 274 | -8.04% |
| 2026-04-23 | 828 | +3.24% |
| 2026-06-07 | 90 | +5.16% |
| 2026-07-22 | 472 | -5.16% |
| 2026-09-05 | 43 | +11.44% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| betrivers | 2183 | +3.47% | +75.7 |
| fanduel | 749 | -11.69% | -87.6 |
| betmgm | 738 | -0.89% | -6.6 |
| novig | 674 | +0.17% | +1.2 |
| prophetx | 520 | +2.20% | +11.5 |
| draftkings | 372 | -4.18% | -15.6 |
| bovada | 288 | -0.27% | -0.8 |
| betonlineag | 196 | +9.80% | +19.2 |
| fanatics | 94 | -8.05% | -7.6 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 3940 | +0.59% |
| ladder | 1887 | -1.61% |

</details>

### `batter_home_runs` -- Batter home runs -- **FAIL**

- Bets: **2907** (642W / 2265L / 0P) across **1851** distinct games
- ROI: **-4.22%** (95% CI -12.90% ... +4.81%), p = 0.8291
- CLV: +0.0034 on 677 matched bets; beat the close 28.8%
- Stability: 14 folds, 43% profitable, largest fold = n/a of profit
- Calibration error: 0.0269; average price +536
- Executable at major US books only: 1613 bets at -6.07%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 401 | +6.14% |
| 2024-06-17 | 133 | -3.51% |
| 2024-08-01 | 137 | -8.69% |
| 2024-09-15 | 101 | -50.59% |
| 2025-03-14 | 145 | +2.92% |
| 2025-04-28 | 460 | -15.66% |
| 2025-06-12 | 136 | +10.41% |
| 2025-07-27 | 126 | +3.26% |
| 2025-09-10 | 581 | -9.74% |
| 2026-03-09 | 123 | +16.33% |
| 2026-04-23 | 97 | -22.09% |
| 2026-06-07 | 273 | +16.07% |
| 2026-07-22 | 176 | -3.36% |
| 2026-09-05 | 18 | -55.56% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| prophetx | 629 | -3.70% | -23.3 |
| fanduel | 625 | -14.19% | -88.7 |
| novig | 508 | -6.27% | -31.8 |
| betrivers | 484 | +7.47% | +36.1 |
| betmgm | 450 | -8.85% | -39.8 |
| pinnacle | 128 | +20.92% | +26.8 |
| williamhill_us | 41 | -23.17% | -9.5 |
| bovada | 15 | -13.33% | -2.0 |
| betonlineag | 14 | +41.07% | +5.8 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 2352 | -5.11% |
| ladder | 555 | -0.45% |

</details>

### `batter_rbis` -- Batter RBIs -- **FAIL**

- Bets: **9833** (5074W / 4759L / 0P) across **3961** distinct games
- ROI: **-2.81%** (95% CI -4.96% ... -0.61%), p = 0.9944
- CLV: +0.0022 on 2589 matched bets; beat the close 16.2%
- Stability: 14 folds, 29% profitable, largest fold = n/a of profit
- Calibration error: 0.0163; average price -34
- Executable at major US books only: 5474 bets at -2.52%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 566 | -7.23% |
| 2024-06-17 | 650 | -10.22% |
| 2024-08-01 | 433 | -2.23% |
| 2024-09-15 | 220 | -8.10% |
| 2025-03-14 | 687 | -0.53% |
| 2025-04-28 | 1434 | -4.26% |
| 2025-06-12 | 1361 | -2.32% |
| 2025-07-27 | 1715 | -3.59% |
| 2025-09-10 | 127 | +3.39% |
| 2026-03-09 | 46 | +14.69% |
| 2026-04-23 | 392 | +1.04% |
| 2026-06-07 | 306 | -4.03% |
| 2026-07-22 | 1759 | -0.99% |
| 2026-09-05 | 137 | +22.76% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| prophetx | 2943 | -5.04% | -148.3 |
| betmgm | 2565 | -1.16% | -29.6 |
| betrivers | 1516 | -3.25% | -49.3 |
| novig | 1415 | +0.77% | +10.9 |
| draftkings | 689 | -5.77% | -39.7 |
| fanatics | 280 | +1.40% | +3.9 |
| williamhill_us | 260 | +2.15% | +5.6 |
| fanduel | 164 | -17.44% | -28.6 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 9333 | -2.37% |
| ladder | 500 | -11.04% |

</details>

### `batter_strikeouts` -- Batter strikeouts -- **FAIL**

- Bets: **698** (337W / 361L / 0P) across **550** distinct games
- ROI: **-1.82%** (95% CI -9.72% ... +6.07%), p = 0.6707
- CLV: +0.0000 on 2 matched bets; beat the close 0.0%
- Stability: 10 folds, 20% profitable, largest fold = n/a of profit
- Calibration error: 0.0325; average price +4
- Executable at major US books only: 0 bets at n/a

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 154 | +7.53% |
| 2024-06-17 | 165 | -2.24% |
| 2024-08-01 | 282 | -1.76% |
| 2024-09-15 | 24 | -17.56% |
| 2025-03-14 | 5 | -11.00% |
| 2025-04-28 | 1 | -100.00% |
| 2025-06-12 | 49 | -18.94% |
| 2025-07-27 | 16 | -1.55% |
| 2026-03-09 | 1 | +62.50% |
| 2026-07-22 | 1 | -100.00% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| hardrockbet | 566 | -3.85% | -21.8 |
| fliff | 132 | +6.87% | +9.1 |

</details>

### `batter_total_bases` -- Batter total bases -- **FAIL**

- Bets: **3672** (1876W / 1796L / 0P) across **2311** distinct games
- ROI: **+2.09%** (95% CI -1.35% ... +5.70%), p = 0.1305
- CLV: +0.0027 on 1579 matched bets; beat the close 17.7%
- Stability: 14 folds, 57% profitable, largest fold = 53% of profit
- Calibration error: 0.0324; average price +15
- Executable at major US books only: 1246 bets at +1.03%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 173 | +1.23% |
| 2024-06-17 | 165 | -0.12% |
| 2024-08-01 | 305 | -2.65% |
| 2024-09-15 | 173 | +10.56% |
| 2025-03-14 | 208 | -5.39% |
| 2025-04-28 | 421 | +3.67% |
| 2025-06-12 | 452 | -2.96% |
| 2025-07-27 | 100 | +40.60% |
| 2025-09-10 | 42 | +15.53% |
| 2026-03-09 | 280 | +3.64% |
| 2026-04-23 | 657 | +4.56% |
| 2026-06-07 | 371 | -5.38% |
| 2026-07-22 | 291 | -1.68% |
| 2026-09-05 | 34 | +33.64% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| pinnacle | 856 | +0.95% | +8.2 |
| prophetx | 700 | +1.54% | +10.8 |
| betmgm | 424 | +4.24% | +18.0 |
| betrivers | 400 | -1.92% | -7.7 |
| novig | 323 | +0.45% | +1.4 |
| betonlineag | 289 | +6.89% | +19.9 |
| bovada | 224 | +13.18% | +29.5 |
| fanatics | 183 | -7.51% | -13.7 |
| williamhill_us | 158 | +14.39% | +22.7 |
| draftkings | 45 | -1.14% | -0.5 |
| fanduel | 36 | -16.39% | -5.9 |
| mybookieag | 34 | -17.43% | -5.9 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 2707 | +2.15% |
| ladder | 965 | +1.93% |

</details>

### `h2h` -- Moneyline -- **FAIL**

- Bets: **2939** (1267W / 1672L / 0P) across **2939** distinct games
- ROI: **+1.35%** (95% CI -3.07% ... +5.57%), p = 0.2772
- CLV: +0.0026 on 1094 matched bets; beat the close 49.9%
- Stability: 14 folds, 57% profitable, largest fold = 94% of profit
- Calibration error: 0.0370; average price +114
- Executable at major US books only: 381 bets at +5.74%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 228 | +5.10% |
| 2024-06-17 | 382 | -9.30% |
| 2024-08-01 | 211 | +10.34% |
| 2024-09-15 | 62 | -29.03% |
| 2025-03-14 | 116 | -24.11% |
| 2025-04-28 | 52 | -38.74% |
| 2025-06-12 | 261 | +13.43% |
| 2025-07-27 | 400 | +9.27% |
| 2025-09-10 | 130 | -7.62% |
| 2026-03-09 | 260 | +8.28% |
| 2026-04-23 | 319 | +6.27% |
| 2026-06-07 | 218 | +7.61% |
| 2026-07-22 | 285 | -4.89% |
| 2026-09-05 | 15 | +9.07% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| novig | 915 | +2.64% | +24.1 |
| prophetx | 852 | +3.95% | +33.6 |
| pinnacle | 345 | -5.57% | -19.2 |
| betrivers | 175 | +6.05% | +10.6 |
| lowvig | 169 | -11.15% | -18.9 |
| betonlineag | 149 | +10.69% | +15.9 |
| williamhill_us | 102 | +15.33% | +15.6 |
| betus | 96 | -18.84% | -18.1 |
| betmgm | 53 | +13.19% | +7.0 |
| draftkings | 28 | -33.75% | -9.4 |
| fanduel | 22 | -15.95% | -3.5 |
| mybookieag | 19 | +36.38% | +6.9 |
| bovada | 13 | -51.92% | -6.8 |

</details>

### `pitcher_outs` -- Pitcher outs -- **FAIL**

- Bets: **5193** (2440W / 2753L / 0P) across **3547** distinct games
- ROI: **-2.41%** (95% CI -5.27% ... +0.54%), p = 0.9480
- CLV: +0.0039 on 2108 matched bets; beat the close 28.2%
- Stability: 14 folds, 36% profitable, largest fold = n/a of profit
- Calibration error: 0.0404; average price +39
- Executable at major US books only: 1411 bets at -3.31%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 131 | +6.29% |
| 2024-06-17 | 524 | -10.23% |
| 2024-08-01 | 80 | +2.44% |
| 2024-09-15 | 56 | +6.07% |
| 2025-03-14 | 276 | -1.43% |
| 2025-04-28 | 259 | -7.33% |
| 2025-06-12 | 816 | -0.46% |
| 2025-07-27 | 553 | +2.51% |
| 2025-09-10 | 290 | +12.01% |
| 2026-03-09 | 233 | -2.33% |
| 2026-04-23 | 450 | -8.23% |
| 2026-06-07 | 561 | -3.64% |
| 2026-07-22 | 795 | -3.97% |
| 2026-09-05 | 169 | -7.47% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| novig | 1805 | -4.56% | -82.4 |
| prophetx | 1439 | +0.79% | +11.3 |
| betrivers | 418 | -4.26% | -17.8 |
| draftkings | 256 | +2.16% | +5.5 |
| fanduel | 252 | -6.66% | -16.8 |
| pinnacle | 230 | -5.98% | -13.8 |
| betonlineag | 191 | +10.96% | +20.9 |
| betmgm | 182 | -10.83% | -19.7 |
| fanatics | 165 | +8.92% | +14.7 |
| williamhill_us | 138 | -9.18% | -12.7 |
| bovada | 117 | -12.38% | -14.5 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 4530 | -2.40% |
| ladder | 663 | -2.46% |

</details>

### `pitcher_strikeouts` -- Pitcher strikeouts -- **FAIL**

- Bets: **6478** (2872W / 3606L / 0P) across **3947** distinct games
- ROI: **-1.34%** (95% CI -4.59% ... +1.96%), p = 0.7897
- CLV: +0.0049 on 1642 matched bets; beat the close 37.9%
- Stability: 14 folds, 36% profitable, largest fold = n/a of profit
- Calibration error: 0.0353; average price +89
- Executable at major US books only: 3194 bets at -3.35%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 934 | -0.40% |
| 2024-06-17 | 824 | -6.89% |
| 2024-08-01 | 323 | -7.47% |
| 2024-09-15 | 110 | +13.66% |
| 2025-03-14 | 737 | -0.63% |
| 2025-04-28 | 591 | +0.80% |
| 2025-06-12 | 603 | +9.36% |
| 2025-07-27 | 505 | -1.56% |
| 2025-09-10 | 149 | +0.43% |
| 2026-03-09 | 447 | -3.36% |
| 2026-04-23 | 727 | -2.60% |
| 2026-06-07 | 165 | -8.39% |
| 2026-07-22 | 237 | -8.47% |
| 2026-09-05 | 126 | +1.35% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| betrivers | 2086 | -5.64% | -117.7 |
| novig | 1674 | +3.09% | +51.7 |
| prophetx | 968 | -2.28% | -22.0 |
| fanduel | 370 | +2.08% | +7.7 |
| draftkings | 312 | -2.32% | -7.3 |
| betmgm | 306 | +1.48% | +4.5 |
| bovada | 243 | +3.03% | +7.4 |
| mybookieag | 161 | -12.94% | -20.8 |
| betonlineag | 122 | +3.34% | +4.1 |
| pinnacle | 116 | +0.17% | +0.2 |
| williamhill_us | 77 | +2.52% | +1.9 |
| fanatics | 43 | +8.90% | +3.8 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 4481 | +1.54% |
| ladder | 1997 | -7.78% |

</details>

### `runs_scored` -- Batter runs scored -- **FAIL**

- Bets: **6433** (3354W / 3079L / 0P) across **2967** distinct games
- ROI: **-6.05%** (95% CI -8.64% ... -3.43%), p = 1.0000
- CLV: +0.0013 on 1567 matched bets; beat the close 10.7%
- Stability: 14 folds, 21% profitable, largest fold = n/a of profit
- Calibration error: 0.0309; average price -60
- Executable at major US books only: 4606 bets at -6.38%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 946 | -11.85% |
| 2024-06-17 | 191 | -1.95% |
| 2024-08-01 | 161 | -15.36% |
| 2024-09-15 | 332 | -1.49% |
| 2025-03-14 | 780 | -5.56% |
| 2025-04-28 | 1305 | -10.83% |
| 2025-06-12 | 36 | -3.95% |
| 2025-07-27 | 884 | -4.06% |
| 2025-09-10 | 198 | +6.91% |
| 2026-03-09 | 200 | -7.29% |
| 2026-04-23 | 153 | -0.64% |
| 2026-06-07 | 643 | +0.45% |
| 2026-07-22 | 595 | -4.06% |
| 2026-09-05 | 9 | +15.91% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| betmgm | 2158 | -3.19% | -68.9 |
| novig | 1203 | -5.08% | -61.1 |
| betrivers | 1071 | -13.73% | -147.0 |
| draftkings | 807 | -5.84% | -47.1 |
| prophetx | 607 | -6.27% | -38.1 |
| williamhill_us | 475 | -0.18% | -0.9 |
| fanduel | 95 | -31.37% | -29.8 |
| bovada | 17 | +20.90% | +3.6 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 5221 | -6.71% |
| ladder | 1212 | -3.23% |

</details>

### `spreads` -- Run line -- **FAIL**

- Bets: **13310** (3419W / 9891L / 0P) across **4492** distinct games
- ROI: **-2.17%** (95% CI -6.86% ... +2.58%), p = 0.8105
- CLV: +0.0027 on 206 matched bets; beat the close 47.1%
- Stability: 14 folds, 36% profitable, largest fold = n/a of profit
- Calibration error: 0.0190; average price +334
- Executable at major US books only: 10643 bets at -1.58%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 301 | +1.35% |
| 2024-06-17 | 2468 | +4.67% |
| 2024-08-01 | 2621 | +0.50% |
| 2024-09-15 | 462 | -10.47% |
| 2025-03-14 | 322 | -27.44% |
| 2025-04-28 | 1051 | -5.47% |
| 2025-06-12 | 1252 | +5.65% |
| 2025-07-27 | 1412 | -5.83% |
| 2025-09-10 | 700 | -12.49% |
| 2026-03-09 | 640 | +0.88% |
| 2026-04-23 | 484 | -2.28% |
| 2026-06-07 | 1308 | -5.41% |
| 2026-07-22 | 137 | -27.27% |
| 2026-09-05 | 152 | -9.44% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| draftkings | 4940 | -4.11% | -203.0 |
| fanduel | 3034 | -1.29% | -39.1 |
| pinnacle | 1777 | -4.84% | -85.9 |
| betrivers | 1083 | +8.40% | +91.0 |
| fanatics | 635 | -12.57% | -79.8 |
| betmgm | 527 | +5.14% | +27.1 |
| prophetx | 360 | -12.47% | -44.9 |
| espnbet | 281 | +9.63% | +27.1 |
| novig | 152 | -1.77% | -2.7 |
| williamhill_us | 143 | +6.00% | +8.6 |
| lowvig | 89 | -13.34% | -11.9 |
| rebet | 72 | -7.97% | -5.7 |
| fliff | 61 | +15.31% | +9.3 |
| hardrockbet | 61 | +60.75% | +37.1 |
| bovada | 33 | -25.91% | -8.6 |
| betus | 26 | -15.97% | -4.2 |
| mybookieag | 25 | +9.00% | +2.2 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 10056 | +0.09% |
| ladder | 3254 | -9.14% |

</details>

### `totals` -- Game total -- **FAIL**

- Bets: **3594** (1558W / 2036L / 0P) across **1241** distinct games
- ROI: **+1.26%** (95% CI -6.92% ... +9.72%), p = 0.3787
- CLV: +0.0056 on 63 matched bets; beat the close 46.0%
- Stability: 14 folds, 57% profitable, largest fold = 156% of profit
- Calibration error: 0.0927; average price +157
- Executable at major US books only: 2210 bets at -2.83%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 712 | -4.81% |
| 2024-06-17 | 143 | +20.47% |
| 2024-08-01 | 113 | +56.70% |
| 2024-09-15 | 73 | +13.41% |
| 2025-03-14 | 627 | -6.70% |
| 2025-04-28 | 97 | -11.31% |
| 2025-06-12 | 86 | +38.94% |
| 2025-07-27 | 228 | +30.99% |
| 2025-09-10 | 164 | +10.43% |
| 2026-03-09 | 78 | -15.86% |
| 2026-04-23 | 1140 | -10.25% |
| 2026-06-07 | 72 | +62.82% |
| 2026-07-22 | 54 | -15.93% |
| 2026-09-05 | 7 | +11.52% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| pinnacle | 815 | +11.00% | +89.7 |
| betmgm | 776 | -22.06% | -171.1 |
| fanduel | 617 | -3.07% | -19.0 |
| fanatics | 352 | +30.15% | +106.1 |
| draftkings | 219 | +0.80% | +1.8 |
| ballybet | 201 | -14.76% | -29.7 |
| betrivers | 150 | -3.21% | -4.8 |
| betparx | 103 | -2.58% | -2.7 |
| novig | 84 | +37.35% | +31.4 |
| prophetx | 76 | -14.71% | -11.2 |
| williamhill_us | 59 | +24.00% | +14.2 |
| hardrockbet | 49 | +41.34% | +20.3 |
| espnbet | 37 | +28.26% | +10.5 |
| fliff | 18 | +21.96% | +4.0 |
| bovada | 14 | +10.89% | +1.5 |
| lowvig | 13 | +17.94% | +2.3 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 3472 | +0.48% |
| ladder | 122 | +23.57% |

</details>
