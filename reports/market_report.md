# MLB market validation -- Phase 1

Seasons: 2023, 2024, 2025, 2026  
Walk-forward: 20 expanding-window folds, 2024-05-03 -> 2026-09-14  
Verdict: **0/11 PASS**

## Summary

| Market | Verdict | Deploy | Bets | Games | ROI | 95% CI | p | CLV | Hit | Shrink |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| `batter_hits` | **FAIL** | killed | 7826 | 3271 | -0.84% | -3.91% ... +2.33% | 0.6970 | +0.0041 | 38.5% | 0.46 |
| `batter_home_runs` | **FAIL** | killed | 3291 | 2016 | -13.13% | -21.64% ... -4.71% | 0.9986 | +0.0022 | 16.9% | 0.52 |
| `batter_rbis` | **FAIL** | killed | 2427 | 1332 | -5.12% | -9.72% ... -0.56% | 0.9851 | +0.0035 | 48.9% | 0.31 |
| `batter_strikeouts` | **FAIL** | killed | 1562 | 762 | -0.42% | -4.88% ... +4.10% | 0.5650 | n/a | 54.4% | 0.47 |
| `batter_total_bases` | **FAIL** | veto_filtered | 3971 | 2386 | +1.79% | -1.52% ... +5.16% | 0.1447 | +0.0027 | 51.6% | 0.68 |
| `h2h` | **FAIL** | killed | 2825 | 2825 | -0.87% | -5.04% ... +3.50% | 0.6494 | +0.0028 | 43.2% | 0.29 |
| `pitcher_outs` | **FAIL** | killed | 2238 | 1807 | -2.54% | -7.18% ... +2.15% | 0.8490 | +0.0043 | 45.4% | 0.29 |
| `pitcher_strikeouts` | **FAIL** | killed | 5158 | 3257 | -0.75% | -4.43% ... +2.83% | 0.6495 | +0.0049 | 45.0% | 0.41 |
| `runs_scored` | **FAIL** | killed | 2443 | 1193 | -2.44% | -6.60% ... +1.74% | 0.8750 | +0.0010 | 54.4% | 0.42 |
| `spreads` | **FAIL** | veto_filtered | 6411 | 2634 | +0.06% | -6.84% ... +6.72% | 0.4843 | +0.0042 | 24.6% | 0.08 |
| `totals` | **FAIL** | veto_filtered | 7274 | 1041 | +2.18% | -4.01% ... +8.35% | 0.2431 | +0.0040 | 41.8% | 0.27 |

### How to read this

- **ROI** is flat-stake return per unit risked on out-of-sample bets only; every fold's model, calibration and EV threshold were fitted strictly before that fold began.
- **95% CI** is a percentile bootstrap over **games**, not bets. A market passes only if the lower bound clears zero -- a positive point estimate with an interval straddling zero is not evidence.
- **p** is a one-sided bootstrap p-value, then corrected across all eleven markets with Benjamini-Hochberg. Testing eleven things and reporting the best one turns a 5% false-positive rate into roughly 43%.
- **CLV** is mean closing-line value in probability terms, matched on the same book and the same proposition. It is measured on every bet rather than through outcome noise, so it is the lower-variance evidence that an edge is real.
- **Major-book ROI** restricts to DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, BetRivers and Fanatics. 'Best price across twenty books' overstates what is executable; an edge that survives only at obscure or offshore shops is a different product from one available at DraftKings.
- **Shrink** is how far the model was allowed to move from the market consensus, fitted per fold on training data. Near zero means the edge is line-shopping rather than projection -- a real edge, but a different one, and worth knowing which.

## Per-market detail

### `batter_hits` -- Batter hits -- **FAIL**

- Bets: **7826** (3013W / 4813L / 0P) across **3271** distinct games
- ROI: **-0.84%** (95% CI -3.91% ... +2.33%), p = 0.6970
- CLV: +0.0041 on 1040 matched bets; beat the close 18.5%
- Stability: 14 folds, 36% profitable, largest fold = n/a of profit
- Calibration error: 0.0185; average price +163
- Executable at major US books only: 4977 bets at -0.80%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 469 | +5.19% |
| 2024-06-17 | 1807 | -2.70% |
| 2024-08-01 | 614 | -0.65% |
| 2024-09-15 | 43 | +10.28% |
| 2025-03-14 | 459 | +1.63% |
| 2025-04-28 | 37 | -55.95% |
| 2025-06-12 | 1225 | -0.40% |
| 2025-07-27 | 1863 | +1.54% |
| 2025-09-10 | 233 | -2.72% |
| 2026-03-09 | 428 | -0.37% |
| 2026-04-23 | 34 | -13.66% |
| 2026-06-07 | 190 | +0.91% |
| 2026-07-22 | 336 | -9.30% |
| 2026-09-05 | 88 | -11.51% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| betrivers | 2443 | +1.13% | +27.6 |
| novig | 1205 | -4.19% | -50.5 |
| betmgm | 877 | +0.30% | +2.6 |
| prophetx | 749 | -0.37% | -2.8 |
| fanduel | 713 | -9.50% | -67.8 |
| bovada | 644 | +2.37% | +15.3 |
| draftkings | 537 | -4.92% | -26.4 |
| fanatics | 388 | +3.10% | +12.0 |
| betonlineag | 248 | +5.27% | +13.1 |
| williamhill_us | 19 | +64.18% | +12.2 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 5850 | +0.32% |
| ladder | 1976 | -4.27% |

</details>

### `batter_home_runs` -- Batter home runs -- **FAIL**

- Bets: **3291** (555W / 2736L / 0P) across **2016** distinct games
- ROI: **-13.13%** (95% CI -21.64% ... -4.71%), p = 0.9986
- CLV: +0.0022 on 1367 matched bets; beat the close 25.6%
- Stability: 14 folds, 29% profitable, largest fold = n/a of profit
- Calibration error: 0.0282; average price +638
- Executable at major US books only: 1579 bets at -11.10%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 181 | -12.91% |
| 2024-06-17 | 72 | -10.72% |
| 2024-08-01 | 449 | +0.71% |
| 2024-09-15 | 210 | -24.64% |
| 2025-03-14 | 40 | -17.96% |
| 2025-04-28 | 633 | -15.97% |
| 2025-06-12 | 128 | -17.36% |
| 2025-07-27 | 74 | -30.31% |
| 2025-09-10 | 120 | +20.67% |
| 2026-03-09 | 206 | +12.27% |
| 2026-04-23 | 308 | -20.56% |
| 2026-06-07 | 17 | +34.47% |
| 2026-07-22 | 841 | -21.44% |
| 2026-09-05 | 12 | -100.00% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| prophetx | 1002 | -22.49% | -225.3 |
| novig | 588 | -7.62% | -44.8 |
| fanduel | 562 | -12.57% | -70.6 |
| betmgm | 503 | -10.34% | -52.0 |
| betrivers | 475 | -5.40% | -25.6 |
| pinnacle | 90 | +16.72% | +15.0 |
| williamhill_us | 30 | -75.00% | -22.5 |
| bovada | 20 | -29.31% | -5.9 |
| betonlineag | 12 | +33.33% | +4.0 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 2197 | -11.16% |
| ladder | 1094 | -17.09% |

</details>

### `batter_rbis` -- Batter RBIs -- **FAIL**

- Bets: **2427** (1187W / 1240L / 0P) across **1332** distinct games
- ROI: **-5.12%** (95% CI -9.72% ... -0.56%), p = 0.9851
- CLV: +0.0035 on 548 matched bets; beat the close 16.4%
- Stability: 14 folds, 50% profitable, largest fold = n/a of profit
- Calibration error: 0.0326; average price -9
- Executable at major US books only: 1434 bets at -5.12%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 765 | -7.07% |
| 2024-06-17 | 32 | -38.39% |
| 2024-08-01 | 71 | +4.73% |
| 2024-09-15 | 62 | +10.40% |
| 2025-03-14 | 132 | +0.46% |
| 2025-04-28 | 733 | -5.19% |
| 2025-06-12 | 34 | +44.28% |
| 2025-07-27 | 18 | +63.61% |
| 2025-09-10 | 14 | -100.00% |
| 2026-03-09 | 50 | +17.11% |
| 2026-04-23 | 70 | +1.06% |
| 2026-06-07 | 36 | -25.34% |
| 2026-07-22 | 145 | -16.64% |
| 2026-09-05 | 265 | -7.06% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| novig | 589 | -4.54% | -26.7 |
| betrivers | 552 | -5.78% | -31.9 |
| prophetx | 403 | -5.72% | -23.1 |
| betmgm | 402 | +0.90% | +3.6 |
| draftkings | 205 | -17.49% | -35.8 |
| williamhill_us | 166 | +1.62% | +2.7 |
| fanduel | 69 | -16.01% | -11.1 |
| fanatics | 40 | -2.21% | -0.9 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 2055 | -3.14% |
| ladder | 372 | -16.05% |

</details>

### `batter_strikeouts` -- Batter strikeouts -- **FAIL**

- Bets: **1562** (849W / 713L / 0P) across **762** distinct games
- ROI: **-0.42%** (95% CI -4.88% ... +4.10%), p = 0.5650
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 7 folds, 43% profitable, largest fold = n/a of profit
- Calibration error: 0.0380; average price -54
- Executable at major US books only: 0 bets at n/a

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 1228 | +1.45% |
| 2024-06-17 | 22 | +7.16% |
| 2024-08-01 | 203 | -5.17% |
| 2024-09-15 | 78 | -14.93% |
| 2025-03-14 | 9 | -25.00% |
| 2025-04-28 | 3 | -100.00% |
| 2025-07-27 | 19 | +7.47% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| hardrockbet | 1433 | -0.45% | -6.5 |
| fliff | 129 | -0.07% | -0.1 |

</details>

### `batter_total_bases` -- Batter total bases -- **FAIL**

- Bets: **3971** (2050W / 1921L / 0P) across **2386** distinct games
- ROI: **+1.79%** (95% CI -1.52% ... +5.16%), p = 0.1447
- CLV: +0.0027 on 2092 matched bets; beat the close 16.6%
- Stability: 14 folds, 64% profitable, largest fold = 67% of profit
- Calibration error: 0.0347; average price +7
- Executable at major US books only: 1357 bets at +1.39%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 152 | +3.62% |
| 2024-06-17 | 197 | +0.86% |
| 2024-08-01 | 259 | -0.95% |
| 2024-09-15 | 173 | +10.56% |
| 2025-03-14 | 195 | -3.98% |
| 2025-04-28 | 252 | +2.19% |
| 2025-06-12 | 283 | -4.10% |
| 2025-07-27 | 226 | +21.02% |
| 2025-09-10 | 75 | +20.40% |
| 2026-03-09 | 186 | +0.32% |
| 2026-04-23 | 430 | +4.76% |
| 2026-06-07 | 638 | -5.04% |
| 2026-07-22 | 851 | -0.09% |
| 2026-09-05 | 54 | +20.25% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| pinnacle | 922 | -1.64% | -15.1 |
| prophetx | 865 | +3.17% | +27.4 |
| betmgm | 540 | +2.84% | +15.3 |
| betrivers | 414 | -1.19% | -4.9 |
| novig | 308 | +0.36% | +1.1 |
| betonlineag | 306 | +1.08% | +3.3 |
| bovada | 178 | +22.21% | +39.5 |
| williamhill_us | 168 | +10.11% | +17.0 |
| fanatics | 159 | -1.09% | -1.7 |
| draftkings | 43 | -0.53% | -0.2 |
| mybookieag | 35 | -11.88% | -4.2 |
| fanduel | 33 | -19.70% | -6.5 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 3002 | +2.36% |
| ladder | 969 | +0.02% |

</details>

### `h2h` -- Moneyline -- **FAIL**

- Bets: **2825** (1221W / 1604L / 0P) across **2825** distinct games
- ROI: **-0.87%** (95% CI -5.04% ... +3.50%), p = 0.6494
- CLV: +0.0028 on 632 matched bets; beat the close 46.7%
- Stability: 14 folds, 43% profitable, largest fold = n/a of profit
- Calibration error: 0.0339; average price +97
- Executable at major US books only: 393 bets at +4.63%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 322 | +1.55% |
| 2024-06-17 | 353 | -9.00% |
| 2024-08-01 | 342 | +2.46% |
| 2024-09-15 | 68 | -17.97% |
| 2025-03-14 | 109 | -16.10% |
| 2025-04-28 | 38 | -34.22% |
| 2025-06-12 | 433 | +9.77% |
| 2025-07-27 | 332 | +7.49% |
| 2025-09-10 | 198 | -7.99% |
| 2026-03-09 | 136 | +11.31% |
| 2026-04-23 | 124 | -5.55% |
| 2026-06-07 | 54 | -23.30% |
| 2026-07-22 | 313 | -3.64% |
| 2026-09-05 | 3 | +27.67% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| prophetx | 782 | +1.72% | +13.5 |
| novig | 715 | +0.17% | +1.2 |
| pinnacle | 410 | -7.97% | -32.7 |
| betrivers | 187 | -0.34% | -0.6 |
| lowvig | 184 | -5.14% | -9.5 |
| betonlineag | 184 | -1.03% | -1.9 |
| betus | 125 | -11.94% | -14.9 |
| williamhill_us | 92 | +19.13% | +17.6 |
| betmgm | 64 | +18.50% | +11.8 |
| draftkings | 28 | -34.82% | -9.7 |
| mybookieag | 20 | +25.12% | +5.0 |
| fanduel | 20 | -16.64% | -3.3 |
| bovada | 12 | -28.50% | -3.4 |

</details>

### `pitcher_outs` -- Pitcher outs -- **FAIL**

- Bets: **2238** (1017W / 1221L / 0P) across **1807** distinct games
- ROI: **-2.54%** (95% CI -7.18% ... +2.15%), p = 0.8490
- CLV: +0.0043 on 686 matched bets; beat the close 28.0%
- Stability: 14 folds, 36% profitable, largest fold = n/a of profit
- Calibration error: 0.0547; average price +64
- Executable at major US books only: 539 bets at -1.80%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 89 | +2.07% |
| 2024-06-17 | 139 | -18.16% |
| 2024-08-01 | 44 | +9.98% |
| 2024-09-15 | 29 | -2.79% |
| 2025-03-14 | 167 | -1.66% |
| 2025-04-28 | 136 | -1.48% |
| 2025-06-12 | 544 | -4.60% |
| 2025-07-27 | 223 | +1.89% |
| 2025-09-10 | 146 | +18.67% |
| 2026-03-09 | 283 | -4.54% |
| 2026-04-23 | 140 | -13.19% |
| 2026-06-07 | 50 | -8.74% |
| 2026-07-22 | 111 | +2.43% |
| 2026-09-05 | 137 | -4.12% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| novig | 822 | -4.58% | -37.7 |
| prophetx | 638 | -0.43% | -2.7 |
| betrivers | 143 | +2.47% | +3.5 |
| pinnacle | 108 | -11.65% | -12.6 |
| fanduel | 107 | -11.77% | -12.6 |
| draftkings | 93 | -3.22% | -3.0 |
| fanatics | 86 | +16.49% | +14.2 |
| betonlineag | 80 | +9.61% | +7.7 |
| williamhill_us | 57 | +11.94% | +6.8 |
| betmgm | 53 | -35.16% | -18.6 |
| bovada | 51 | -3.45% | -1.8 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 1945 | -2.65% |
| ladder | 293 | -1.80% |

</details>

### `pitcher_strikeouts` -- Pitcher strikeouts -- **FAIL**

- Bets: **5158** (2322W / 2836L / 0P) across **3257** distinct games
- ROI: **-0.75%** (95% CI -4.43% ... +2.83%), p = 0.6495
- CLV: +0.0049 on 1146 matched bets; beat the close 37.7%
- Stability: 14 folds, 43% profitable, largest fold = n/a of profit
- Calibration error: 0.0360; average price +82
- Executable at major US books only: 2382 bets at -1.34%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 782 | +1.73% |
| 2024-06-17 | 482 | -5.89% |
| 2024-08-01 | 64 | -14.30% |
| 2024-09-15 | 82 | +14.39% |
| 2025-03-14 | 676 | -3.15% |
| 2025-04-28 | 915 | -1.25% |
| 2025-06-12 | 346 | +11.25% |
| 2025-07-27 | 492 | -3.20% |
| 2025-09-10 | 134 | +0.18% |
| 2026-03-09 | 126 | -13.07% |
| 2026-04-23 | 682 | +1.57% |
| 2026-06-07 | 163 | -7.48% |
| 2026-07-22 | 99 | +6.73% |
| 2026-09-05 | 115 | -5.21% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| betrivers | 1519 | -5.46% | -83.0 |
| novig | 1439 | +3.10% | +44.5 |
| prophetx | 858 | -4.14% | -35.5 |
| betmgm | 270 | -0.52% | -1.4 |
| draftkings | 257 | +4.29% | +11.0 |
| fanduel | 257 | +9.58% | +24.6 |
| bovada | 170 | -4.52% | -7.7 |
| mybookieag | 128 | -9.93% | -12.7 |
| pinnacle | 91 | +3.81% | +3.5 |
| betonlineag | 90 | +0.92% | +0.8 |
| williamhill_us | 47 | +17.14% | +8.1 |
| fanatics | 32 | +27.81% | +8.9 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 3701 | +1.26% |
| ladder | 1457 | -5.87% |

</details>

### `runs_scored` -- Batter runs scored -- **FAIL**

- Bets: **2443** (1330W / 1113L / 0P) across **1193** distinct games
- ROI: **-2.44%** (95% CI -6.60% ... +1.74%), p = 0.8750
- CLV: +0.0010 on 127 matched bets; beat the close 8.7%
- Stability: 14 folds, 50% profitable, largest fold = n/a of profit
- Calibration error: 0.0337; average price -74
- Executable at major US books only: 1205 bets at -4.15%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 361 | +1.80% |
| 2024-06-17 | 137 | -1.17% |
| 2024-08-01 | 118 | -11.99% |
| 2024-09-15 | 22 | +15.98% |
| 2025-03-14 | 202 | -3.37% |
| 2025-04-28 | 112 | -31.17% |
| 2025-06-12 | 1317 | -0.91% |
| 2025-07-27 | 17 | -7.30% |
| 2025-09-10 | 26 | +0.59% |
| 2026-03-09 | 106 | -10.27% |
| 2026-04-23 | 11 | +95.41% |
| 2026-06-07 | 3 | +6.74% |
| 2026-07-22 | 5 | +4.38% |
| 2026-09-05 | 6 | +14.27% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| novig | 1017 | -0.25% | -2.5 |
| betmgm | 464 | -4.83% | -22.4 |
| betrivers | 377 | -8.41% | -31.7 |
| draftkings | 249 | -0.59% | -1.5 |
| prophetx | 216 | -4.13% | -8.9 |
| williamhill_us | 85 | +5.35% | +4.5 |
| fanduel | 30 | +3.33% | +1.0 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 2069 | -3.17% |
| ladder | 374 | +1.61% |

</details>

### `spreads` -- Run line -- **FAIL**

- Bets: **6411** (1578W / 4833L / 0P) across **2634** distinct games
- ROI: **+0.06%** (95% CI -6.84% ... +6.72%), p = 0.4843
- CLV: +0.0042 on 26 matched bets; beat the close 46.2%
- Stability: 14 folds, 36% profitable, largest fold = 3567% of profit
- Calibration error: 0.0275; average price +355
- Executable at major US books only: 5347 bets at +0.21%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 353 | -4.43% |
| 2024-06-17 | 1643 | +8.36% |
| 2024-08-01 | 943 | -0.32% |
| 2024-09-15 | 579 | -13.54% |
| 2025-03-14 | 377 | -22.77% |
| 2025-04-28 | 42 | -8.33% |
| 2025-06-12 | 1651 | +6.71% |
| 2025-07-27 | 229 | -5.15% |
| 2025-09-10 | 13 | +4.77% |
| 2026-03-09 | 50 | +24.80% |
| 2026-04-23 | 207 | +5.79% |
| 2026-06-07 | 277 | -20.72% |
| 2026-07-22 | 41 | -23.66% |
| 2026-09-05 | 6 | -66.33% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| draftkings | 2880 | -2.69% | -77.6 |
| fanduel | 1457 | +8.26% | +120.3 |
| pinnacle | 805 | -2.62% | -21.1 |
| betrivers | 506 | +6.28% | +31.8 |
| betmgm | 222 | -5.39% | -12.0 |
| espnbet | 119 | -22.77% | -27.1 |
| fanatics | 118 | -19.52% | -23.0 |
| prophetx | 63 | -28.72% | -18.1 |
| rebet | 53 | -8.81% | -4.7 |
| williamhill_us | 45 | -3.16% | -1.4 |
| fliff | 40 | +28.88% | +11.6 |
| hardrockbet | 28 | +68.78% | +19.3 |
| novig | 26 | +22.74% | +5.9 |
| lowvig | 23 | +24.94% | +5.7 |
| bovada | 15 | -17.00% | -2.6 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 4633 | +2.27% |
| ladder | 1778 | -5.70% |

</details>

### `totals` -- Game total -- **FAIL**

- Bets: **7274** (3039W / 4235L / 0P) across **1041** distinct games
- ROI: **+2.18%** (95% CI -4.01% ... +8.35%), p = 0.2431
- CLV: +0.0040 on 28 matched bets; beat the close 42.9%
- Stability: 14 folds, 71% profitable, largest fold = 44% of profit
- Calibration error: 0.0507; average price +139
- Executable at major US books only: 4407 bets at +0.98%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2024-05-03 | 5882 | -2.34% |
| 2024-06-17 | 91 | +26.83% |
| 2024-08-01 | 119 | +55.75% |
| 2024-09-15 | 73 | +13.41% |
| 2025-03-14 | 423 | +3.54% |
| 2025-04-28 | 96 | -10.26% |
| 2025-06-12 | 83 | +40.65% |
| 2025-07-27 | 151 | +45.92% |
| 2025-09-10 | 85 | +23.19% |
| 2026-03-09 | 78 | -15.86% |
| 2026-04-23 | 58 | +69.74% |
| 2026-06-07 | 75 | +60.46% |
| 2026-07-22 | 53 | -14.35% |
| 2026-09-05 | 7 | +30.14% |

</details>

<details><summary>Where the bets landed</summary>

This is the execution question: an edge concentrated at one book is only as good as that book's limits and how long it tolerates the action.

| Book | Bets | ROI | Profit (u) |
|---|---:|---:|---:|
| pinnacle | 2398 | +2.72% | +65.3 |
| fanduel | 1343 | -1.01% | -13.6 |
| draftkings | 1105 | -11.65% | -128.7 |
| betrivers | 973 | +3.41% | +33.2 |
| betmgm | 444 | +4.57% | +20.3 |
| fanatics | 278 | +33.07% | +91.9 |
| williamhill_us | 165 | +12.69% | +20.9 |
| hardrockbet | 133 | +16.51% | +22.0 |
| lowvig | 103 | +7.15% | +7.4 |
| espnbet | 99 | +19.32% | +19.1 |
| ballybet | 47 | +4.28% | +2.0 |
| novig | 44 | +39.13% | +17.2 |
| fliff | 38 | +10.46% | +4.0 |
| bovada | 33 | -6.12% | -2.0 |
| prophetx | 33 | -10.82% | -3.6 |
| mybookieag | 19 | +9.17% | +1.7 |
| betus | 13 | +33.84% | +4.4 |

</details>

<details><summary>Direct line vs ladder-priced</summary>

| Anchor | Bets | ROI |
|---|---:|---:|
| direct | 7192 | +1.58% |
| ladder | 82 | +55.01% |

</details>
