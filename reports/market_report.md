# MLB market validation — Phase 1

Seasons: 2025  
Walk-forward: 5 expanding-window folds, 2025-07-05 → 2025-09-28  
Verdict: **0/11 PASS**, 4 VETO (leakage)

## Summary

| Market | Verdict | Deploy | Bets | ROI | 95% CI | p | CLV | Hit | Shrink |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| `batter_hits` | **VETO** | killed | 2138 | -2.73% | -7.60% … +2.44% | 0.8713 | n/a | 44.3% | 0.42 |
| `batter_home_runs` | **VETO** | killed | 633 | +9.53% | -6.87% … +26.51% | 0.1288 | n/a | 42.3% | 0.31 |
| `batter_rbis` | **FAIL** | killed | 3078 | -1.12% | -6.03% … +3.72% | 0.6706 | n/a | 41.9% | 0.36 |
| `batter_strikeouts` | **VETO** | killed | 14 | -48.77% | -87.36% … +1.54% | 0.9924 | n/a | 28.6% | 0.32 |
| `batter_total_bases` | **VETO** | killed | 2241 | -1.57% | -5.89% … +2.86% | 0.7537 | n/a | 46.7% | 0.51 |
| `h2h` | **FAIL** | veto_filtered | 989 | +3.72% | -3.28% … +10.91% | 0.1532 | +0.0000 | 46.6% | 0.26 |
| `pitcher_outs` | **FAIL** | killed | 1482 | -1.16% | -6.49% … +4.22% | 0.6632 | n/a | 47.3% | 0.56 |
| `pitcher_strikeouts` | **FAIL** | killed | 873 | -0.76% | -7.83% … +6.54% | 0.5787 | n/a | 46.2% | 0.49 |
| `runs_scored` | **FAIL** | killed | 802 | -12.21% | -20.31% … -3.95% | 0.9988 | n/a | 36.8% | 0.81 |
| `spreads` | **FAIL** | killed | 781 | -2.24% | -9.31% … +4.85% | 0.7369 | -0.0018 | 51.1% | 0.43 |
| `totals` | **FAIL** | killed | 1107 | -1.49% | -7.38% … +4.43% | 0.6933 | -0.0026 | 49.9% | 0.97 |

### How to read this

- **ROI** is flat-stake return per unit risked on out-of-sample bets only; every fold's model, calibration and EV threshold were fitted strictly before that fold began.
- **95% CI** is a percentile bootstrap over bets. A market passes only if the lower bound clears zero — a positive point estimate with an interval straddling zero is not evidence.
- **p** is a one-sided bootstrap p-value, then corrected across all eleven markets with Benjamini-Hochberg. Testing eleven things and reporting the best one turns a 5% false-positive rate into roughly 43%.
- **CLV** is mean closing-line value in probability terms, matched on the same book and the same proposition. It is measured on every bet rather than through outcome noise, so it is the lower-variance evidence that an edge is real.
- **Major-book ROI** restricts to DraftKings, FanDuel, BetMGM, Caesars, ESPN Bet, BetRivers and Fanatics. 'Best price across twenty books' overstates what is executable; an edge that survives only at obscure or offshore shops is a different product from one available at DraftKings.
- **Shrink** is how far the model was allowed to move from the market consensus, fitted per fold on training data. Near zero means the edge is line-shopping rather than projection — a real edge, but a different one, and worth knowing which.

## Per-market detail

### `batter_hits` — Batter hits — **VETO**

- Bets: **2138** (947W / 1191L / 0P)
- ROI: **-2.73%** (95% CI -7.60% … +2.44%), p = 0.8713
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 20% profitable, largest fold = n/a of profit
- Calibration error: 0.0490; average price +55
- Executable at major US books only: 975 bets at -2.21%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 1/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 870 | -1.59% |
| 2025-07-26 | 147 | -19.64% |
| 2025-08-16 | 344 | +4.36% |
| 2025-09-06 | 769 | -3.93% |
| 2025-09-27 | 8 | -6.25% |

</details>

### `batter_home_runs` — Batter home runs — **VETO**

- Bets: **633** (268W / 365L / 0P)
- ROI: **+9.53%** (95% CI -6.87% … +26.51%), p = 0.1288
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 60% profitable, largest fold = 98% of profit
- Calibration error: 0.0505; average price +221
- Executable at major US books only: 259 bets at +22.74%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: conditional_signal -- p_primary separates residual outcomes at AUC 0.741 -- implausible without leakage

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 37 | -52.00% |
| 2025-07-26 | 115 | +11.68% |
| 2025-08-16 | 448 | +13.21% |
| 2025-09-06 | 32 | -0.14% |
| 2025-09-27 | 1 | +700.00% |

</details>

### `batter_rbis` — Batter RBIs — **FAIL**

- Bets: **3078** (1289W / 1789L / 0P)
- ROI: **-1.12%** (95% CI -6.03% … +3.72%), p = 0.6706
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.0287; average price +82
- Executable at major US books only: 1212 bets at -7.48%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 1525 | +0.24% |
| 2025-07-26 | 59 | +38.73% |
| 2025-08-16 | 863 | -0.84% |
| 2025-09-06 | 620 | -8.29% |
| 2025-09-27 | 11 | -22.29% |

</details>

### `batter_strikeouts` — Batter strikeouts — **VETO**

- Bets: **14** (4W / 10L / 0P)
- ROI: **-48.77%** (95% CI -87.36% … +1.54%), p = 0.9924
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 2 folds, 0% profitable, largest fold = n/a of profit
- Calibration error: 0.2755; average price -68
- Executable at major US books only: 0 bets at n/a

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 2/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

### `batter_total_bases` — Batter total bases — **VETO**

- Bets: **2241** (1047W / 1194L / 0P)
- ROI: **-1.57%** (95% CI -5.89% … +2.86%), p = 0.7537
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.0532; average price +54
- Executable at major US books only: 822 bets at +2.96%

**Cause: `leakage`**

**Lever.** A feature or price carries information from at or after the decision instant. Fix the as-of boundary and re-validate from scratch. No statistical result from this market means anything until it is clean. Detail: VETO: asof::bat_hits_ppa_r10 -- 2/250 sampled rows match a window that includes the current game -- the feature sees its own outcome

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 435 | +3.72% |
| 2025-07-26 | 510 | +5.32% |
| 2025-08-16 | 406 | -19.02% |
| 2025-09-06 | 889 | -0.02% |
| 2025-09-27 | 1 | -100.00% |

</details>

### `h2h` — Moneyline — **FAIL**

- Bets: **989** (461W / 528L / 0P)
- ROI: **+3.72%** (95% CI -3.28% … +10.91%), p = 0.1532
- CLV: +0.0000 on 3 matched bets; beat the close 0.0%
- Stability: 5 folds, 60% profitable, largest fold = 68% of profit
- Calibration error: 0.0436; average price +73
- Executable at major US books only: 36 bets at -1.72%

**Cause: `no_edge`**

**Lever.** The market prices this efficiently at the books we can reach. The only lever with real headroom is a sharper anchor: weight the consensus harder toward the zero-vig exchanges (novig, prophetx) and Pinnacle and re-fit the anchor weights on train folds, then re-test. If the edge is still flat, this is a line-shopping market only -- run it veto-filtered at a high EV cut rather than killing it.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 190 | +3.79% |
| 2025-07-26 | 278 | +5.87% |
| 2025-08-16 | 250 | +10.01% |
| 2025-09-06 | 245 | -0.70% |
| 2025-09-27 | 26 | -38.64% |

</details>

### `pitcher_outs` — Pitcher outs — **FAIL**

- Bets: **1482** (701W / 781L / 0P)
- ROI: **-1.16%** (95% CI -6.49% … +4.22%), p = 0.6632
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 60% profitable, largest fold = n/a of profit
- Calibration error: 0.0670; average price +42
- Executable at major US books only: 365 bets at +2.70%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 409 | -4.13% |
| 2025-07-26 | 390 | -2.68% |
| 2025-08-16 | 396 | +0.91% |
| 2025-09-06 | 271 | +1.12% |
| 2025-09-27 | 16 | +22.14% |

</details>

### `pitcher_strikeouts` — Pitcher strikeouts — **FAIL**

- Bets: **873** (403W / 470L / 0P)
- ROI: **-0.76%** (95% CI -7.83% … +6.54%), p = 0.5787
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.0783; average price +66
- Executable at major US books only: 152 bets at -9.27%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 174 | -7.38% |
| 2025-07-26 | 387 | +2.55% |
| 2025-08-16 | 169 | -5.57% |
| 2025-09-06 | 106 | -1.35% |
| 2025-09-27 | 37 | +19.48% |

</details>

### `runs_scored` — Batter runs scored — **FAIL**

- Bets: **802** (295W / 507L / 0P)
- ROI: **-12.21%** (95% CI -20.31% … -3.95%), p = 0.9988
- CLV: n/a on 0 matched bets; beat the close n/a
- Stability: 5 folds, 0% profitable, largest fold = n/a of profit
- Calibration error: 0.1059; average price +124
- Executable at major US books only: 663 bets at -16.23%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 118 | -4.67% |
| 2025-07-26 | 70 | -19.88% |
| 2025-08-16 | 239 | -6.53% |
| 2025-09-06 | 335 | -12.82% |
| 2025-09-27 | 40 | -49.86% |

</details>

### `spreads` — Run line — **FAIL**

- Bets: **781** (399W / 382L / 0P)
- ROI: **-2.24%** (95% CI -9.31% … +4.85%), p = 0.7369
- CLV: -0.0018 on 1 matched bets; beat the close 0.0%
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.0594; average price -26
- Executable at major US books only: 53 bets at -10.20%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 103 | +9.24% |
| 2025-07-26 | 117 | -8.54% |
| 2025-08-16 | 282 | +0.00% |
| 2025-09-06 | 265 | -5.47% |
| 2025-09-27 | 14 | -17.96% |

</details>

### `totals` — Game total — **FAIL**

- Bets: **1107** (552W / 555L / 0P)
- ROI: **-1.49%** (95% CI -7.38% … +4.43%), p = 0.6933
- CLV: -0.0026 on 4 matched bets; beat the close 25.0%
- Stability: 5 folds, 40% profitable, largest fold = n/a of profit
- Calibration error: 0.1683; average price -9
- Executable at major US books only: 128 bets at +1.06%

**Cause: `negative_edge`**

**Lever.** Losing, not merely flat: the selection rule is picking the wrong side systematically. Check the de-vig model first -- a multiplicative de-vig on a longshot-heavy market overstates longshot probability and will reliably buy the wrong tail. Re-run with Shin and power and compare. Kill it live until it is positive out of sample.

<details><summary>Per-fold breakdown</summary>

| Fold | Bets | ROI |
|---|---:|---:|
| 2025-07-05 | 300 | +8.47% |
| 2025-07-26 | 320 | -1.18% |
| 2025-08-16 | 230 | -9.67% |
| 2025-09-06 | 233 | -10.02% |
| 2025-09-27 | 24 | +31.31% |

</details>
