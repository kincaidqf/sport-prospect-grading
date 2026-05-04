# Text model interpretability

Methods: **synthetic phrase probes** (Δ vs neutral baseline), **aggregated occlusion** (original − masked prediction), **corpus log-odds** (top vs bottom predicted quartiles).

## Caveats

- Probes are out-of-distribution; prefer **relative** ordering and Δ vs baseline.
- Occlusion and log-odds use **word-level** n-grams; subword effects are not shown.
- Player names and common NBA team tokens are stripped for log-odds; occlusion uses raw reports.

## nba_role_zscore (regression)

### Probes (top / bottom Δ vs baseline)
| phrase | category | Δ |
| --- | --- | --- |
| fringe NBA player | Projection | 0.0452 |
| lottery pick | Projection | 0.0449 |
| high level athlete | Athleticism | 0.0438 |
| blue collar | Motor / Character | 0.0414 |
| elite passer | Skills | 0.0388 |
| All-Star potential | Projection | 0.0383 |
| G-League prospect | Projection | 0.0380 |
| explosive athlete | Athleticism | 0.0370 |
| relentless competitor | Motor / Character | 0.0369 |
| high basketball IQ | Motor / Character | 0.0348 |
| elite finisher at the rim | Skills | 0.0309 |
| NBA-ready frame | Size / Frame | 0.0292 |
| average athlete | Athleticism | 0.0240 |
| poor free throw shooter | Shooting | 0.0233 |
| great length | Size / Frame | 0.0231 |

| phrase | category | Δ |
| --- | --- | --- |
| slight build | Size / Frame | -0.0144 |
| limited shooting range | Shooting | -0.0087 |
| undersized | Size / Frame | -0.0084 |
| switchable defender | Defense | -0.0071 |
| needs to add strength | Size / Frame | 0.0005 |
| limited burst | Athleticism | 0.0020 |
| low motor | Motor / Character | 0.0032 |
| turnover prone | Skills | 0.0045 |
| knockdown shooter from deep | Shooting | 0.0094 |
| elite shooter | Shooting | 0.0108 |
| poor defensive instincts | Defense | 0.0114 |
| below average athleticism | Athleticism | 0.0116 |
| weak defender | Defense | 0.0123 |
| future starter | Projection | 0.0134 |
| poor handle | Skills | 0.0137 |

### Occlusion (mean Δ over reports)
| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 40 | 0.0121 |

| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 40 | 0.0121 |

### Log-odds (predicted top vs bottom quartile)
| ngram | log_odds |
| --- | --- |
| great speed | 9.8880 |
| aran 1 | 9.7182 |
| averaged nearly | 9.7182 |
| wash | 9.7182 |
| clem | 9.7182 |
| aran 3 | 9.5258 |
| u | 9.5258 |
| contributes | 9.5258 |
| clutch performer | 9.5258 |
| innate | 9.3038 |
| 7 turnovers | 9.3038 |
| practice | 9.3038 |
| doubled | 9.3038 |
| game solid | 9.3038 |
| okst | 9.3038 |

| ngram | log_odds |
| --- | --- |
| promising | -10.7424 |
| 4 fg | -10.1578 |
| wasserman | -9.8950 |
| four man | -9.7431 |
| 94 | -9.7431 |
| md | -9.7431 |
| foundation | -9.5733 |
| syr | -9.5733 |
| topg | -9.5733 |
| dive | -9.5733 |
| hes | -9.5733 |
| marq | -9.5733 |
| isu | -9.5733 |
| sie | -9.5733 |
| tex | -9.5733 |

### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)

**Positive-associated:** _none_

**Negative-associated:** _none_

### VADER sentiment vs predictions (Pearson / Spearman)
| section | Pearson r | Spearman r | n |
| --- | --- | --- | --- |
| strengths | 0.0240 | 0.0486 | 677 |
| weaknesses | 0.0218 | -0.0156 | 682 |
| outlook | 0.0586 | 0.0691 | 194 |
