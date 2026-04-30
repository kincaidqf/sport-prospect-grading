# Text model interpretability

Methods: **synthetic phrase probes** (Δ vs neutral baseline), **aggregated occlusion** (original − masked prediction), **corpus log-odds** (top vs bottom predicted quartiles).

## Caveats

- Probes are out-of-distribution; prefer **relative** ordering and Δ vs baseline.
- Occlusion and log-odds use **word-level** n-grams; subword effects are not shown.
- Player names and common NBA team tokens are stripped for log-odds; occlusion uses raw reports.

## VORP (regression)

### Probes (top / bottom Δ vs baseline)
| phrase | category | Δ |
| --- | --- | --- |
| G-League prospect | Projection | 0.1292 |
| fringe NBA player | Projection | 0.1235 |
| elite passer | Skills | 0.0629 |
| All-Star potential | Projection | 0.0586 |
| NBA-ready frame | Size / Frame | 0.0409 |
| below average athleticism | Athleticism | 0.0400 |
| explosive athlete | Athleticism | 0.0099 |
| poor free throw shooter | Shooting | 0.0029 |
| high basketball IQ | Motor / Character | -0.0013 |
| lockdown defender | Defense | -0.0050 |
| average athlete | Athleticism | -0.0059 |
| inconsistent jumper | Shooting | -0.0068 |
| poor defensive instincts | Defense | -0.0092 |
| high level athlete | Athleticism | -0.0169 |
| low basketball IQ | Motor / Character | -0.0205 |

| phrase | category | Δ |
| --- | --- | --- |
| limited burst | Athleticism | -0.1992 |
| low motor | Motor / Character | -0.1890 |
| high motor | Motor / Character | -0.1664 |
| poor handle | Skills | -0.1595 |
| lottery pick | Projection | -0.1508 |
| undersized | Size / Frame | -0.1465 |
| limited shooting range | Shooting | -0.1348 |
| great length | Size / Frame | -0.1321 |
| questionable work ethic | Motor / Character | -0.1306 |
| blue collar | Motor / Character | -0.1298 |
| relentless competitor | Motor / Character | -0.1261 |
| two-way player | Projection | -0.1134 |
| rim protector | Defense | -0.0985 |
| slight build | Size / Frame | -0.0968 |
| turnover prone | Skills | -0.0840 |

### Occlusion (mean Δ over reports)
| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0711 |

| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0711 |

### Log-odds (predicted top vs bottom quartile)
| ngram | log_odds |
| --- | --- |
| n | 10.0410 |
| pf 2009 | 9.9156 |
| pps 2009 | 9.9156 |
| 250 | 9.7782 |
| regularly | 9.7782 |
| pick n | 9.7782 |
| squad | 9.7782 |
| offensive repertoire | 9.6263 |
| measured 7 | 9.6263 |
| marq | 9.4566 |
| msst | 9.4566 |
| motenko | 9.4566 |
| 1 13 | 9.4566 |
| dnp | 9.4566 |
| n roll | 9.4566 |

| ngram | log_odds |
| --- | --- |
| overseas | -10.0047 |
| last two | -9.8349 |
| solid court | -9.8349 |
| conn | -9.8349 |
| wash | -9.8349 |
| clem | -9.8349 |
| southern | -9.8349 |
| dipped | -9.6425 |
| gained | -9.6425 |
| good quickness | -9.6425 |
| team win | -9.6425 |
| u16 european | -9.6425 |
| fastest | -9.6425 |
| goal percentage | -9.6425 |
| around screens | -9.6425 |

### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)

**Positive-associated:** _none_

**Negative-associated:** _none_

### VADER sentiment vs predictions (Pearson / Spearman)
| section | Pearson r | Spearman r | n |
| --- | --- | --- | --- |
| strengths | 0.0726 | 0.0672 | 911 |
| weaknesses | 0.0221 | 0.0494 | 919 |
| outlook | 0.0377 | 0.0069 | 301 |

## is_star

### Probes (top / bottom Δ vs baseline)
| phrase | category | Δ |
| --- | --- | --- |
| G-League prospect | Projection | -0.0107 |
| fringe NBA player | Projection | -0.0118 |
| below average athleticism | Athleticism | -0.0121 |
| high level athlete | Athleticism | -0.0216 |
| high basketball IQ | Motor / Character | -0.0220 |
| low basketball IQ | Motor / Character | -0.0221 |
| All-Star potential | Projection | -0.0236 |
| explosive athlete | Athleticism | -0.0241 |
| NBA-ready frame | Size / Frame | -0.0250 |
| great length | Size / Frame | -0.0252 |
| average athlete | Athleticism | -0.0262 |
| slight build | Size / Frame | -0.0268 |
| blue collar | Motor / Character | -0.0268 |
| needs to add strength | Size / Frame | -0.0281 |
| rotation player | Projection | -0.0291 |

| phrase | category | Δ |
| --- | --- | --- |
| advanced ball handler | Skills | -0.0475 |
| limited burst | Athleticism | -0.0440 |
| questionable work ethic | Motor / Character | -0.0399 |
| elite shooter | Shooting | -0.0397 |
| switchable defender | Defense | -0.0396 |
| relentless competitor | Motor / Character | -0.0394 |
| limited shooting range | Shooting | -0.0391 |
| weak defender | Defense | -0.0391 |
| turnover prone | Skills | -0.0381 |
| elite finisher at the rim | Skills | -0.0380 |
| two-way player | Projection | -0.0375 |
| future starter | Projection | -0.0367 |
| undersized | Size / Frame | -0.0361 |
| lottery pick | Projection | -0.0361 |
| knockdown shooter from deep | Shooting | -0.0354 |

### Occlusion (mean Δ over reports)
| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | 0.0065 |

| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | 0.0065 |

### Log-odds (predicted top vs bottom quartile)
| ngram | log_odds |
| --- | --- |
| u17 | 10.7206 |
| u16 americas | 10.7206 |
| u17 world | 10.6426 |
| ku | 10.4728 |
| pf 2009 | 9.7952 |
| pick n | 9.7952 |
| pps 2009 | 9.7952 |
| isu | 9.7952 |
| impacts | 9.6433 |
| projections | 9.6433 |
| 1st step | 9.4736 |
| n roll | 9.4736 |
| offensive per | 9.4736 |
| rose | 9.4736 |
| msst | 9.4736 |

| ngram | log_odds |
| --- | --- |
| syracuse | -10.4023 |
| plug | -10.1395 |
| conn | -9.8179 |
| 4 00 | -9.8179 |
| overseas | -9.8179 |
| year career | -9.8179 |
| southern | -9.8179 |
| clem | -9.8179 |
| wash | -9.8179 |
| tajh | -9.6255 |
| ingle 6 | -9.6255 |
| goal percentage | -9.6255 |
| good quickness | -9.6255 |
| career high | -9.6255 |
| wildcats | -9.6255 |

### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)

**Positive-associated:** _none_

**Negative-associated:** _none_

### VADER sentiment vs predictions (Pearson / Spearman)
| section | Pearson r | Spearman r | n |
| --- | --- | --- | --- |
| strengths | 0.0741 | 0.0883 | 911 |
| weaknesses | 0.0387 | 0.0851 | 919 |
| outlook | 0.0394 | -0.0253 | 301 |

## survived_3yrs

### Probes (top / bottom Δ vs baseline)
| phrase | category | Δ |
| --- | --- | --- |
| fringe NBA player | Projection | 0.0507 |
| G-League prospect | Projection | 0.0430 |
| NBA-ready frame | Size / Frame | 0.0404 |
| elite passer | Skills | 0.0371 |
| poor free throw shooter | Shooting | 0.0364 |
| explosive athlete | Athleticism | 0.0298 |
| high basketball IQ | Motor / Character | 0.0285 |
| average athlete | Athleticism | 0.0280 |
| All-Star potential | Projection | 0.0271 |
| below average athleticism | Athleticism | 0.0253 |
| advanced ball handler | Skills | 0.0233 |
| future starter | Projection | 0.0228 |
| low basketball IQ | Motor / Character | 0.0207 |
| knockdown shooter from deep | Shooting | 0.0204 |
| inconsistent jumper | Shooting | 0.0198 |

| phrase | category | Δ |
| --- | --- | --- |
| blue collar | Motor / Character | -0.0121 |
| low motor | Motor / Character | -0.0113 |
| lottery pick | Projection | -0.0091 |
| limited burst | Athleticism | -0.0085 |
| undersized | Size / Frame | -0.0080 |
| poor handle | Skills | -0.0063 |
| questionable work ethic | Motor / Character | -0.0061 |
| high motor | Motor / Character | -0.0046 |
| great length | Size / Frame | -0.0043 |
| relentless competitor | Motor / Character | -0.0018 |
| two-way player | Projection | -0.0013 |
| slight build | Size / Frame | 0.0016 |
| rim protector | Defense | 0.0044 |
| weak defender | Defense | 0.0048 |
| turnover prone | Skills | 0.0048 |

### Occlusion (mean Δ over reports)
| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0145 |

| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0145 |

### Log-odds (predicted top vs bottom quartile)
| ngram | log_odds |
| --- | --- |
| matter | 10.2834 |
| 250 | 10.0612 |
| 128 | 9.7984 |
| pf 2009 | 9.6465 |
| offensive repertoire | 9.6465 |
| motenko | 9.6465 |
| 4 4 | 9.6465 |
| pps 2009 | 9.6465 |
| extremely long | 9.4768 |
| rutg | 9.4768 |
| msst | 9.4768 |
| squad | 9.4768 |
| regularly | 9.4768 |
| c | 9.4768 |
| spin moves | 9.4768 |

| ngram | log_odds |
| --- | --- |
| armour | -12.0232 |
| armour association | -12.0232 |
| secondary handler | -10.2737 |
| rotational | -10.1363 |
| clem | -9.8147 |
| dipped | -9.8147 |
| conn | -9.8147 |
| team win | -9.6223 |
| attack closeouts | -9.6223 |
| scrimmages | -9.6223 |
| 2011 durant | -9.6223 |
| gained | -9.6223 |
| bump | -9.6223 |
| around screens | -9.6223 |
| usf | -9.4003 |

### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)

**Positive-associated:** _none_

**Negative-associated:** _none_

### VADER sentiment vs predictions (Pearson / Spearman)
| section | Pearson r | Spearman r | n |
| --- | --- | --- | --- |
| strengths | 0.0699 | 0.0752 | 911 |
| weaknesses | 0.0026 | 0.0649 | 919 |
| outlook | 0.0013 | -0.0125 | 301 |

## became_starter

### Probes (top / bottom Δ vs baseline)
| phrase | category | Δ |
| --- | --- | --- |
| below average athleticism | Athleticism | -0.0132 |
| great length | Size / Frame | -0.0160 |
| G-League prospect | Projection | -0.0191 |
| high level athlete | Athleticism | -0.0197 |
| slight build | Size / Frame | -0.0227 |
| inconsistent jumper | Shooting | -0.0228 |
| elite passer | Skills | -0.0231 |
| blue collar | Motor / Character | -0.0237 |
| high motor | Motor / Character | -0.0241 |
| explosive athlete | Athleticism | -0.0245 |
| needs to add strength | Size / Frame | -0.0247 |
| average athlete | Athleticism | -0.0256 |
| fringe NBA player | Projection | -0.0261 |
| low motor | Motor / Character | -0.0261 |
| All-Star potential | Projection | -0.0268 |

| phrase | category | Δ |
| --- | --- | --- |
| lottery pick | Projection | -0.0502 |
| knockdown shooter from deep | Shooting | -0.0478 |
| advanced ball handler | Skills | -0.0474 |
| limited burst | Athleticism | -0.0454 |
| future starter | Projection | -0.0433 |
| elite shooter | Shooting | -0.0426 |
| switchable defender | Defense | -0.0404 |
| limited shooting range | Shooting | -0.0391 |
| two-way player | Projection | -0.0372 |
| elite finisher at the rim | Skills | -0.0364 |
| poor free throw shooter | Shooting | -0.0354 |
| rim protector | Defense | -0.0342 |
| rotation player | Projection | -0.0342 |
| poor handle | Skills | -0.0321 |
| relentless competitor | Motor / Character | -0.0320 |

### Occlusion (mean Δ over reports)
| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0049 |

| ngram | n_reports | mean Δ |
| --- | --- | --- |
| nba | 20 | -0.0049 |

### Log-odds (predicted top vs bottom quartile)
| ngram | log_odds |
| --- | --- |
| gauntlet | 10.9915 |
| adidas gauntlet | 10.9915 |
| good tools | 9.9279 |
| 8 fg | 9.7905 |
| isu | 9.7905 |
| impacts | 9.7905 |
| coordinated | 9.7905 |
| matter | 9.7905 |
| long ways | 9.6387 |
| shot 40 | 9.6387 |
| projections | 9.6387 |
| gambling | 9.6387 |
| 303 | 9.4689 |
| msst | 9.4689 |
| 154 | 9.4689 |

| ngram | log_odds |
| --- | --- |
| syracuse | -10.5224 |
| plug | -10.1442 |
| overseas | -9.9923 |
| marquette | -9.9923 |
| 4 00 | -9.8226 |
| goal percentage | -9.8226 |
| clem | -9.8226 |
| conn | -9.8226 |
| wash | -9.8226 |
| southern | -9.6302 |
| career high | -9.6302 |
| 8 standing | -9.6302 |
| visenberg | -9.6302 |
| limitless | -9.6302 |
| limitless range | -9.6302 |

### Cross-method agreement (term in ≥2 of probe / occlusion / log-odds top-15 lists)

**Positive-associated:** _none_

**Negative-associated:** _none_

### VADER sentiment vs predictions (Pearson / Spearman)
| section | Pearson r | Spearman r | n |
| --- | --- | --- | --- |
| strengths | 0.0780 | 0.1152 | 911 |
| weaknesses | 0.0374 | 0.0798 | 919 |
| outlook | 0.0303 | -0.0773 | 301 |
