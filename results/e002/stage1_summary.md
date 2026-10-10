
## lang_mode = clean  (643 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_spatial | Light Conditions | 292 / 292 | 0 | 100.0 [98.7, 100.0] | 98.3 | +1.7 (±1.5) | ambiguous (ceiling: our variance term is 0) |
| libero_spatial | Sensor Noise | 351 / 351 | 0 | 94.6 [91.7, 96.5] | 96.3 | -1.7 (±3.1) | match |
| ALL | Light Conditions | 292 / 1142 | 0 | 100.0 [98.7, 100.0] | 88.7 | +11.3 (±1.8) | — (partial) |
| ALL | Sensor Noise | 351 / 1601 | 0 | 94.6 [91.7, 96.5] | 75.8 | +18.8 (±3.2) | — (partial) |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Light Conditions | libero_spatial | 100 (n=52) | 100 (n=104) | 100 (n=92) | 100 (n=38) | 100 (n=6) |
| Light Conditions | ALL | 100 (n=52) | 100 (n=104) | 100 (n=92) | 100 (n=38) | 100 (n=6) |
| Sensor Noise | libero_spatial | 98 (n=51) | 97 (n=136) | 98 (n=95) | 93 (n=57) | 33 (n=12) |
| Sensor Noise | ALL | 98 (n=51) | 97 (n=136) | 98 (n=95) | 93 (n=57) | 33 (n=12) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 56 | 89.3 [78.5, 95.0] |
| motion blur | 76 | 92.1 [83.8, 96.3] |
| gaussian blur | 82 | 93.9 [86.5, 97.4] |
| fog | 52 | 96.2 [87.0, 98.9] |
| glass blur | 85 | 100.0 [95.7, 100.0] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | 100 (1) | 100 (3) | 100 (7) | 100 (8) | 100 (10) | 100 (9) | 80 (10) | 89 (9) | 78 (9) | 90 (10) |
| gaussian blur | 100 (1) | 100 (7) | 100 (9) | 90 (10) | 90 (10) | 100 (9) | 90 (10) | 86 (7) | 90 (10) | 100 (9) |
| zoom blur | — | 100 (3) | 100 (2) | 75 (4) | 100 (3) | 80 (5) | 90 (10) | 90 (10) | 90 (10) | 89 (9) |
| fog | — | 100 (2) | 100 (1) | — | 100 (5) | 100 (6) | 100 (9) | 90 (10) | 100 (9) | 90 (10) |
| glass blur | 100 (5) | 100 (1) | 100 (10) | 100 (9) | 100 (10) | 100 (10) | 100 (10) | 100 (10) | 100 (10) | 100 (10) |

## lang_mode = raw  (60 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_spatial | Light Conditions | 30 / 292 | 0 | 100.0 [88.6, 100.0] | 98.3 | +1.7 (±1.5) | — (partial) |
| libero_spatial | Sensor Noise | 30 / 351 | 0 | 90.0 [74.4, 96.5] | 96.3 | -6.3 (±10.9) | — (partial) |
| ALL | Light Conditions | 30 / 1142 | 0 | 100.0 [88.6, 100.0] | 88.7 | +11.3 (±1.8) | — (partial) |
| ALL | Sensor Noise | 30 / 1601 | 0 | 90.0 [74.4, 96.5] | 75.8 | +14.2 (±10.9) | — (partial) |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Light Conditions | libero_spatial | 100 (n=5) | 100 (n=11) | 100 (n=9) | 100 (n=4) | 100 (n=1) |
| Light Conditions | ALL | 100 (n=5) | 100 (n=11) | 100 (n=9) | 100 (n=4) | 100 (n=1) |
| Sensor Noise | libero_spatial | 100 (n=4) | 92 (n=12) | 100 (n=8) | 80 (n=5) | 0 (n=1) |
| Sensor Noise | ALL | 100 (n=4) | 92 (n=12) | 100 (n=8) | 80 (n=5) | 0 (n=1) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 2 | 50.0 [9.5, 90.5] |
| glass blur | 6 | 83.3 [43.6, 97.0] |
| motion blur | 10 | 90.0 [59.6, 98.2] |
| gaussian blur | 7 | 100.0 [64.6, 100.0] |
| fog | 5 | 100.0 [56.6, 100.0] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | — | 100 (2) | 100 (1) | 100 (2) | 100 (1) | — | — | 0 (1) | 100 (2) | 100 (1) |
| gaussian blur | — | — | 100 (1) | 100 (1) | 100 (1) | 100 (1) | 100 (2) | — | 100 (1) | — |
| zoom blur | — | — | — | — | — | 0 (1) | — | 100 (1) | — | — |
| fog | — | — | — | — | — | — | 100 (2) | 100 (1) | 100 (1) | 100 (1) |
| glass blur | 100 (1) | — | 100 (1) | — | 100 (1) | — | — | 100 (1) | 100 (1) | 0 (1) |

## Paired instruction ablation (clean vs raw, same tasks; errors excluded)

| Suite | Category | pairs | clean succ | raw succ | clean-only | raw-only |
|---|---|---|---|---|---|---|
| libero_spatial | Light Conditions | 30 | 30 | 30 | 0 | 0 |
| libero_spatial | Sensor Noise | 30 | 27 | 27 | 1 | 1 |
