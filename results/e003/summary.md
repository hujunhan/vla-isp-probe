
## lang_mode = clean  (658 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_goal | Light Conditions | 279 / 279 | 0 | 93.9 [90.5, 96.2] | 93.9 | +0.0 (±4.0) | match |
| libero_goal | Sensor Noise | 379 / 379 | 0 | 87.1 [83.3, 90.1] | 75.2 | +11.9 (±5.5) | diverge |
| ALL | Light Conditions | 279 / 1142 | 0 | 93.9 [90.5, 96.2] | 88.7 | +5.2 (±3.4) | — (partial) |
| ALL | Sensor Noise | 379 / 1601 | 0 | 87.1 [83.3, 90.1] | 75.8 | +11.3 (±4.0) | — (partial) |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Light Conditions | libero_goal | 89 (n=46) | 96 (n=49) | 100 (n=28) | 90 (n=21) | 86 (n=14) |
| Light Conditions | ALL | 89 (n=46) | 96 (n=49) | 100 (n=28) | 90 (n=21) | 86 (n=14) |
| Sensor Noise | libero_goal | 95 (n=59) | 95 (n=110) | 92 (n=53) | 88 (n=98) | 59 (n=59) |
| Sensor Noise | ALL | 95 (n=59) | 95 (n=110) | 92 (n=53) | 88 (n=98) | 59 (n=59) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 67 | 82.1 [71.3, 89.4] |
| motion blur | 80 | 85.0 [75.6, 91.2] |
| fog | 60 | 86.7 [75.8, 93.1] |
| gaussian blur | 82 | 89.0 [80.4, 94.1] |
| glass blur | 90 | 91.1 [83.4, 95.4] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | 100 (6) | 75 (4) | 100 (6) | 100 (8) | 100 (10) | 89 (9) | 80 (10) | 80 (10) | 78 (9) | 50 (8) |
| gaussian blur | 100 (6) | 100 (4) | 100 (9) | 100 (10) | 100 (9) | 100 (8) | 89 (9) | 75 (8) | 67 (9) | 70 (10) |
| zoom blur | 100 (4) | 100 (4) | 80 (5) | 75 (4) | 88 (8) | 86 (7) | 80 (10) | 70 (10) | 100 (7) | 62 (8) |
| fog | 33 (3) | 100 (4) | 100 (4) | 100 (3) | 100 (5) | 88 (8) | 90 (10) | 100 (9) | 86 (7) | 57 (7) |
| glass blur | 100 (7) | 100 (5) | 100 (10) | 100 (10) | 90 (10) | 89 (9) | 78 (9) | 80 (10) | 80 (10) | 100 (10) |

## lang_mode = raw  (379 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_goal | Sensor Noise | 379 / 379 | 0 | 73.9 [69.2, 78.0] | 75.2 | -1.3 (±6.2) | match |
| ALL | Sensor Noise | 379 / 1601 | 0 | 73.9 [69.2, 78.0] | 75.8 | -1.9 (±4.9) | — (partial) |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Sensor Noise | libero_goal | 92 (n=59) | 83 (n=110) | 75 (n=53) | 72 (n=98) | 41 (n=59) |
| Sensor Noise | ALL | 92 (n=59) | 83 (n=110) | 75 (n=53) | 72 (n=98) | 41 (n=59) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 67 | 62.7 [50.7, 73.3] |
| gaussian blur | 82 | 69.5 [58.9, 78.4] |
| glass blur | 90 | 75.6 [65.8, 83.3] |
| motion blur | 80 | 76.2 [65.9, 84.2] |
| fog | 60 | 86.7 [75.8, 93.1] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | 100 (6) | 100 (4) | 100 (6) | 100 (8) | 90 (10) | 78 (9) | 60 (10) | 60 (10) | 56 (9) | 50 (8) |
| gaussian blur | 83 (6) | 100 (4) | 89 (9) | 70 (10) | 67 (9) | 62 (8) | 67 (9) | 62 (8) | 56 (9) | 60 (10) |
| zoom blur | 75 (4) | 100 (4) | 80 (5) | 100 (4) | 75 (8) | 86 (7) | 50 (10) | 40 (10) | 43 (7) | 38 (8) |
| fog | 100 (3) | 100 (4) | 100 (4) | 100 (3) | 80 (5) | 100 (8) | 100 (10) | 78 (9) | 71 (7) | 57 (7) |
| glass blur | 100 (7) | 80 (5) | 90 (10) | 90 (10) | 80 (10) | 67 (9) | 67 (9) | 60 (10) | 70 (10) | 60 (10) |

## Paired instruction ablation (clean vs raw, same tasks; errors excluded)

| Suite | Category | pairs | clean succ | raw succ | clean-only | raw-only | McNemar p |
|---|---|---|---|---|---|---|---|
| libero_goal | Sensor Noise | 379 | 330 | 280 | 74 | 24 | 4.22e-07 |
