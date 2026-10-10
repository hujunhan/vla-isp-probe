
## lang_mode = clean  (2743 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_10 | Light Conditions | 274 / 274 | 0 | 89.4 [85.2, 92.5] | 89.4 | +0.0 (±5.2) | match |
| libero_10 | Sensor Noise | 449 / 449 | 0 | 63.3 [58.7, 67.6] | 63.5 | -0.2 (±6.3) | match |
| libero_goal | Light Conditions | 279 / 279 | 0 | 93.9 [90.5, 96.2] | 93.9 | +0.0 (±4.0) | match |
| libero_goal | Sensor Noise | 379 / 379 | 0 | 87.1 [83.3, 90.1] | 75.2 | +11.9 (±5.5) | diverge |
| libero_object | Light Conditions | 297 / 297 | 0 | 82.2 [77.4, 86.1] | 73.7 | +8.5 (±6.6) | ambiguous |
| libero_object | Sensor Noise | 422 / 422 | 0 | 63.0 [58.3, 67.5] | 72.3 | -9.3 (±6.3) | ambiguous |
| libero_spatial | Light Conditions | 292 / 292 | 0 | 100.0 [98.7, 100.0] | 98.3 | +1.7 (±1.5) | ambiguous (ceiling: our variance term is 0) |
| libero_spatial | Sensor Noise | 351 / 351 | 0 | 94.6 [91.7, 96.5] | 96.3 | -1.7 (±3.1) | match |
| ALL | Light Conditions | 1142 / 1142 | 0 | 91.3 [89.6, 92.8] | 88.7 | +2.6 (±2.5) | ambiguous |
| ALL | Sensor Noise | 1601 / 1601 | 0 | 75.7 [73.5, 77.7] | 75.8 | -0.1 (±3.0) | match |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Light Conditions | libero_10 | 100 (n=11) | 97 (n=35) | 85 (n=53) | 97 (n=73) | 82 (n=102) |
| Light Conditions | libero_goal | 89 (n=46) | 96 (n=49) | 100 (n=28) | 90 (n=21) | 86 (n=14) |
| Light Conditions | libero_object | 100 (n=2) | 100 (n=9) | 94 (n=71) | 89 (n=110) | 65 (n=105) |
| Light Conditions | libero_spatial | 100 (n=52) | 100 (n=104) | 100 (n=92) | 100 (n=38) | 100 (n=6) |
| Light Conditions | ALL | 95 (n=111) | 98 (n=197) | 95 (n=244) | 93 (n=242) | 75 (n=227) |
| Sensor Noise | libero_10 | 85 (n=27) | 91 (n=97) | 84 (n=81) | 57 (n=124) | 28 (n=120) |
| Sensor Noise | libero_goal | 95 (n=59) | 95 (n=110) | 92 (n=53) | 88 (n=98) | 59 (n=59) |
| Sensor Noise | libero_object | 85 (n=40) | 91 (n=157) | 69 (n=71) | 35 (n=104) | 8 (n=50) |
| Sensor Noise | libero_spatial | 98 (n=51) | 97 (n=136) | 98 (n=95) | 93 (n=57) | 33 (n=12) |
| Sensor Noise | ALL | 92 (n=177) | 93 (n=500) | 86 (n=300) | 64 (n=383) | 32 (n=241) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 288 | 60.8 [55.0, 66.2] |
| gaussian blur | 341 | 70.1 [65.0, 74.7] |
| motion blur | 336 | 81.0 [76.4, 84.8] |
| glass blur | 364 | 81.9 [77.6, 85.5] |
| fog | 272 | 83.8 [79.0, 87.7] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | 96 (23) | 88 (24) | 97 (31) | 97 (36) | 98 (40) | 86 (37) | 74 (38) | 72 (39) | 63 (35) | 45 (33) |
| gaussian blur | 94 (18) | 93 (30) | 100 (38) | 85 (39) | 72 (39) | 68 (34) | 55 (38) | 50 (30) | 46 (39) | 50 (36) |
| zoom blur | 92 (13) | 81 (16) | 89 (19) | 71 (24) | 80 (30) | 53 (32) | 52 (40) | 50 (40) | 51 (37) | 41 (37) |
| fog | 73 (11) | 94 (16) | 94 (18) | 100 (18) | 97 (30) | 97 (34) | 97 (39) | 92 (38) | 62 (34) | 41 (34) |
| glass blur | 100 (29) | 91 (23) | 97 (39) | 97 (39) | 85 (40) | 86 (36) | 64 (39) | 67 (39) | 72 (40) | 68 (40) |

## lang_mode = raw  (120 episodes)

| Suite | Category | done / bench n | errors | ours % [95% CI] | paper % | diff | verdict |
|---|---|---|---|---|---|---|---|
| libero_goal | Light Conditions | 30 / 279 | 0 | 100.0 [88.6, 100.0] | 93.9 | +6.1 (±2.8) | — (partial) |
| libero_goal | Sensor Noise | 30 / 379 | 0 | 73.3 [55.6, 85.8] | 75.2 | -1.9 (±16.4) | — (partial) |
| libero_spatial | Light Conditions | 30 / 292 | 0 | 100.0 [88.6, 100.0] | 98.3 | +1.7 (±1.5) | — (partial) |
| libero_spatial | Sensor Noise | 30 / 351 | 0 | 90.0 [74.4, 96.5] | 96.3 | -6.3 (±10.9) | — (partial) |
| ALL | Light Conditions | 60 / 1142 | 0 | 100.0 [94.0, 100.0] | 88.7 | +11.3 (±1.8) | — (partial) |
| ALL | Sensor Noise | 60 / 1601 | 0 | 81.7 [70.1, 89.4] | 75.8 | +5.9 (±10.0) | — (partial) |

Outcome-defined levels (not severity). Paper OFT: 100% on L1, 0% on L5.

| Category | Suite | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Light Conditions | libero_goal | 100 (n=5) | 100 (n=5) | 100 (n=3) | 100 (n=2) | 100 (n=2) |
| Light Conditions | libero_spatial | 100 (n=5) | 100 (n=11) | 100 (n=9) | 100 (n=4) | 100 (n=1) |
| Light Conditions | ALL | 100 (n=10) | 100 (n=16) | 100 (n=12) | 100 (n=6) | 100 (n=3) |
| Sensor Noise | libero_goal | 100 (n=5) | 75 (n=8) | 100 (n=4) | 75 (n=8) | 20 (n=5) |
| Sensor Noise | libero_spatial | 100 (n=4) | 92 (n=12) | 100 (n=8) | 80 (n=5) | 0 (n=1) |
| Sensor Noise | ALL | 100 (n=9) | 85 (n=20) | 100 (n=12) | 77 (n=13) | 17 (n=6) |

| Noise family | n | success % [95% CI] |
|---|---|---|
| zoom blur | 8 | 50.0 [21.5, 78.5] |
| glass blur | 13 | 76.9 [49.7, 91.8] |
| motion blur | 17 | 82.4 [59.0, 93.8] |
| gaussian blur | 12 | 91.7 [64.6, 98.5] |
| fog | 10 | 100.0 [72.2, 100.0] |

| Noise family | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|---|---|---|---|---|
| motion blur | 100 (2) | 100 (2) | 100 (1) | 100 (2) | 100 (2) | 100 (1) | — | 0 (3) | 100 (3) | 100 (1) |
| gaussian blur | — | — | 100 (2) | 100 (2) | 100 (1) | 100 (1) | 100 (2) | 50 (2) | 100 (2) | — |
| zoom blur | — | — | 100 (1) | — | 0 (1) | 67 (3) | 0 (1) | 100 (1) | — | 0 (1) |
| fog | 100 (1) | — | — | — | — | — | 100 (3) | 100 (3) | 100 (2) | 100 (1) |
| glass blur | 100 (1) | — | 100 (3) | 100 (1) | 100 (1) | 0 (1) | — | 67 (3) | 100 (1) | 50 (2) |

## Paired instruction ablation (clean vs raw, same tasks; errors excluded)

| Suite | Category | pairs | clean succ | raw succ | clean-only | raw-only |
|---|---|---|---|---|---|---|
| libero_goal | Light Conditions | 30 | 29 | 30 | 0 | 1 |
| libero_goal | Sensor Noise | 30 | 28 | 22 | 8 | 2 |
| libero_spatial | Light Conditions | 30 | 30 | 30 | 0 | 0 |
| libero_spatial | Sensor Noise | 30 | 27 | 27 | 1 | 1 |
