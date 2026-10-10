
## oft

| condition | n | success % [95% CI] | reference | ref % | drop | CI separate | suites ≥10 drop | qualifies |
|---|---|---|---|---|---|---|---|---|
| inf_clean_off | 180 | 99.4 [96.9, 99.9] | — | — | — | — | — | — |
| inf_clean_repo | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| inf_clean_sharp | 180 | 97.8 [94.4, 99.1] | — | — | — | — | — | — |
| render | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| toy_clean | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| inf_lam1.2e-2_off | 180 | 86.1 [80.3, 90.4] | inf_clean_off | 99.4 | +13.3 | True | 2/3 (+0, +30, +10) | **yes** |
| inf_lam1.2e-2_repo | 180 | 91.7 [86.7, 94.9] | inf_clean_sharp | 97.8 | +6.1 | False | 1/3 (+2, +15, +2) | no |
| inf_lam1.2e-2_strong | 180 | 95.0 [90.8, 97.3] | inf_clean_sharp | 97.8 | +2.8 | False | 0/3 (+0, +8, +0) | no |
| inf_lam1.2e-2_tuned | 180 | 96.7 [92.9, 98.5] | inf_clean_sharp | 97.8 | +1.1 | False | 0/3 (+0, -2, +5) | no |
| toy_lam1.2e-2_default | 180 | 98.3 [95.2, 99.4] | toy_clean | 98.9 | +0.6 | False | 0/3 (+0, +0, +2) | no |
| toy_lam1.2e-2_dnstrong | 41 | 90.2 [77.5, 96.1] | toy_clean | 98.9 | +8.6 | False | 0/3 (+10, +nan, +nan) | no (partial) |

## oft_w

| condition | n | success % [95% CI] | reference | ref % | drop | CI separate | suites ≥10 drop | qualifies |
|---|---|---|---|---|---|---|---|---|
| inf_clean_off | 180 | 97.2 [93.7, 98.8] | — | — | — | — | — | — |
| inf_clean_repo | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| inf_clean_sharp | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| render | 180 | 98.3 [95.2, 99.4] | — | — | — | — | — | — |
| toy_clean | 180 | 98.9 [96.0, 99.7] | — | — | — | — | — | — |
| inf_lam1.2e-2_off | 180 | 81.1 [74.8, 86.2] | inf_clean_off | 97.2 | +16.1 | True | 2/3 (+33, +2, +13) | **yes** |
| inf_lam1.2e-2_repo | 180 | 82.8 [76.6, 87.6] | inf_clean_sharp | 98.9 | +16.1 | True | 2/3 (+25, +7, +17) | **yes** |
| inf_lam1.2e-2_strong | 180 | 90.6 [85.4, 94.0] | inf_clean_sharp | 98.9 | +8.3 | True | 1/3 (+18, +3, +3) | no |
| inf_lam1.2e-2_tuned | 180 | 98.3 [95.2, 99.4] | inf_clean_sharp | 98.9 | +0.6 | False | 0/3 (+2, +0, +0) | no |
| toy_lam1.2e-2_default | 180 | 96.7 [92.9, 98.5] | toy_clean | 98.9 | +2.2 | False | 0/3 (+8, -2, +0) | no |
| toy_lam1.2e-2_dnoff | 180 | 85.6 [79.7, 89.9] | toy_clean | 98.9 | +13.3 | True | 1/3 (+27, +5, +8) | no |
| toy_lam1.2e-2_dnstrong | 180 | 65.6 [58.4, 72.1] | toy_clean | 98.9 | +33.3 | True | 3/3 (+57, +30, +13) | **yes** |

**R1 (primary, Infinite-ISP): REPLICATES** — qualifying: {'oft': ['inf_lam1.2e-2_off'], 'oft_w': ['inf_lam1.2e-2_off', 'inf_lam1.2e-2_repo']}
Secondary (toy ISP on 9 tasks): does not replicate (or incomplete) — {'oft': [], 'oft_w': ['toy_lam1.2e-2_dnstrong']}
