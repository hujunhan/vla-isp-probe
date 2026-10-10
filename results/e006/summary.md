
## oft

| condition | n | success % [95% CI] | reference | ref % | drop | CI separate | suites ≥10 drop | qualifies |
|---|---|---|---|---|---|---|---|---|
| inf_clean_off | 270 | 97.8 [95.2, 99.0] | — | — | — | — | — | — |
| inf_clean_sharp | 270 | 98.9 [96.8, 99.6] | — | — | — | — | — | — |
| inf_lam1.2e-2_off | 270 | 90.0 [85.8, 93.0] | inf_clean_off | 97.8 | +7.8 | True | 1/3 (-1, +16, +9) | no |
| inf_lam1.2e-2_repo | 270 | 93.0 [89.3, 95.4] | inf_clean_sharp | 98.9 | +5.9 | True | 1/3 (+0, +13, +4) | no |

**R1 (primary, Infinite-ISP): FAILS** — qualifying: {'oft': []}
Secondary (toy ISP on 9 tasks): does not replicate (or incomplete) — {'oft': []}
