# Results

One CSV row per episode. Columns: `success` (0/1), `error` (1 = harness exception; excluded from rates, never
scored as a policy failure), `policy_steps`, `n_queries`, `wall_s`.

| Dir | Experiment | Report | Key files |
|---|---|---|---|
| `e002/` | LIBERO-Plus Noise + Light, OpenVLA-OFT 4-bit, clean instructions, 1 episode/task | §2, §3.1, §3.3 | `{spatial,object,goal,long}_clean.csv`, `final_summary.md`; `*_raw_ablation.csv` = 30-task suffix pilot; `spatial_anchor_metrics.json` = unperturbed anchor; `stage0_smoke.csv` = smoke test |
| `e003/` | All 379 Goal Noise tasks with the **suffixed** (raw) instruction | §3.2 | `goal_raw_full_{a,b}.csv` (two halves, task ids in `ids_{a,b}.txt`), `summary.md` |
| `e004/` | Toy-ISP probe, 3 tasks × 50 episodes, OFT and OFT_w | §4 | `oft.csv`, `oft_w.csv`, `conditions.json`, `denoise_calibration.json`, `final_summary.md`, `frames.npz` (LIBERO renders for tests/calibration), `visual_check.png` |
| `e005/` | Infinite-ISP replication (+ toy arm), 9 tasks × 20 episodes, λ = 0.012 | §4 | `oft.csv`, `oft_w.csv`, `conditions.json`, `calibration.json`, `final_summary.md` |
| `e006/` | OFT confirmation on fresh init states 20–49, 9 tasks × 30 episodes | §4 | `oft.csv`, `conditions.json`, `summary.md` |

Kept for transparency, excluded from all reported numbers:
- `e002/long_clean_BUGGY_scene_prefix.csv` holds 8 Long-suite rows run with a mis-cleaned instruction. The
  instruction kept a scene prefix, and the bug was fixed by reusing LIBERO's own parser.
- `e004/interim_summary.md` and `e005/primary_summary.md` are snapshots written before the runs completed.

The first-frame sample images (`samples/*.npz`, about 40 MB) are not included. Re-running a condition
regenerates them.
