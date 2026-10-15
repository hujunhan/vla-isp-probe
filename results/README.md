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
| `e007/` | C1 gate: image motion per control step in OFT rollouts (classic render), 9 tasks × 5 | C1 §2.1 | `frames.csv` (per-frame motion, from `e007/motion.py`), `summary.json`, `oft.csv`, `eef_overlay_goal1.png` |
| `e008/` | Filament renderer matching: classic vs Filament as shipped / matched HDR / matched RAW + ISP, 3 tasks × 20 | C1 §2.1 | `oft.csv`, `conditions.json`, `exposure/*.json` (per-task brightness match) |
| `e009/` | C1 exposure sweep and decomposition (Stages A–C), OFT and OFT_w, 9 tasks × 20, init states 20–39 | C1 §3 | `oft.csv`, `oft_w.csv`, `conditions.json`, `nr_calibration.json`, `preflight_c.json`, `c1_frames.npz` (Figure 3a) |
| `e009_s1/`, `e009_s2/` | C1 Stage D: noise seeds 1 and 2 on Goal 0 and Object 0 | C1 §3.4 | `oft.csv`, `conditions.json` |
| `e009_tune/` | Clean run on init state 45 that dumps sub-frames for NR tuning | C1 §2.3 | `oft.csv`, `conditions.json` |
| `e010/` | Open-loop probe along clean rollouts, both policies, 9 tasks × 5 | C1 §3.3 | `oft.csv`, `oft_w.csv`, `probe/*.npz` (clean and 8 noisy action chunks per query), `summary.json` |

Kept for transparency, excluded from all reported numbers:
- `e002/long_clean_BUGGY_scene_prefix.csv` holds 8 Long-suite rows run with a mis-cleaned instruction. The
  instruction kept a scene prefix, and the bug was fixed by reusing LIBERO's own parser.
- `e004/interim_summary.md` and `e005/primary_summary.md` are snapshots written before the runs completed.

The first-frame sample images (`samples/*.npz`, about 40 MB) are not included. Re-running a condition
regenerates them.
