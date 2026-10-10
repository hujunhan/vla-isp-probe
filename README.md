# vla-isp-probe

**Does the camera's RAW processing (ISP) matter for a vision-language-action robot policy?**
A reproduction of the LIBERO-Plus robustness numbers for OpenVLA-OFT, plus a pre-registered probe
of simulated sensor noise and ISP noise reduction in closed-loop LIBERO manipulation. Everything was
run on one 16 GB laptop GPU.

![Figure 1: ISP noise reduction and closed-loop success](figures/fig1_isp.png)

Full write-up: **[REPORT.md](REPORT.md)**. Every number in it comes from a per-episode CSV in
[`results/`](results/), and you can recompute it without a GPU (see "Reproduce" below).

## Findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **LIBERO-Plus Table 2 reproduces with a 4-bit model:** Sensor Noise 75.7% vs 75.8% in the paper, Light 91.3% vs 88.7% (2,743 tasks) | REPORT §2, `results/e002/` |
| 2 | **LIBERO-Plus "sensor noise" corrupts only the third-person camera.** The wrist image stays clean | REPORT §3.1 |
| 3 | **LIBERO-Plus task instructions carry a file-name suffix** ("… view 0 0 100 0 0 initstate 0 noise 3"). It costs 13 points on Goal (87.1 → 73.9, McNemar p = 4e-7), about half of that suite's reported noise drop | REPORT §3.2, `results/e003/` |
| 4 | **LIBERO-Plus difficulty labels contradict Table 2.** 19.9% of Light tasks are labelled "solved by none", which caps OFT at 80.1%, yet Table 2 reports 88.7% | REPORT §3.3 |
| 5 | **Realistic sensor noise plus a noise reduction tuned to the gain: no measurable effect** (≤ 3.3 points, two ISPs, two policies) | REPORT §4, `results/e004/`, `results/e005/` |
| 6 | **Un-tuned noise reduction at high gain hurts:** a drop of about 16 points for a third-person-only policy across suites, and about 8 for a wrist + third-person policy (the estimate shrank on replication: 29 → 13 → 8) | REPORT §4.2–4.3, `results/e005/`, `results/e006/` |
| 7 | **Image-space Gaussian noise at matched PSNR has no effect,** so image-space robustness benchmarks would miss finding 6 | REPORT §4.2 |

![Figure 2: instruction suffix](figures/fig2_suffix.png)

**Scope:** simulation only, one ISP setting (spatial noise-reduction strength), RAW approximated from
8-bit renders, 4-bit models, 9 tasks for the ISP probe. REPORT §5 lists the limitations.

## Reproduce

There are three tiers, from minutes on a CPU to days on a GPU.

### Tier 0: recompute every table from the CSVs (CPU, about a minute)

```bash
pip install numpy opencv-python pytest matplotlib  # aggregators need only the stdlib
python scripts/aggregate_e002.py results/e002/{spatial,object,goal,long}_clean.csv   # §2, §3.1, §3.3
python scripts/aggregate_e002.py results/e002/goal_clean.csv results/e003/goal_raw_full_{a,b}.csv  # §3.2 suffix, McNemar
python scripts/aggregate_e004.py results/e004/oft.csv results/e004/oft_w.csv           # §4.2 toy ISP
python scripts/aggregate_e005.py results/e005/oft.csv results/e005/oft_w.csv           # §4.2 Infinite-ISP
EXP=e006 python scripts/aggregate_e005.py results/e006/oft.csv                         # §4.2 confirmation
python -m pytest -q tests/                                                              # camera model (8 tests)
```

### Tier 1: the camera model on its own (CPU)

`isp/camera.py` (sensor model + toy ISP) and `isp/infinite.py` (adapter to
[Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP)) map a `uint8` render to a `uint8`
camera image:

```python
import numpy as np
from isp.camera import CameraConfig, simulate, read_noise_for
from isp.infinite import preset, simulate_infinite      # needs INFINITE_ISP_ROOT (below)

frame = np.load("results/e004/frames.npz")["agentview"][0]          # a LIBERO render, 256x256x3
rng = np.random.default_rng(0)
toy = simulate(frame, CameraConfig(shot_noise=0.012, denoise_h=0), rng)       # toy ISP, NR off
inf = simulate_infinite(frame, 0.012, read_noise_for(0.012), preset("off"), rng)
```

`python scripts/make_figures.py` redraws both report figures into `figures/` (panel 1a uses Infinite-ISP).

`scripts/calibrate_e004.py` and `scripts/calibrate_e005.py` re-derive the noise-reduction strengths
(the "tuned" presets) from these frames.

### Tier 2: closed-loop runs (NVIDIA GPU, ≥ 8 GB with 4-bit)

**1. Clone the upstream code at the commits we used, as siblings:**

| Repo | Commit | Default location (override with env var) |
|---|---|---|
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | `e4287e9` | `~/research/openvla-oft` (`OPENVLA_OFT_ROOT`) |
| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc3` | `~/research/LIBERO-plus` (`LIBERO_PLUS_ROOT`) |
| [10x-Engineers/Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) | `3221207` | `~/research/Infinite-ISP` (`INFINITE_ISP_ROOT`) |

Logs go to `LOG_DIR` (default `~/research/logs`). See `scripts/env.sh`.

**2. Create the environment:** `bash scripts/setup_env.sh`
- It creates conda env `oft` with Python 3.10.14 and torch 2.2.0, then applies the pins that
  unpinned installs need (numpy 1.26.4, mujoco 3.2.6, accelerate 0.30.1, bitsandbytes 0.44.1, …).
  The comments in the script explain why each pin is there.
- LIBERO-Plus needs `libmagickwand-dev` and its assets: unzip `assets.zip` and move
  `…/LIBERO-plus-0/assets` to `LIBERO-plus/libero/libero/assets`.
- At run time, set `PYTHONPATH=$LIBERO_PLUS_ROOT MUJOCO_GL=egl`. The editable install does not resolve.

**3. Run.** Episode counts and summed wall-clock times are from our RTX 3080 Laptop runs. Model
loading is not included.

| Experiment | Command | Episodes | Time |
|---|---|---|---|
| E002, Table 2 reproduction | `bash scripts/e002_stage.sh libero_spatial spatial_clean`; repeat for `object`, `goal`, and `10` (→ `long_clean`) | 2,743 | ≈ 40 h |
| E003, instruction-suffix ablation | `LANG_MODE=raw TASK_IDS=$(cat results/e003/ids_a.txt) RESULTS_DIR=e003 bash scripts/e002_stage.sh libero_goal goal_raw_full_a`; repeat with `ids_b` | 379 | ≈ 6.5 h |
| E004, toy ISP probe | `python scripts/e004_run.py --exp e004 --policy oft --blocks B01,B2,B3,B4,B0c,B5 --tasks libero_spatial:0,libero_object:0,libero_goal:0 --episodes 50`; `--policy oft_w` ran blocks `B01,B2,B3,B4` | 5,700 | ≈ 43 h |
| E005, Infinite-ISP replication | `python scripts/e004_run.py --exp e005 --policy oft --blocks G,C,N --tasks libero_spatial:0,libero_spatial:1,libero_spatial:2,libero_object:0,libero_object:1,libero_object:2,libero_goal:0,libero_goal:1,libero_goal:2 --episodes 20`; repeat with `oft_w` | 4,320 | ≈ 41 h |
| E006, fresh-episode confirmation | the same `--tasks`, with `--exp e006 --policy oft --blocks A,B --episodes 30 --episode-offset 20` | 1,080 | ≈ 12 h |

Notes on running:
- Checkpoints download from Hugging Face on first use. `e004_run.py` then runs offline from the
  local cache.
- All runs resume: finished (condition, task) cells are skipped.
- To write fresh CSVs instead of appending, move the shipped ones out of `results/<exp>/` first.
- `results/<exp>/conditions.json` defines every condition.
- Camera noise is seeded per (task, episode). The simulator itself is deterministic per init state:
  re-running 30 tasks flipped 0 outcomes (REPORT §3.2). Expect different hardware or driver
  versions to shift individual episodes.

## Layout

```
REPORT.md                 the write-up
figures/                  report figures (scripts/make_figures.py)
isp/camera.py             sensor model (Brooks et al. unprocessing, shot/read noise, 12-bit) + toy ISP
isp/infinite.py           runs Infinite-ISP stages on our simulated RAW
scripts/run_oft_*.py      wrappers around the OpenVLA-OFT LIBERO eval (original tasks, LIBERO-Plus
                          categories, camera model on observations); errors counted, not scored as failures
scripts/e004_run.py       driver for the probe experiments (E004–E006)
scripts/aggregate_*.py    tables and pre-registered decision rules
scripts/calibrate_*.py    noise-reduction strength calibration
results/                  per-episode CSVs, condition definitions, summaries (see results/README.md)
tests/                    camera-model unit tests
```

Code comments that cite `decisions/NNN`, an `E00N record`, or `lit #N` point into the author's
private lab notebook, where each experiment's predictions and decision rules were written down
before it ran. The decision rules themselves are restated in REPORT.md and the aggregator
docstrings.

## Credits and licenses

- Upstream code is not vendored. It is cloned at the commits above under its own licenses.
- The exception is `scripts/libero_original_task_map.py`, the original LIBERO task list (MIT).
- Unprocessing follows Brooks et al., CVPR 2019 (arXiv 1811.11127) and
  [google-research/unprocessing](https://github.com/google-research/unprocessing) (Apache-2.0).

The code was drafted with an AI coding assistant under the author's direction. Commits tagged
`[ai-scaffold]` in the source history mark the drafted code. REPORT §8 lists what was adopted,
adapted, and authored.

License: [MIT](LICENSE) for the code in this repository.
