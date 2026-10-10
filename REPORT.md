# Camera ISP tuning and closed-loop VLA manipulation: a reproduction and a probe

> **Status: draft for author review (2026-10-09). Not yet public.**
> Before publishing:
> - [ ] Author reviews every number against `results/` and the cited sources.
> - [ ] Author explain-back of `isp/camera.py`, `isp/infinite.py`, and the `aggregate_*.py` metric
>   scripts.
> - [x] Choose a license (MIT).
> - [ ] Remove this box.

## Summary

We ran OpenVLA-OFT, a 7B vision-language-action policy, on the LIBERO simulation benchmark on a single
16 GB laptop GPU (4-bit quantized). We asked two questions.

**1. Can we reproduce the LIBERO-Plus robustness numbers?** Yes, for the Sensor Noise and Light
columns of LIBERO-Plus Table 2:

| Column | Ours | Paper |
|---|---|---|
| Noise | 75.7% | 75.8% |
| Light | 91.3% | 88.7% |

Along the way we found three properties of the benchmark that matter for interpreting it:
- its "sensor noise" corrupts only the third-person camera;
- its default task instructions carry a file-name suffix that measurably lowers success on one
  suite;
- its difficulty labels are arithmetically inconsistent with its own headline table.

**2. Does simulated camera RAW processing (the ISP) change closed-loop success?** We tested a narrow
slice of the ISP: spatial noise-reduction strength at high sensor gain.
- Realistic sensor noise with a noise reduction tuned to the gain has **no measurable effect**.
- Un-tuned noise reduction (off, or a fixed shipped default) costs **≈ 16 points** for a
  third-person-only policy across suites. For a policy that also has a wrist camera, it costs
  **≈ 8 points**, mostly on texture-dependent tasks.
- Image-space noise at the same PSNR has no effect, so image-space robustness benchmarks would not
  detect this.
- The effect for the wrist-camera policy shrank on replication (−29 → −13 → −8 points). We report
  that decline rather than the first estimate.

**Scope.** One ISP knob, one gain level, simulated RAW derived from 8-bit renders, 4-bit models, 9
LIBERO tasks. Nothing here is a claim about physical robots or real cameras.

---

## 1. Setup

| | |
|---|---|
| Hardware | One laptop, RTX 3080 Laptop 16 GB, 31 GB RAM, Ubuntu 24.04 |
| Policy | OpenVLA-OFT per-suite checkpoints `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` (Kim et al., arXiv 2502.19645). Third-person-only variant ("OFT_w"): `Sylvest/openvla-7b-oft-finetuned-libero-without-wrist` (per-suite subfolders) |
| Precision | **4-bit (bitsandbytes NF4).** bf16 does not fit next to the EGL renderer on 16 GB: weights load at 14.7 GiB and the first forward pass runs out of memory |
| Benchmark | LIBERO-Plus @ `4976dc3` (Fei et al., arXiv 2510.13626), OpenVLA-OFT eval harness @ `e4287e9` |
| Environment | Python 3.10.14, torch 2.2.0, transformers (OFT fork), mujoco 3.2.6, robosuite 1.4.1. Full pins: `scripts/setup_env.sh` |
| Pre-registration | Every experiment's predictions and decision rules were committed before it ran (private lab notebook, E001–E006). Prediction outcomes are listed in §6 |

Feasibility: 4-bit OFT peaks at about 6.5 GB, runs about 0.47 s per model query, and takes about
7.5 s per short successful episode. One easy task gave 3/3 successes.

## 2. Reproduction of LIBERO-Plus Table 2 (OpenVLA-OFT, Noise and Light)

**Protocol:** all 2,743 Noise + Light tasks, one episode each, as in LIBERO-Plus, using the four
per-suite checkpoints. Success is pooled over tasks. The paper's averages are task-count-weighted:
pooling its own Table 10 suite values reproduces its 75.78 and 88.69, while plain suite means do not.

**Match rule (fixed before running):**
- **match:** \|diff\| ≤ 1.96 SE, where the SE of the difference combines both binomial estimates
  over the same n;
- **diverge:** \|diff\| > 10 points;
- **ambiguous:** anything else.

| Suite | Noise ours [95% CI] | Paper | Verdict | Light ours [95% CI] | Paper | Verdict |
|---|---|---|---|---|---|---|
| Spatial | 94.6 [91.7, 96.5] | 96.3 | match | 100.0 [98.7, 100] | 98.3 | ambiguous (ceiling) |
| Object | 63.0 [58.3, 67.5] | 72.3 | ambiguous | 82.2 [77.4, 86.1] | 73.7 | ambiguous |
| Goal | 87.1 [83.3, 90.1] | 75.2 | **diverge** | 93.9 [90.5, 96.2] | 93.9 | match |
| Long | 63.3 [58.7, 67.6] | 63.5 | match | 89.4 [85.2, 92.5] | 89.4 | match |
| **Pooled (Table 2)** | **75.7** [73.5, 77.7] | **75.8** | **match** | **91.3** [89.6, 92.8] | **88.7** | ambiguous (+2.6 vs ±2.5) |

Sources: paper values from arXiv 2510.13626 Table 10 (p31, per suite) and Table 2 (p9, pooled). Our
data: `results/e002/`.

**Notes:**
- An unperturbed anchor (Spatial, 10 tasks × 10 episodes) gave 98/100, against 97.6 in the OFT
  paper (Table I, p6).
- **Target confusion to avoid.** The 97.1 "Original" figure sometimes quoted alongside 75.8/88.7
  comes from LIBERO-Plus **Table 1** (p3), whose Noise and Light values (76.7/85.8) differ from
  Table 2's.
- **Deviations from the paper's setup:**
  - 4-bit instead of bf16.
  - RTX 3080 instead of an A100.
  - The checkpoint the paper used for its "OpenVLA-OFT" row is not stated exactly. Its Table 6
    (p16) lists only `huggingface.co/moojink`. We used the per-suite checkpoints.
- The Object cells miss in opposite directions, and we did not resolve them. A bf16 run would
  separate the quantization effect from other causes.

## 3. Findings about the benchmark

**3.1 "Sensor noise" corrupts only the third-person camera.**
- In `libero/libero/envs/env_wrapper.py` (`ControlEnv.step`/`reset`), the perturbation is applied
  to `agentview_image` only. The wrist image is untouched.
- Noise codes: 1–10 motion blur, 11–20 Gaussian blur, 21–30 zoom blur, 31–40 fog, 41–50 glass blur.
- So policies with a wrist camera get a clean fallback view. This is consistent with the paper's
  own Table 10, where OFT without the wrist camera scores 49.9 on Noise vs 75.8 with it.
- Success falls with physical severity for every family, with different shapes. Blurs degrade
  gradually (e.g. motion blur ≈ 96% → 45% from severity 1 to 10, pooled). Fog holds until severity
  8, then drops to 41–62%. (`results/e002/final_summary.md`)

**3.2 The default task instruction includes a file-name suffix, and it costs 13 points on Goal.**
- LIBERO-Plus derives `task.language` from the task file name. Perturbed tasks therefore receive
  instructions like "…place it on the plate **view 0 0 100 0 0 initstate 0 noise 3**" or
  "…**light 1**" (`libero/libero/benchmark/__init__.py`, `grab_language_from_filename`).
- The OFT harness passes this text to the policy unchanged.
- We ran all 379 Goal Noise tasks both ways:

  | Instruction | Goal Noise success |
  |---|---|
  | Clean (suffix stripped, verified equal to the original instruction) | 87.1% |
  | **Suffixed** | **73.9%** |
  | Paper | 75.2 |

  - Paired over the same tasks: 74 clean-only vs 24 suffixed-only successes, exact McNemar
    p = 4 × 10⁻⁷.
  - With the suffixed instruction, our number matches the paper's Goal Noise cell. With the clean
    one, it diverges by +11.9. This is strong circumstantial evidence the published cell was
    measured with the suffix. We have not confirmed the authors' code.
  - The short Light suffix had no effect (Goal Light 29/30 vs 30/30 on 30 pairs). The Spatial suite
    showed no effect either (27/30 vs 27/30).
- So on Goal, about half of the paper's 25-point "sensor noise" drop is language corruption, not
  image degradation. (`results/e003/`)
- Our pipeline is deterministic per task: re-running 30 tasks gave 0 of 30 outcome flips. The
  74/24 split therefore reflects the instruction, not run-to-run noise.

**3.3 The difficulty labels contradict Table 2.**
- App. C.3 (p18) defines level L5 as "solved by none" of four models, OpenVLA-OFT included.
- In `task_classification.json`, 227 of 1,142 Light tasks (19.9%) are L5. That caps OpenVLA-OFT's
  Light success at 80.1%, yet Table 2 reports 88.7.
- Our per-task agreement at the extremes is also poor:

  | Level | Ours, Noise | Ours, Light | Paper's OFT (by definition) |
  |---|---|---|---|
  | L1 | 92% | 95% | 100% |
  | L5 | 32% | 75% | 0% |

- The levels therefore cannot come from the same OpenVLA-OFT run as Table 2. They are not a
  severity scale. For a severity axis, use the noise code within each family.

**3.4 Harness pitfalls (OpenVLA-OFT `run_libero_eval.py`).**
- `run_episode` catches every exception and scores the episode as a **policy failure**. Out-of-memory
  errors showed up as "0% success".
- Our wrappers count these errors separately and abort after 3 in a row.
- Unpinned installs broke in five places (numpy 2.2, tensorflow-metadata 1.21 / protobuf, wandb 0.28,
  mujoco 3.14 joint-type assert in robosuite 1.4.1, accelerate 1.15 with bitsandbytes). The working pins are in
  `scripts/setup_env.sh`.

## 4. Probe: does simulated RAW processing change closed-loop success?

### 4.1 Method

```
MuJoCo render (8-bit)
  → unprocess (Brooks et al. 2019): inverse tone → inverse gamma → inverse CCM → inverse WB/gain → RGGB mosaic
  → sensor: shot + read noise (variance = x·λ_shot + λ_read), 12-bit quantisation
  → ISP → policy
```

**Where the RAW comes from.**
- Unprocessing follows Brooks et al., "Unprocessing Images for Learned Raw Denoising" (CVPR 2019,
  arXiv 1811.11127), §3–4.
- Camera parameters are fixed at the centre of Brooks' sampling ranges:
  - mean of their 4 colour matrices;
  - red gain 2.15, blue gain 1.7 (§3.4, p4);
  - brightness gain 1/0.8.
- **Plausible noise:** λ_shot ∈ [1e-4, 1.2e-2] (Brooks §3.1, p4), with read noise from their
  fitted log-log line. Higher levels are labelled "stress".
- MuJoCo 3.2.6 renders with fixed-function lighting and no gamma (`src/render/render_gl*.c`). We
  treat the render as display-referred camera output. A "render is linear" variant changed nothing
  at the tested levels.

**Two ISPs:**
- **Toy ISP** (`isp/camera.py`):
  - edge-aware demosaic, chroma-median false-colour suppression, then colour NLM;
  - "tuned" strength = PSNR-optimal per noise level; "strong" = 2× tuned.
- **Infinite-ISP** (10x-Engineers, Apache-2.0, @ `3221207`; `isp/infinite.py`):
  - its own Bayer joint-bilateral NR, Malvar-He-Cutler demosaic, sharpening, and luma-only NLM;
  - presets: **off** (no NR, no sharpening), **repo default** (as shipped), **tuned** (PSNR-optimal
    against the ISP's own noise-free look), **strong** (2× tuned);
  - WB, CCM, and tone curve are fixed to ours, so a noiseless frame matches the render. AWB, AE, and
    saturation boost are off.

**Controls:**
- The clean camera (RAW round trip, no noise) vs the render: no effect for either policy.
- Image-space Gaussian noise matched in PSNR to each RAW condition.

**Policies:** OFT (wrist + third-person) and OFT_w (third-person only), both 4-bit with clean
instructions.

### 4.2 Results

**Toy ISP, 3 tasks × 50 episodes (E004).** Drops are against the clean camera (98.7% for both
policies).

| Condition | OFT | OFT_w |
|---|---|---|
| Plausible noise (λ 1e-4 … 1.2e-2), tuned NR | ≤ 2.0 | ≤ 3.3 |
| +1 EV over-exposure | none | 0.7 |
| λ 1.2e-2, NR off | −29.3 | −6.7 |
| λ 1.2e-2, NR strong | −26.7 | −40.7 |
| Stress λ 3e-2 / 6e-2 | −49 / −71 | −78 / −96 |
| Image-space noise, matched PSNR (up to stress λ 3e-2) | none (≤ 0) | ≤ 2.0 |

**Infinite-ISP, 9 tasks (3 per suite) × 20 episodes, λ = 0.012 (E005).** Drops are against the
matching clean frame. Suite drops are listed as Spatial / Object / Goal.

| Preset | OFT | OFT suite drops | OFT_w | OFT_w suite drops |
|---|---|---|---|---|
| NR off | −13.3 | 0 / 30 / 10 | **−16.1** | 33 / 2 / 13 |
| Repo default | −6.1 | 2 / 15 / 2 | **−16.1** | 25 / 7 / 17 |
| Tuned | −1.1 | — | −0.6 | — |
| Strong | −2.8 | 0 / 8 / 0 | −8.3 | 18 / 3 / 3 |

**OFT confirmation on fresh init states, 9 tasks × 30 episodes (E006).**

| Preset | Drop | Spatial | Object | Goal |
|---|---|---|---|---|
| NR off | −7.8 (CIs separate) | −1 | +16 | +9 |
| Repo default | −5.9 | 0 | +13 | +4 |

**Pre-registered replication rule:** a ≥ 10-point drop with non-overlapping Wilson CIs, and ≥ 10
points in ≥ 2 of 3 suites, for both policies.
- E005 passed: OFT through NR off (borderline, with one suite exactly at +10), OFT_w through NR off
  and the repo default.
- E006 did **not** confirm OFT on fresh episodes.
- Net: the rule holds for OFT_w only.

### 4.3 What we conclude

1. **Realistic sensor noise is harmless when noise reduction is tuned to the gain.** This held with
   both ISPs and both policies.
2. **Un-tuned noise reduction at high gain hurts:**
   - a third-person-only policy by ≈ 16 points, across suites;
   - a wrist-camera policy by ≈ 8 points, concentrated in texture-dependent Object tasks.
   - The wrist-camera estimate declined with each replication: toy ISP −29 → real ISP −13 → fresh
     episodes −8.
3. **How noise reduction fails matters.** Over-smoothing with the toy colour-NLM hurts OFT_w broadly
   (−33 on 9 tasks, all 3 suites). Infinite-ISP's luma-only strong preset barely hurts (−8). One
   plausible difference is chroma smoothing. That is untested.
4. **Image-space noise does not predict any of this.** At matched PSNR, image-space noise stays
   ≥ 96.7% even where RAW-space stress noise drops success to 21–49%. Fine-grain noise is harmless;
   texture loss after a real pipeline is not.

## 5. Limitations

- **Only one ISP knob** (spatial NR strength) at one gain level for the real ISP. Untested:
  - temporal NR;
  - AE/AWB dynamics;
  - exposure time and motion blur;
  - tone mapping / HDR;
  - sharpening alone;
  - compression;
  - lens effects;
  - rolling shutter.

  Several of these interact with motion and could matter more in closed loop.
- **Approximated RAW** from 8-bit renders: no true HDR, no spectra, fixed camera parameters.
- **4-bit models** throughout. bf16 was not testable on this hardware.
- **Few tasks** (3 for the toy ISP, 9 for Infinite-ISP). 20–30 episodes per task means per-suite CIs
  of about ±10 points.
- **One open ISP** with its own presets. These are not tunings of real cameras.
- **Simulation only.** No real camera and no physical robot.

## 6. Pre-registered predictions vs outcomes

Predictions were frozen before each run. Roughly half missed. The notable misses:
- **Reproduction:** we predicted Object 65–80 (got 63.0 and 82.2), Goal Noise 68–82 (got 87.1, later
  explained by §3.2), and "Long diverges most" (Long matched best).
- **Noise families:** we predicted "fog and glass hurt most". Zoom and Gaussian blur did.
- **Instruction suffix:** we predicted it would cost ≤ 5 points. It cost 13 on Goal.
- **ISP probe:**
  - We predicted plausible noise would cost 5–30 points. It cost ≤ 3.3.
  - We predicted RAW noise would hurt *less* than image-space noise at matched PSNR. There was no
    difference at plausible levels, and it hurt far *more* at stress levels.
- **Run-to-run variation:** we predicted 3–6 of 30 outcomes would flip. None did.

## 7. Reproducing

See the README's **Reproduce** section. It has three tiers:
- **Tier 0:** recompute every table in this report from the per-episode CSVs in `results/`. CPU only, about a minute.
- **Tier 1:** run the camera model by itself.
- **Tier 2:** re-run the closed-loop experiments. It gives the upstream commits, environment pins, exact commands,
  and our per-experiment wall-clock times.

## 8. What was adopted, adapted, and authored

| | |
|---|---|
| **Adopted** | LIBERO-Plus benchmark and assets; OpenVLA-OFT harness and checkpoints; Infinite-ISP modules; the upstream LIBERO task list (MIT, vendored in `scripts/libero_original_task_map.py`) |
| **Adapted** | Brooks et al. unprocessing (fixed parameters, plus a forward ISP made round-trip-consistent); OFT harness wrappers (original-task suites, category selection, instruction cleaning, fail-loud error accounting, camera-model observation wrapper) |
| **Authored** | Experiment design and pre-registered decision rules; the camera/ISP probe (sensor model, toy ISP, Infinite-ISP integration and calibration); the benchmark findings in §3. Code was drafted with an AI coding assistant under the author's direction (commits tagged `[ai-scaffold]`) |

## References

- S. Fei et al., *LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*,
  arXiv 2510.13626. Table 1 p3, Table 2 p9, App. C.3 p18, Table 6 p16, Table 10 p31.
- M. J. Kim et al., *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success*
  (OpenVLA-OFT), arXiv 2502.19645. Table I p6.
- M. J. Kim et al., *OpenVLA*, arXiv 2406.09246. §3.5 p6, Table 2 p10 (quantized inference
  memory).
- T. Brooks et al., *Unprocessing Images for Learned Raw Denoising*, CVPR 2019, arXiv 1811.11127.
  §3.1–3.4 p4, §4 p5. Reference code: google-research/unprocessing (Apache-2.0).
- 10x-Engineers, *Infinite-ISP*, github.com/10x-Engineers/Infinite-ISP @ `3221207` (Apache-2.0).
- B. Liu et al., *LIBERO*, NeurIPS 2023 Datasets & Benchmarks.
