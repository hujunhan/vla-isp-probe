"""E005 calibration (before any policy run): the 'tuned' Infinite-ISP preset per noise level.

For each shot-noise level, grid-search Bayer-NR range scale (bnr_scale) and luma-NLM strength (wts)
to maximise mean PSNR of preset(noisy) against the ISP's NOISE-FREE LOOK: the same frame, noiseless,
with the repo sharpening ON and both noise-reduction stages OFF. (A first version compared against
preset(noiseless), which smooths the reference too and so never penalises over-smoothing; replaced
before any policy run.) 'strong' is defined as 2x the tuned strengths. Writes results/e005/calibration.json.
[ai-scaffold]
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isp.camera import psnr, read_noise_for  # noqa: E402
from isp.infinite import preset, simulate_infinite  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
F = np.load(os.path.join(ROOT, "results", "e004", "frames.npz"))
FRAMES = np.concatenate([F["agentview"][::3], F["wrist"][::3]])   # 12 frames: both cams, all 3 suites
LEVELS = [1.2e-2]
BNR_SCALES = [0, 1, 2, 4, 8, 16]
WTS = [0, 10, 20, 30, 45, 60, 90]


def score(lam, bnr_scale, wts):
    p = preset("tuned", bnr_scale=bnr_scale, wts=wts)
    look = preset("tuned", bnr_scale=0, wts=0)          # sharpening on, NR off
    vals = []
    for i, f in enumerate(FRAMES):
        ref = simulate_infinite(f, 0.0, 0.0, look, np.random.default_rng(0))
        out = simulate_infinite(f, lam, read_noise_for(lam), p, np.random.default_rng(100 + i))
        vals.append(psnr(ref, out))
    return float(np.mean(vals))


def main():
    out = {}
    for lam in LEVELS:
        grid = {(b, w): score(lam, b, w) for b, w in itertools.product(BNR_SCALES, WTS)}
        (b, w), best = max(grid.items(), key=lambda kv: kv[1])
        out[f"{lam:g}"] = {"bnr_scale": b, "wts": w, "psnr_tuned": best,
                           "psnr_off": grid[(0, 0)],
                           "psnr_strong": score(lam, 2 * b, 2 * w),
                           "grid": {f"{k[0]},{k[1]}": v for k, v in grid.items()}}
        print(lam, {k: v for k, v in out[f"{lam:g}"].items() if k != "grid"}, flush=True)
    os.makedirs(os.path.join(ROOT, "results", "e005"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "results", "e005", "calibration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
