"""E009 Stage C: PSNR-optimal Infinite-ISP NR (sharpening off) for u8_noise4, i.e. the 8-bit unprocess path
(isp.infinite.simulate_infinite, E005) at E009's 4 ms noise parameters: lam_shot = 1/N4, read_var = (10/N4)^2.
Reference: the same frame through the same chain, noise-free, NR off, sharpening off (as e009/calibrate_nr.py).
Frames: E005's calibration frames (results/e004/frames.npz; classic render, 12 frames).
Grid as e009/calibrate_nr.py. Writes key "u8_4" into results/e009/nr_calibration.json. [ai-scaffold]
"""
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from isp import raw_formation as rf  # noqa: E402
from isp.camera import psnr  # noqa: E402
from isp.infinite import preset, simulate_infinite  # noqa: E402
from calibrate_nr import BNR_SCALES, OUT, WTS, nr  # noqa: E402

F = np.load(os.path.join(HERE, "..", "results", "e004", "frames.npz"))
FRAMES = np.concatenate([F["agentview"][::3], F["wrist"][::3]])
LAM = 1.0 / rf.FULL_SCALE_4MS
READ = (rf.READ_NOISE_E / rf.FULL_SCALE_4MS) ** 2


def score_one(i):
    f = FRAMES[i]
    ref = simulate_infinite(f, 0.0, 0.0, preset("off"), np.random.default_rng(0))
    res = {}
    for b, w in itertools.product(BNR_SCALES, WTS):
        out = simulate_infinite(f, LAM, READ, nr(b, w), np.random.default_rng(4000 + i))
        res[(b, w)] = psnr(ref, out)
    return res


def main():
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(score_one, range(len(FRAMES))))
    grid = {g: float(np.mean([r[g] for r in res])) for g in res[0]}
    (b, w), best = max(grid.items(), key=lambda kv: kv[1])
    out = json.load(open(OUT))
    out["u8_4"] = {"bnr_scale": b, "wts": w, "psnr_tuned": best, "psnr_off": grid[(0, 0)], "lam": LAM,
                   "read_var": READ, "n_images": len(FRAMES), "grid": {f"{g[0]},{g[1]}": v for g, v in grid.items()}}
    json.dump(out, open(OUT, "w"), indent=1)
    print({k: v for k, v in out["u8_4"].items() if k != "grid"})


if __name__ == "__main__":
    main()
