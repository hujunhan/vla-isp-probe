"""E009 P3: per-exposure PSNR-optimal Infinite-ISP NR (sharpening off), before any Stage A run.

Frames: ~/research/logs/e009_tune_frames (dumped by the tuning run, init state 45; disjoint from Stage A's
init states 20-39). For exposure t (K = t / 2 ms), the input is the K-sub-frame average, so blur is identical in
reference and test; only noise and NR differ:
  reference = noise-free RAW (form_raw_gain, rng=None) -> ISP preset "off"   (E008 condition 4's look)
  test      = noisy RAW at gain G(t)                   -> ISP NR (bnr_scale, wts), sharpening off
Grid as scripts/calibrate_e005.py. Writes results/e009/nr_calibration.json. [ai-scaffold]
"""
import glob
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "e008"))
from isp import raw_formation as rf  # noqa: E402
from isp.camera import psnr  # noqa: E402
from isp.infinite import preset  # noqa: E402
from isp.infinite_v1 import run_isp_v1  # noqa: E402

FRAMES = os.path.join(os.path.expanduser(os.environ.get("LOG_DIR", "~/research/logs")), "e009_tune_frames")
OUT = os.path.join(HERE, "..", "results", "e009", "nr_calibration.json")
EXPOSURES_MS = tuple(int(a) for a in sys.argv[1:]) or (4, 8, 16, 32)   # Stage B adds 48 (merged into the json)
BNR_SCALES = [0, 1, 2, 4, 8, 16, 32, 64]   # E005 grid + 32, 64 (best hit the edge at 4 ms in a smoke test)
WTS = [0, 10, 20, 30, 45, 60, 90]
CAMS = ("agentview", "wrist")
FOVY = {"agentview": 45.0, "wrist": 75.0}
COS4 = {c: rf.cos4_map(256, FOVY[c]) for c in CAMS}


def nr(b, w):
    p = preset("tuned", bnr_scale=b, wts=w)
    p["sharpen"]["is_enable"] = False
    return p


def frames():
    """[(file index, camera index)], every frame of every dump."""
    fs = sorted(glob.glob(os.path.join(FRAMES, "*.npz")))
    return fs, [(i, j) for i in range(len(fs)) for j in range(len(CAMS))]


def score_one(args):
    """PSNR of every grid point on one (frame, camera, exposure); one noise draw per frame, shared by the grid."""
    path, j, t_ms, seed = args
    d = np.load(path)
    k = min(t_ms // 2, 16)       # dumps hold 32 ms of sub-frames; 48 ms uses the 32 ms average (noise level is what matters)
    L = rf.scene_luminance(d[f"avg{k}"][j].astype(np.float32))
    c = CAMS[j]
    clean, _ = rf.form_raw_gain(L, float(d["tstar"][j]), COS4[c], 1.0)
    ref = run_isp_v1(clean, COS4[c], preset("off"))
    noisy, _ = rf.form_raw_gain(L, float(d["tstar"][j]), COS4[c], rf.gain_for(t_ms * 1e-3),
                                np.random.default_rng(seed))
    return {(b, w): psnr(ref, run_isp_v1(noisy, COS4[c], nr(b, w))) for b, w in itertools.product(BNR_SCALES, WTS)}


def main():
    fs, fr = frames()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    out.update({"_doc": __doc__.splitlines()[0], "n_images": len(fr), "frames_dir": FRAMES})
    with ProcessPoolExecutor(max_workers=int(os.environ.get("NPROC", "8"))) as ex:
        for t_ms in EXPOSURES_MS:
            jobs = [(fs[i], j, t_ms, 1000 * t_ms + n) for n, (i, j) in enumerate(fr)]
            res = list(ex.map(score_one, jobs))
            grid = {g: float(np.mean([r[g] for r in res])) for g in res[0]}
            (b, w), best = max(grid.items(), key=lambda kv: kv[1])
            out[str(t_ms)] = {"bnr_scale": b, "wts": w, "psnr_tuned": best, "psnr_off": grid[(0, 0)],
                              "grid": {f"{g[0]},{g[1]}": v for g, v in grid.items()}}
            print(t_ms, {k: v for k, v in out[str(t_ms)].items() if k != "grid"}, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
