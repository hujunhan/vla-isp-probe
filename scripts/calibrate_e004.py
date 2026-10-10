"""E004 calibration on rendered frames (results/e004/frames.npz), done once before any policy run.

For each input domain and shot-noise level:
  * denoise_h: the NLM strength that maximises mean PSNR against the clean camera (an ISP tuned for
    its noise level) -> 'default'; 'strong' = 2x.
  * imgspace_sigma: the 8-bit Gaussian sigma whose mean PSNR against the clean camera equals that of
    the RAW condition with default denoise (the S4 matched-severity control).
Writes results/e004/denoise_calibration.json. [ai-scaffold]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isp.camera import CameraConfig, image_space_gaussian, psnr, simulate  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
F = np.load(os.path.join(ROOT, "results", "e004", "frames.npz"))
FRAMES = np.concatenate([F["agentview"], F["wrist"]])
LEVELS = {"srgb": [1e-4, 1e-3, 5e-3, 1.2e-2, 3e-2, 6e-2], "linear": [5e-3, 1.2e-2]}
H_GRID = [0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30]


def mean_psnr(cfg, refs, seed0=0):
    return float(np.mean([psnr(r, simulate(f, cfg, np.random.default_rng(seed0 + i)))
                          for i, (f, r) in enumerate(zip(FRAMES, refs))]))


def main():
    out = {}
    for domain, levels in LEVELS.items():
        clean = [simulate(f, CameraConfig(input_domain=domain), np.random.default_rng(0)) for f in FRAMES]
        out[domain] = {"clean_camera_psnr_vs_render": float(np.mean([psnr(f, c) for f, c in zip(FRAMES, clean)]))}
        for lam in levels:
            scores = {h: mean_psnr(CameraConfig(input_domain=domain, shot_noise=lam, denoise="default", denoise_h=h), clean)
                      for h in H_GRID}
            h_best = max(scores, key=scores.get)
            p_default = scores[h_best]
            p_strong = mean_psnr(CameraConfig(input_domain=domain, shot_noise=lam, denoise="strong", denoise_h=h_best), clean)
            # sigma such that image-space Gaussian on the clean camera has the same mean PSNR (bisection)
            lo, hi = 0.0, 128.0
            for _ in range(30):
                mid = (lo + hi) / 2
                p = float(np.mean([psnr(c, image_space_gaussian(c, mid, np.random.default_rng(i))) for i, c in enumerate(clean)]))
                lo, hi = (mid, hi) if p > p_default else (lo, mid)
            out[domain][f"{lam:g}"] = {"denoise_h": h_best, "psnr_off": scores[0], "psnr_default": p_default,
                                       "psnr_strong": p_strong, "imgspace_sigma": round((lo + hi) / 2, 3)}
            print(domain, lam, out[domain][f"{lam:g}"], flush=True)
    path = os.path.join(ROOT, "results", "e004", "denoise_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
