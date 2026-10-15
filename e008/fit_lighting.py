"""E008 L2: fit the matched Filament lighting to classic frame-0 images (image difference only).

Objective: mean |8-bit difference| between the classic render and Filament HDR -> per-image exposure
matched to classic mean luminance (the D3 rule) -> sRGB OETF, over the 6 exported images
(3 suites x {agentview, wrist}). Policy success is never used.
Intensities only matter as ratios once exposure is matched, so the scene lights are fixed and the
ambient and headlight are swept relative to them.

python e008/fit_lighting.py --scenes ~/research/logs/e008/g1 --out ~/research/logs/e008/fit.csv
[ai-scaffold]
"""
import argparse
import csv
import itertools
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filament_match import prepare_xml  # noqa: E402
from filament_render import CAMS, FilamentCams  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal"]
# fit 1 (textures as exported, i.e. linear): spec {0,.2,.6} x shin {.1,.5} x amb {0,2e3,5e3,1.5e4,4e4} x
# head {0,1e4,4e4,1.6e5}; best 22.09 (spec 0, amb 0, head 1e4). Fit 2 (textures decoded as sRGB):
GRID = dict(spec=[0.0, 0.2], shin=[0.1], amb=[0, 1e3, 3e3, 1e4], head=[0, 5e3, 1e4, 2e4, 4e4, 1.6e5])
LUM = np.array([0.2126, 0.7152, 0.0722])


def srgb_decode(u8):
    c = u8 / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255


def exposure_match(hdr, classic_u8):
    """D3: scale so the frame's mean luminance equals the classic frame's (both linear)."""
    return (srgb_decode(classic_u8) @ LUM).mean() / max((hdr @ LUM).mean(), 1e-12)


def image_error(hdr, classic_u8):
    k = exposure_match(hdr, classic_u8)
    return np.abs(srgb_encode(hdr * k) - classic_u8).mean()


class Scene:
    def __init__(self, scenes_dir, suite, mat_params):
        xml = open(os.path.join(scenes_dir, f"{suite}.xml")).read()
        self.m = mujoco.MjModel.from_xml_string(prepare_xml(xml, mat_params))
        self.d = mujoco.MjData(self.m)
        z = np.load(os.path.join(scenes_dir, f"{suite}_state.npz"))
        self.d.qpos[:], self.d.qvel[:] = z["qpos"], z["qvel"]
        self.d.mocap_pos[:], self.d.mocap_quat[:] = z["mocap_pos"], z["mocap_quat"]
        mujoco.mj_forward(self.m, self.d)
        self.classic = {c: z[c] for c in CAMS}
        self.fc = FilamentCams(self.m)
        name = lambda i: mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_LIGHT, i)
        self.lid = {name(i): i for i in range(self.m.nlight)}

    def set_intensity(self, amb, scene, head):
        self.m.light_intensity[self.lid["e008_ambient"]] = amb
        self.m.light_intensity[self.lid["light1"]] = scene
        self.m.light_intensity[self.lid["light2"]] = scene
        self.m.light_intensity[self.lid["e008_headlight"]] = head

    def errors(self):
        return [image_error(self.fc.render(self.d, c, "hdr"), self.classic[c]) for c in CAMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fields = ["spec", "shin", "amb", "scene", "head"] + [f"{s}_{c}" for s in SUITES for c in CAMS] + ["mean"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for spec, shin in itertools.product(GRID["spec"], GRID["shin"]):
            scenes = [Scene(a.scenes, s, dict(spec=spec, shin=shin)) for s in SUITES]
            best = None
            for amb, head in itertools.product(GRID["amb"], GRID["head"]):
                errs = []
                for sc in scenes:
                    sc.set_intensity(amb, 4e4, head)
                    errs += sc.errors()
                row = [spec, shin, amb, 4e4, head] + [round(e, 2) for e in errs] + [round(float(np.mean(errs)), 2)]
                w.writerow(row)
                f.flush()
                best = row if best is None or row[-1] < best[-1] else best
            print("spec", spec, "shin", shin, "best", best[2:5], "mean", best[-1], flush=True)
            for sc in scenes:
                del sc.fc


if __name__ == "__main__":
    main()
