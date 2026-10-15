"""E009 Stage C pre-flight S1-S3 (notebook E009 "Stage C"). Frames: e009 tuning dumps (init state 45).
S1  noise off: signed chain == v1 chain within 1 LSB (8-bit) on >= 99% of pixels, BNR on and off.
S2  4 ms noise: dark-region bias (mean noisy - clean over clean luma < 40/255) for v1 vs signed.
S3  u8_noise4 RAW-domain variance at x = 0.18 (Monte Carlo of isp.camera.add_sensor_noise). [ai-scaffold]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "e008"))
from isp import raw_formation as rf  # noqa: E402
from isp.camera import add_sensor_noise  # noqa: E402
from isp.infinite import preset  # noqa: E402
from isp.infinite_v1 import run_isp_v1, run_isp_v1_signed  # noqa: E402
from calibrate_nr import COS4, CAMS, FRAMES, nr  # noqa: E402

Y = np.array([0.299, 0.587, 0.114])


def main():
    cal = json.load(open(os.path.join(HERE, "..", "results", "e009", "nr_calibration.json")))["4"]
    p4, off = nr(cal["bnr_scale"], cal["wts"]), preset("off")
    g4 = rf.gain_for(0.004)
    fs = sorted(glob.glob(os.path.join(FRAMES, "*.npz")))[::4]
    s1 = {"bnr_on": [], "bnr_off": []}
    s2 = {c: {"v1": [], "signed": [], "v1_all": [], "signed_all": [], "neg": [], "cclip": []} for c in CAMS}
    for n, f in enumerate(fs):
        d = np.load(f)
        for j, c in enumerate(CAMS):
            L = rf.scene_luminance(d["avg1"][j].astype(np.float32))
            t = float(d["tstar"][j])
            dn, _ = rf.form_raw_gain(L, t, COS4[c], g4)
            dns, _ = rf.form_raw_gain(L, t, COS4[c], g4, signed=True)
            for key, pp in (("bnr_on", p4), ("bnr_off", off)):
                a = run_isp_v1(dn, COS4[c], pp).astype(int)
                b = run_isp_v1_signed(dns, COS4[c], pp)[0].astype(int)
                s1[key].append(float((np.abs(a - b) <= 1).mean()))
            clean = run_isp_v1(dn, COS4[c], p4).astype(float)
            dark = clean @ Y < 40
            seed = 7000 + 10 * n + j
            nv, _ = rf.form_raw_gain(L, t, COS4[c], g4, np.random.default_rng(seed))
            ns, _ = rf.form_raw_gain(L, t, COS4[c], g4, np.random.default_rng(seed), signed=True)
            ov = run_isp_v1(nv, COS4[c], p4).astype(float)
            os_, info = run_isp_v1_signed(ns, COS4[c], p4)
            os_ = os_.astype(float)
            s2[c]["v1"].append(float((ov - clean)[dark].mean()) if dark.any() else np.nan)
            s2[c]["signed"].append(float((os_ - clean)[dark].mean()) if dark.any() else np.nan)
            s2[c]["v1_all"].append(float((ov - clean).mean()))
            s2[c]["signed_all"].append(float((os_ - clean).mean()))
            s2[c]["neg"].append(info["neg_frac"])
            s2[c]["cclip"].append(info.get("container_clip", 0.0))
    out = {"n_frames": len(fs),
           "S1_min_share_within_1lsb": {k: float(np.min(v)) for k, v in s1.items()},
           "S2": {c: {k: float(np.nanmean(v)) for k, v in s2[c].items()} for c in CAMS}}
    x, n4 = 0.18, rf.FULL_SCALE_4MS
    sig = add_sensor_noise(np.full(2_000_000, x), 1.0 / n4, (rf.READ_NOISE_E / n4) ** 2, np.random.default_rng(1))
    out["S3"] = {"lam_shot": 1.0 / n4, "read_var": (rf.READ_NOISE_E / n4) ** 2, "var_at_0.18": float(sig.var()),
                 "target": x * (1.0 / n4) + (rf.READ_NOISE_E / n4) ** 2}
    json.dump(out, open(os.path.join(HERE, "..", "results", "e009", "preflight_c.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
