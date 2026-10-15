"""E009 (C1) conditions: exposure time t -> motion blur (sub-frames) + gain-dependent noise, on E008's matched
Filament HDR -> RAW -> Infinite-ISP chain. Notebook: experiments/E009-c1-exposure-tradeoff.md, decisions/012.

Per query frame (the only frames OFT looks at):
  1. the K = t / 2 ms physics-substep states at the END of the control step (global shutter, exposure ending
     at the observation) are rendered with Filament HDR and averaged in linear light (time integration);
  2. RAW: isp.raw_formation.form_raw_gain with E008's brightness-matched t* (frame 0, per camera per task)
     and gain G(t) (noise on/off);
  3. Infinite-ISP v1 with the NR preset tuned for this exposure (results/e009/nr_calibration.json), sharpening
     off; the clean baseline uses preset "off" (= E008 condition 4).
Condition keys: "t_ms" (None = instantaneous, K = 1), "noise" (bool), "nr" ("tuned" | "off").
Stage B (decisions/013): "k" overrides the sub-frame count (noise-only: k = 1 at a short exposure's gain);
"imgblur": {cam: length_px} replaces the time average by a constant linear blur of that length in linear HDR
on the instantaneous render (image-space control), angle uniform at random per episode and camera.
Stage C (decisions/014): "signed" keeps the RAW signed until after the Bayer NR (no black-level rectification
before NR; isp.infinite_v1.run_isp_v1_signed).
E010 (decisions/015, option C): "probe": {"n": N, "t_ms": t} -- after each (clean) query, probe_variants() forms N
noisy RAW -> ISP versions of the same instantaneous render at exposure t's gain and tuned NR, for an open-loop
action-sensitivity probe along the clean trajectory (the policy acts on the clean image only).
[ai-scaffold] — core: explain-back owed (decisions/012 D7).
"""
import json
import os
import sys
import zlib

import numpy as np
from scipy.ndimage import shift as nd_shift

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e008"))
sys.path.insert(0, os.path.join(HERE, ".."))

from conditions import FOVY  # noqa: E402  (E008)
from fil_client import CAMS, FilamentClient  # noqa: E402
from isp import raw_formation as rf  # noqa: E402
from isp.camera import gamma_expansion, inverse_smoothstep  # noqa: E402
from isp.infinite import preset  # noqa: E402
from isp.infinite_v1 import run_isp_v1, run_isp_v1_signed  # noqa: E402

SUBSTEP_S = 0.002
NR_CAL = os.path.join(HERE, "..", "results", "e009", "nr_calibration.json")


def nr_preset(t_ms, nr):
    """Infinite-ISP NR preset for exposure t_ms with sharpening off (every condition keeps E008's look)."""
    if nr == "off":
        return preset("off")
    cal = json.load(open(NR_CAL))[str(t_ms)]
    p = preset("tuned", bnr_scale=cal["bnr_scale"], wts=cal["wts"])
    p["sharpen"]["is_enable"] = False
    return p


def blur_angle(tag, cam):
    """Uniform angle in [0, pi), fixed per episode (tag = suite_task_episode) and camera."""
    return np.random.default_rng(zlib.crc32(f"{tag}/{cam}".encode())).uniform(0.0, np.pi)


def line_blur(img, length, angle, n=16):
    """Constant linear motion blur: mean of n sub-pixel shifts (bilinear) along a segment of `length` px at
    `angle`, centred. img: (H, W, 3) linear float."""
    out = np.zeros_like(img, dtype=np.float32)
    for s in (np.arange(n) + 0.5) / n - 0.5:
        out += nd_shift(img, (s * length * np.sin(angle), s * length * np.cos(angle), 0), order=1, mode="nearest")
    return out / n


class C1Conditions:
    def __init__(self, cond, log_path, dump_dir=None):
        self.t_ms = cond.get("t_ms")
        self.k = 1 if self.t_ms is None else int(round(self.t_ms * 1e-3 / SUBSTEP_S))
        if cond.get("k") is not None:
            self.k = int(cond["k"])
        self.imgblur = cond.get("imgblur")
        self.noise = bool(cond.get("noise", False))
        self.signed = bool(cond.get("signed", False))
        self.probe = cond.get("probe")
        if self.probe:
            self.probe_gain = rf.gain_for(self.probe["t_ms"] * 1e-3)
            self.probe_preset = nr_preset(self.probe["t_ms"], "tuned")
        self.gain = rf.gain_for(self.t_ms * 1e-3) if self.t_ms else 1.0
        self.preset = nr_preset(self.t_ms, cond.get("nr", "off"))
        self.k_capture = 16 if dump_dir else self.k          # dumps keep all of the 32 ms window
        self.client = FilamentClient(matched=True, log_path=log_path, ss=2)
        self.cos4 = {c: rf.cos4_map(256, FOVY[c]) for c in FOVY}
        self.dump_dir, self.n_query = dump_dir, 0
        self.tstar, self.stats = {}, {}

    def new_task(self):
        self.tstar, self.stats = {}, {}

    def calibrate(self, sim, classic):
        """Frame 0 (set_init_state obs): E008's raw_isp exposure match, once per camera per task."""
        if all(c in self.tstar for c in CAMS):
            return
        hdr = self.client.render(sim, "hdr")
        for c in CAMS:
            if c not in self.tstar:
                target = float((gamma_expansion(inverse_smoothstep(classic[c] / 255.0)) @ rf.REC709_Y).mean())
                self.tstar[c] = rf.exposure_time(rf.scene_luminance(hdr[c]), target)
                self.stats[c] = dict(t_star_s=float(self.tstar[c]), k=self.k, gain=self.gain, noise=self.noise,
                                     signed=self.signed,
                                     imgblur=(self.imgblur or {}).get(c))

    def query(self, sim, substates, rngs, pool=None, tag=""):
        """substates: list of state vectors at the substeps of the last control step (last = obs state).
        Returns dict cam -> uint8 image for the policy."""
        sub = substates[-self.k_capture:]
        each = self.client.render_each(sim, sub).astype(np.float32)          # (Kc, 2, H, W, 3)
        hdr = each[-self.k:].mean(0)                                          # time average over the exposure
        self.last_hdr = hdr
        if self.imgblur:
            hdr = np.stack([line_blur(hdr[i], self.imgblur[c], blur_angle(tag, c)) for i, c in enumerate(CAMS)])
        self.n_query += 1
        if self.dump_dir and self.n_query % 3 == 1:
            self._dump(each, sub, tag)

        def one(i, c):
            dn, clip = rf.form_raw_gain(rf.scene_luminance(hdr[i]), self.tstar[c], self.cos4[c], self.gain,
                                        rngs[c] if self.noise else None, signed=self.signed)
            if self.signed:
                return run_isp_v1_signed(dn, self.cos4[c], self.preset)[0]
            return run_isp_v1(dn, self.cos4[c], self.preset)
        if pool is not None:
            futs = {c: pool.submit(one, i, c) for i, c in enumerate(CAMS)}
            return {c: f.result() for c, f in futs.items()}
        return {c: one(i, c) for i, c in enumerate(CAMS)}

    def probe_variants(self, rng, cams, pool=None):
        """E010: N noisy versions (probe exposure's gain, noise on, tuned NR) of the last query's render.
        Returns [ {cam: uint8 image} ] of length N. One rng per episode, drawn in a fixed order."""
        out = []
        for _ in range(self.probe["n"]):
            seeds = {c: rng.integers(2 ** 63) for c in cams}

            def one(c):
                i = CAMS.index(c)
                dn, _ = rf.form_raw_gain(rf.scene_luminance(self.last_hdr[i]), self.tstar[c], self.cos4[c],
                                         self.probe_gain, np.random.default_rng(seeds[c]))
                return run_isp_v1(dn, self.cos4[c], self.probe_preset)
            if pool is not None:
                futs = {c: pool.submit(one, c) for c in cams}
                out.append({c: f.result() for c, f in futs.items()})
            else:
                out.append({c: one(c) for c in cams})
        return out

    def _dump(self, each, sub, tag):
        os.makedirs(self.dump_dir, exist_ok=True)
        avg = {f"avg{k}": each[-k:].mean(0).astype(np.float16) for k in (1, 2, 4, 8, 16) if k <= len(each)}
        np.savez(os.path.join(self.dump_dir, f"{tag}_q{self.n_query:03d}.npz"), **avg,
                 first=each[0].astype(np.float16), states=np.stack(sub),
                 tstar=np.array([self.tstar[c] for c in CAMS]))
