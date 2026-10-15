"""E008 conditions 2-4 (oft env), applied to the current sim state (decisions/010, E008 record).

  post     : Filament as shipped (fallback lighting), post-processing on (its tone mapper), 8-bit
  oetf     : Filament matched (L2 frozen), RGBA16F -> exposure k -> clip -> sRGB OETF -> 8-bit
  raw_isp  : Filament matched, RGBA16F -> RAW formation v1 (noise off) -> Infinite-ISP clean -> 8-bit
Exposure (D3): per camera per task, set on frame 0 of the task's first episode (the obs returned by
set_init_state) so the output's mean luminance matches the classic frame, then held for the whole task.
  oetf    : match in sRGB-linear (classic decoded with the sRGB EOTF)
  raw_isp : match in the ISP's pre-tone linear space (classic decoded with Phase 2's inverse tone,
            inverse_smoothstep then gamma 2.2), so each pipeline is matched through its own tone curve.
[ai-scaffold]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from fil_client import FilamentClient  # noqa: E402
from isp import raw_formation as rf  # noqa: E402
from isp.camera import gamma_expansion, inverse_smoothstep  # noqa: E402

FOVY = {"agentview": 45.0, "wrist": 75.0}      # MJCF camera fovy (agentview default 45; wrist 75 in the XML)
LUM = rf.REC709_Y


def srgb_decode(u8):
    c = np.asarray(u8, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_encode_u8(x):
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
    return (y * 255.0 + 0.5).astype(np.uint8)


class FilamentConditions:
    def __init__(self, fil, log_path):
        assert fil in ("post", "oetf", "raw_isp"), fil
        self.fil = fil
        # matched conditions integrate over the pixel area (2x2 supersampling, linear); "post" stays as shipped
        self.client = FilamentClient(matched=(fil != "post"), log_path=log_path, ss=1 if fil == "post" else 2)
        self.exposure = {}          # cam -> k (oetf) or t seconds (raw_isp); cleared per task
        self.stats = {}             # cam -> dict, logged once per task
        self.cos4 = {c: rf.cos4_map(256, FOVY[c]) for c in FOVY}
        if fil == "raw_isp":
            from isp.infinite import preset
            from isp.infinite_v1 import run_isp_v1
            self.preset, self.run_isp = preset("off"), run_isp_v1

    def new_task(self):
        self.exposure, self.stats = {}, {}

    def _calibrate(self, cam, hdr, classic_u8):
        if self.fil == "oetf":
            k = (srgb_decode(classic_u8) @ LUM).mean() / max(float((hdr @ LUM).mean()), 1e-12)
            self.exposure[cam] = k
            self.stats[cam] = dict(k=float(k), clip=float((hdr * k > 1).any(-1).mean()))
        else:
            target = float((gamma_expansion(inverse_smoothstep(classic_u8 / 255.0)) @ LUM).mean())
            t = rf.exposure_time(rf.scene_luminance(hdr), target)
            self.exposure[cam] = t
            _, clip = rf.form_raw(rf.scene_luminance(hdr), t, self.cos4[cam])
            self.stats[cam] = dict(t_s=float(t), clip=clip)

    def apply(self, sim, classic, frame0, pool=None):
        """classic: dict cam -> uint8 obs image. Returns dict cam -> uint8 image for the policy, or None
        before calibration (the reset() obs, which OFT discards)."""
        if self.fil == "post":
            return self.client.render(sim, "post")
        hdr = self.client.render(sim, "hdr")
        if frame0:
            for c in hdr:
                if c not in self.exposure:
                    self._calibrate(c, hdr[c], classic[c])
        if any(c not in self.exposure for c in hdr):
            return None
        if self.fil == "oetf":
            return {c: srgb_encode_u8(hdr[c] * self.exposure[c]) for c in hdr}

        def one(c):
            dn, _ = rf.form_raw(rf.scene_luminance(hdr[c]), self.exposure[c], self.cos4[c])
            return self.run_isp(dn, self.cos4[c], self.preset)
        if pool is not None:
            futs = {c: pool.submit(one, c) for c in hdr}
            return {c: f.result() for c, f in futs.items()}
        return {c: one(c) for c in hdr}
