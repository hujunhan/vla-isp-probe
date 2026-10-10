"""Adapter: run Infinite-ISP (10x-Engineers, Apache-2.0, ~/research/Infinite-ISP @ 3221207) stages on
the simulated RAW from isp.camera, replacing our toy ISP for E005 (R1, decisions/007).

Our side (unchanged from E004): render -> unprocess -> exposure/saturation -> RGGB mosaic -> shot+read
noise -> 12-bit quantisation.
Infinite-ISP side (their module code, called stage by stage; their file loader is bypassed):
    digital gain (ours, x RGB_GAIN) -> Bayer NR (their joint bilateral) -> white balance (their module,
    our fixed gains) -> demosaic (their Malvar-He-Cutler) -> CCM (their module, our CAM2RGB) -> gamma
    (their module, OUR tone LUT so a noiseless frame matches the render) -> RGB->YUV (their module,
    saturation enhancement OFF) -> sharpen (their unsharp mask) -> 2D NR (their NLM on luma) -> YUV->RGB

Deliberate deviations from their default config: AWB/AE off and fixed WB/CCM/tone (so conditions differ
only in the noise-reduction/sharpening preset); saturation enhancement off (it would recolour the scene
relative to the policy's training images). Everything else is their code and their parameters.

[ai-scaffold] — advisor-written at owner's request (decisions/006 D5b, decisions/007); explain-back owed.
"""
import copy
import os
import sys

import numpy as np

from isp.camera import (BLUE_GAIN, CAM2RGB, RED_GAIN, RGB_GAIN, add_sensor_noise, gamma_compression,
                        mosaic_rggb, quantize, smoothstep, unprocess)

INFINITE_ROOT = os.path.expanduser(os.environ.get("INFINITE_ISP_ROOT", "~/research/Infinite-ISP"))
if INFINITE_ROOT not in sys.path:
    sys.path.insert(0, INFINITE_ROOT)

from modules.bayer_noise_reduction.bayer_noise_reduction import BayerNoiseReduction as BNR  # noqa: E402
from modules.color_correction_matrix.color_correction_matrix import ColorCorrectionMatrix as CCM  # noqa: E402
from modules.color_space_conversion.color_space_conversion import ColorSpaceConversion as CSC  # noqa: E402
from modules.demosaic.demosaic import Demosaic  # noqa: E402
from modules.gamma_correction.gamma_correction import GammaCorrection as GC  # noqa: E402
from modules.noise_reduction_2d.noise_reduction_2d import NoiseReduction2d as NR2D  # noqa: E402
from modules.rgb_conversion.rgb_conversion import RGBConversion as RGBC  # noqa: E402
from modules.sharpen.sharpen import Sharpening as Sharpen  # noqa: E402
from modules.white_balance.white_balance import WhiteBalance as WB  # noqa: E402

# Silence the modules' per-stage print() calls without redirecting sys.stdout (which is not
# thread-safe; cameras are processed in parallel threads in E005).
for _m in list(sys.modules.values()):
    if getattr(_m, "__name__", "").startswith("modules."):
        _m.print = lambda *a, **k: None

BIT_DEPTH = 12
MAXV = 2 ** BIT_DEPTH - 1
PLATFORM = {"disable_progress_bar": True, "leave_pbar_string": False, "in_file": "sim", "save_format": "npy",
            "save_lut": False, "out_dir": "/tmp"}

# Repo defaults (config/configs.yml @ 3221207) for the stages we sweep.
REPO_BNR = {"is_enable": True, "filter_window": 9, "r_std_dev_s": 1, "r_std_dev_r": 0.1, "g_std_dev_s": 1,
            "g_std_dev_r": 0.08, "b_std_dev_s": 1, "b_std_dev_r": 0.1, "is_save": False}
REPO_SHARPEN = {"is_enable": True, "sharpen_sigma": 5, "sharpen_strength": 1, "is_save": False}
REPO_NR2D = {"is_enable": True, "window_size": 9, "patch_size": 5, "wts": 10, "is_save": False}


def preset(name, bnr_scale=1.0, wts=10.0):
    """Noise-reduction/sharpening presets.
    off          : BNR off, sharpen off, 2D NR off
    repo_default : the repository's shipped tuning (fixed, not gain-adaptive)
    tuned        : BNR range std x bnr_scale and NLM wts chosen per noise level for max PSNR (calibrated)
    strong       : 2x the tuned strengths (over-smoothing)"""
    bnr, sh, nr = copy.deepcopy(REPO_BNR), copy.deepcopy(REPO_SHARPEN), copy.deepcopy(REPO_NR2D)
    if name == "off":
        bnr["is_enable"] = sh["is_enable"] = nr["is_enable"] = False
    elif name == "repo_default":
        pass
    elif name in ("tuned", "strong"):
        k = 1.0 if name == "tuned" else 2.0
        for c in "rgb":
            bnr[f"{c}_std_dev_r"] = REPO_BNR[f"{c}_std_dev_r"] * bnr_scale * k
        bnr["is_enable"] = bnr_scale > 0
        nr["wts"] = wts * k
        nr["is_enable"] = wts > 0
    else:
        raise ValueError(name)
    return {"bnr": bnr, "sharpen": sh, "nr2d": nr}


def _tone_lut():
    x = np.arange(MAXV + 1) / MAXV
    return np.round(smoothstep(gamma_compression(x)) * MAXV).astype(int).tolist()


SENSOR = {"bayer_pattern": "rggb", "range": MAXV, "bit_depth": BIT_DEPTH, "width": 256, "height": 256}
WB_PARM = {"is_enable": True, "is_debug": False, "is_auto": False, "r_gain": RED_GAIN, "b_gain": BLUE_GAIN,
           "is_save": False}
CCM_PARM = {"is_enable": True, "corrected_red": CAM2RGB[0].tolist(), "corrected_green": CAM2RGB[1].tolist(),
            "corrected_blue": CAM2RGB[2].tolist(), "is_save": False}
GC_PARM = {"is_enable": True, "gamma_lut_12": _tone_lut(), "is_save": False}
CSC_PARM = {"conv_standard": 2, "is_save": False}
CSE_PARM = {"is_enable": False, "saturation_gain": 1.0}
RGB_PARM = {"is_enable": True, "is_save": False}


def run_isp(cfa01, p):
    """cfa01: HxW float RGGB in [0,1] (after noise + quantisation). p: preset dict. Returns uint8 HxWx3."""
    sensor = dict(SENSOR, height=cfa01.shape[0], width=cfa01.shape[1])
    raw = np.uint16(np.clip(np.round(cfa01 * RGB_GAIN * MAXV), 0, MAXV))       # digital gain
    raw = BNR(raw, sensor, p["bnr"], PLATFORM).execute()
    raw = WB(raw, PLATFORM, sensor, WB_PARM).execute()
    rgb = Demosaic(raw, PLATFORM, sensor, {"is_save": False}).execute()
    rgb = CCM(rgb, PLATFORM, sensor, CCM_PARM).execute()
    rgb = GC(rgb, PLATFORM, sensor, GC_PARM).execute()
    yuv = CSC(rgb, PLATFORM, sensor, CSC_PARM, CSE_PARM).execute()
    yuv = Sharpen(yuv, PLATFORM, sensor, p["sharpen"], 2).execute()
    yuv = NR2D(yuv, sensor, p["nr2d"], PLATFORM, 2).execute()
    out = RGBC(yuv, PLATFORM, sensor, RGB_PARM, CSC_PARM).execute()
    return np.asarray(out, dtype=np.uint8)


def simulate_infinite(frame_u8, shot_noise, read_var, p, rng, input_domain="srgb"):
    """Render (uint8) -> our sensor model -> Infinite-ISP -> uint8."""
    x = unprocess(frame_u8.astype(np.float64) / 255.0, input_domain)
    x = mosaic_rggb(x)
    x = add_sensor_noise(x, shot_noise, read_var, rng)
    x = quantize(x, BIT_DEPTH)
    return run_isp(x, p)
