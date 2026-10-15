"""Infinite-ISP chain for RAW formation v1 (E008 condition 4). Phase 2's `isp/infinite.py` is unchanged.

Differences from `infinite.run_isp` (which takes Brooks-unprocessed RAW):
  * input is 12-bit DN with black level 256 -> Infinite-ISP's own BLC module (linearising, sat 4095);
  * ideal lens-shading correction (ours: multiply by 1 / cos^4; Infinite-ISP's LSC module is a stub);
  * no digital gain (exposure is set in the sensor model);
  * white balance and CCM derived from the sensor model (isp.raw_formation.WB / CCM), not Brooks' constants;
  * tone: Phase 2's LUT (smoothstep o gamma 1/2.2), decisions/010 D4.
Noise reduction / sharpening follow a preset from `infinite.preset` (E008: "off").
[ai-scaffold]
"""
import numpy as np

from isp import raw_formation as rf
from isp.infinite import (CSC, CSC_PARM, CSE_PARM, GC, GC_PARM, MAXV, NR2D, PLATFORM, RGB_PARM, RGBC, SENSOR,
                          BNR, CCM, Demosaic, Sharpen, WB)
import modules.black_level_correction.black_level_correction as _blc_mod  # noqa: E402
from modules.black_level_correction.black_level_correction import BlackLevelCorrection as BLC  # noqa: E402

_blc_mod.print = lambda *a, **k: None   # silence per-call prints, as infinite.py does for its modules

BLC_PARM = {"is_enable": True, "is_linear": True, "is_save": False,
            **{f"{c}_offset": rf.BLACK for c in ("r", "gr", "gb", "b")},
            **{f"{c}_sat": rf.MAXDN for c in ("r", "gr", "gb", "b")}}
WB_V1 = {"is_enable": True, "is_debug": False, "is_auto": False, "r_gain": float(rf.WB[0]),
         "b_gain": float(rf.WB[2]), "is_save": False}
CCM_V1 = {"is_enable": True, "corrected_red": rf.CCM[0].tolist(), "corrected_green": rf.CCM[1].tolist(),
          "corrected_blue": rf.CCM[2].tolist(), "is_save": False}


def run_isp_v1(dn, cos4, p):
    """dn: HxW uint16 RGGB (12-bit, BL 256). cos4: HxW relative illumination. p: preset dict. -> uint8 HxWx3."""
    sensor = dict(SENSOR, height=dn.shape[0], width=dn.shape[1])
    raw = BLC(dn, PLATFORM, sensor, BLC_PARM).execute()
    raw = np.uint16(np.clip(np.round(np.asarray(raw, np.float64) / cos4), 0, MAXV))   # ideal LSC
    raw = BNR(raw, sensor, p["bnr"], PLATFORM).execute()
    raw = WB(raw, PLATFORM, sensor, WB_V1).execute()
    rgb = Demosaic(raw, PLATFORM, sensor, {"is_save": False}).execute()
    rgb = CCM(rgb, PLATFORM, sensor, CCM_V1).execute()
    rgb = GC(rgb, PLATFORM, sensor, GC_PARM).execute()
    yuv = CSC(rgb, PLATFORM, sensor, CSC_PARM, CSE_PARM).execute()
    yuv = Sharpen(yuv, PLATFORM, sensor, p["sharpen"], 2).execute()
    yuv = NR2D(yuv, sensor, p["nr2d"], PLATFORM, 2).execute()
    out = RGBC(yuv, PLATFORM, sensor, RGB_PARM, CSC_PARM).execute()
    return np.asarray(out, dtype=np.uint8)


# ---------------------------------------------------------------- E009 Stage C: no rectification before Bayer NR
# Notebook E009 "Stage C", decisions/014. BLC linearisation and ideal LSC are done in float with no lower clip;
# Infinite-ISP's Bayer NR runs on the signed RAW plus a pedestal in a 14-bit container, with its range sigma
# rescaled so that the filter is the same in absolute (12-bit) terms. Its range kernel depends only on pixel
# differences (joint_bf.py:417-430), so the pedestal is transparent. Then the pedestal is removed, the image is
# clipped at 0, and the chain continues unchanged from WB. Rounding mirrors the v1 chain (BLC and BNR truncate).
PEDESTAL, BITS_C = 6000, 14


def run_isp_v1_signed(dn, cos4, p):
    """dn: HxW int32 RGGB DN (BL 256; may be < 0), from form_raw_gain(..., signed=True). -> (uint8 HxWx3, info)."""
    raw = np.floor((dn.astype(np.float64) - rf.BLACK) / (rf.MAXDN - rf.BLACK) * MAXV)    # BLC, no lower clip
    raw = np.minimum(np.round(raw / cos4), MAXV)                                            # ideal LSC
    info = {"neg_frac": float((raw < 0).mean())}
    if p["bnr"]["is_enable"]:
        top = 2 ** BITS_C - 1
        c = raw + PEDESTAL
        info["container_clip"] = float(((c < 0) | (c > top)).mean())
        sensor = dict(SENSOR, height=dn.shape[0], width=dn.shape[1], bit_depth=BITS_C, range=top)
        bnr = dict(p["bnr"], **{f"{ch}_std_dev_r": p["bnr"][f"{ch}_std_dev_r"] * MAXV / top for ch in "rgb"})
        out = BNR(np.uint16(np.clip(c, 0, top)), sensor, bnr, PLATFORM).execute()
        raw = np.asarray(out, np.float64) - PEDESTAL
    raw = np.uint16(np.clip(raw, 0, MAXV))
    sensor = dict(SENSOR, height=dn.shape[0], width=dn.shape[1])
    raw = WB(raw, PLATFORM, sensor, WB_V1).execute()
    rgb = Demosaic(raw, PLATFORM, sensor, {"is_save": False}).execute()
    rgb = CCM(rgb, PLATFORM, sensor, CCM_V1).execute()
    rgb = GC(rgb, PLATFORM, sensor, GC_PARM).execute()
    yuv = CSC(rgb, PLATFORM, sensor, CSC_PARM, CSE_PARM).execute()
    yuv = Sharpen(yuv, PLATFORM, sensor, p["sharpen"], 2).execute()
    yuv = NR2D(yuv, sensor, p["nr2d"], PLATFORM, 2).execute()
    out = RGBC(yuv, PLATFORM, sensor, RGB_PARM, CSC_PARM).execute()
    return np.asarray(out, dtype=np.uint8), info
