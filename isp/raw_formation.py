"""RAW formation v1 (E008 condition 4; design: notebook lit/raw-formation-study-2026-10-11.md §2,
decisions/010 R1-R6). Filament HDR (scene-linear Rec.709 RGB x Filament exposure) -> physical RAW.

  stage 1  RGB -> spectrum      L(lam) = sum_k L_k * P_k(lam): primaries = ISETCam default display
                                (LCD-Apple, displays/displayCreate.m:50), each scaled so a unit value has
                                the Rec.709 luminance weight of its channel (683 * sum P V dlam = Y_k).
                                The method is ISETCam's sceneFromFile.m:164-211.
  stage 2  optics               E_q = pi / (1 + 4 N^2) * q(lam) * cos^4(theta)   (oiCalculateIrradiance.m:89,
                                opticsCos4th.m); N = 2.0 (R6, an assumption), T = 1, no PSF.
  stage 3  electrons            e_c = t * A_pix * FF * sum_lam E_q(lam) QE_c(lam) IR(lam) dlam
                                (SignalCurrentDensity.m:114, sensorCompute.m:160). QE = IMX490 public curve
                                (qe_imx490.mat, "grabit scan of LUCID curves"), IR = ISETCam example filter (R1).
           stages 1-3 are linear in L, so they collapse to e = t * cos^4 * (M @ L), M 3x3 precomputed.
  stage 4  CFA                  RGGB sampling.
  stage 5  noise                OFF in E008 (renderer-gap test); hooks for C1 are deliberately absent.
  stage 6  saturation + ADC     clip at full well; DN = round(e / FW * (4095 - BL) + BL), 12-bit, BL = 256 (R5).
  stage 7  exposure             t from a rule supplied by the caller (E008: D3 match-to-classic, held per task).

Placeholders that do NOT affect E008 (noise off, exposure matched): pixel 3.0 um, fill factor 1, full well
10,000 e- (ISETCam default APS well: pixelCreate.m:93-94). Real values come with R4 (EMVA 1288) in C1.

ISP side: white balance and CCM are derived from M (white = display white), so with noise off the chain is
colorimetrically exact up to clipping/quantisation/demosaic (study §3: no metamerism in a 3-primary scene).

[ai-scaffold] — advisor-written at the owner's request (decisions/010 D6); explain-back owed before E008
results count.
"""
import os

import numpy as np
import scipy.io as sio

ISETCAM = os.path.expanduser(os.environ.get("ISETCAM_ROOT", "~/research/ext/isetcam"))
FILAMENT_EXPOSURE = 2.604e-5   # Filament default camera exposure (mujoco_hdr IMPLEMENTATION_NOTES, "Data semantics")
REC709_Y = np.array([0.2126, 0.7152, 0.0722])
H, C = 6.62607015e-34, 2.99792458e8
WAVE = np.arange(400.0, 701.0, 5.0)            # nm; the IR filter is defined from 400 nm
DLAM = 5.0

F_NUMBER = 2.0
PIXEL_M = 3.0e-6
FILL_FACTOR = 1.0
FULL_WELL = 10000.0
BITS, BLACK = 12, 256
MAXDN = 2 ** BITS - 1


def _load(path, field="data"):
    d = sio.loadmat(os.path.join(ISETCAM, path), squeeze_me=True, struct_as_record=False)
    return d


def _interp(wave, data):
    data = np.atleast_2d(np.asarray(data, float).T).T if np.ndim(data) == 1 else np.asarray(data, float)
    return np.stack([np.interp(WAVE, wave, data[:, i], left=0.0, right=0.0) for i in range(data.shape[1])], 1)


def spectral_matrix():
    """Return (M, parts): M[c, k] = electrons per second per pixel at the image centre for unit L_k (cd/m^2)."""
    disp = _load("data/displays/LCD-Apple.mat")["d"]
    spd = _interp(np.asarray(disp.wave, float), disp.spd)                      # (nw, 3), W/(sr m^2 nm)
    v = _load("data/human/luminosity.mat")
    vlam = _interp(np.asarray(v["wavelength"], float), v["data"])[:, 0]        # V(lambda), energy format
    lum = 683.0 * (spd * vlam[:, None]).sum(0) * DLAM                          # cd/m^2 per unit primary
    spd = spd * (REC709_Y / lum)[None, :]                                      # unit L_k -> luminance Y_k
    qe = _load("data/sensor/colorfilters/auto/SONY/qe_imx490.mat")
    qe = _interp(np.asarray(qe["wavelength"], float), qe["data"])              # (nw, 3)
    ir = _load("data/sensor/irfilters/infrared.mat")
    ir = _interp(np.asarray(ir["wavelength"], float), ir["data"])[:, 0]
    photons = spd * (WAVE * 1e-9 / (H * C))[:, None]                           # quanta/(s sr m^2 nm)
    optics = np.pi / (1.0 + 4.0 * F_NUMBER ** 2)
    area = PIXEL_M ** 2 * FILL_FACTOR
    M = area * optics * ((qe * ir[:, None]).T @ photons) * DLAM                # (3 cam, 3 primaries)
    return M, dict(spd=spd, qe=qe, ir=ir, vlam=vlam)


M, _PARTS = spectral_matrix()
WHITE_CAM = M @ np.ones(3)                                   # camera response to display white
WB = WHITE_CAM[1] / WHITE_CAM                                # gains (R, G=1, B)
CCM = WHITE_CAM[1] * np.linalg.inv(np.diag(WB) @ M)          # rows sum to 1: white stays white


def cos4_map(size, fovy_deg):
    """Relative illumination cos^4(theta) for a pinhole camera with vertical field of view fovy (square image)."""
    f = (size / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    c = (np.arange(size) + 0.5) - size / 2.0
    r2 = c[None, :] ** 2 + c[:, None] ** 2
    return (f ** 2 / (f ** 2 + r2)) ** 2


def scene_luminance(hdr):
    """Filament HDR readback -> scene-linear Rec.709 RGB in cd/m^2 (exposure divided out)."""
    return np.asarray(hdr, np.float64) / FILAMENT_EXPOSURE


def exposure_time(L, target_mean_lum):
    """t (s) so the ISP's linear output (after BLC, ideal LSC, WB, CCM; before tone) has mean luminance
    target_mean_lum in [0, 1] units of full scale. With ideal LSC and the CCM above, that output is
    WHITE_CAM[1] * L * t / FULL_WELL, unclipped."""
    return target_mean_lum * FULL_WELL / (WHITE_CAM[1] * float((L @ REC709_Y).mean()))


def form_raw(L, t, cos4):
    """Noise-free RAW: electrons -> clip at full well -> 12-bit DN with black level. Returns (dn uint16 HxW,
    clipped fraction)."""
    e = t * cos4[..., None] * (L @ M.T)                     # (H, W, 3) electrons per channel
    cfa = np.empty(e.shape[:2])
    cfa[0::2, 0::2] = e[0::2, 0::2, 0]
    cfa[0::2, 1::2] = e[0::2, 1::2, 1]
    cfa[1::2, 0::2] = e[1::2, 0::2, 1]
    cfa[1::2, 1::2] = e[1::2, 1::2, 2]
    clipped = float((cfa >= FULL_WELL).mean())
    dn = np.round(np.clip(cfa, 0.0, FULL_WELL) / FULL_WELL * (MAXDN - BLACK) + BLACK)
    return dn.astype(np.uint16), clipped


# ---------------------------------------------------------------- C1 (E009): exposure time, gain and noise
# Notebook experiments/E009-c1-exposure-tradeoff.md, "Implementation refinements"; decisions/012 D4.
# Light is fixed per task; exposure t changes only (a) the blur, via the number of averaged sub-frames, and
# (b) the analog gain G(t) that keeps brightness constant, and with it the noise.
READ_NOISE_E = 10.0            # e- rms: ISETCam default pixel, read noise 1 mV / conversion gain 1e-4 V/e-
                               # (sensor/pixel/pixelCreate.m:93, 117)
ANCHOR_T, ANCHOR_X = 0.004, 0.18
ANCHOR_LAMBDA = 0.012          # E005's noise level (Brooks shot + paired read variance, isp/camera.py:64-66)


def _anchor_full_scale():
    """Full-scale electrons N4 at 4 ms such that the total noise variance at normalised RAW level 0.18 equals
    E005's: x/N + (READ_NOISE_E/N)^2 = x*lambda + read_var(lambda). Solved as a quadratic in u = 1/N."""
    read_var = float(np.exp(2.18 * np.log(ANCHOR_LAMBDA) + 1.20))
    total = ANCHOR_X * ANCHOR_LAMBDA + read_var
    a, b = READ_NOISE_E ** 2, ANCHOR_X
    u = (-b + np.sqrt(b * b + 4 * a * total)) / (2 * a)
    return 1.0 / u


FULL_SCALE_4MS = _anchor_full_scale()        # ~246.5 e-


def gain_for(t):
    """Analog gain (output full scale = FULL_WELL / G electrons) at exposure t seconds: G proportional to 1/t."""
    return FULL_WELL / (FULL_SCALE_4MS * t / ANCHOR_T)


def form_raw_gain(L, t_star, cos4, gain, rng=None, signed=False):
    """C1 RAW: L = scene-linear radiance (time-averaged over the exposure's sub-frames), t_star = E008's
    brightness-matched exposure (sets the output level). The output-referred electron image E is E008's;
    the sensor actually collects e = E / gain. Noise (rng given): Poisson shot on e, Gaussian read noise.
    DN = e * gain / FULL_WELL * (4095 - BL) + BL, rounded, clipped to [0, 4095]. With rng=None this equals
    form_raw(L, t_star, cos4) exactly, for every gain. Returns (dn uint16 HxW, clipped fraction).
    signed=True (E009 Stage C): no lower clip at the ADC; returns int32 DN that may lie below 0 (same rounding,
    same upper clip), for infinite_v1.run_isp_v1_signed."""
    E = t_star * cos4[..., None] * (L @ M.T)
    cfa = np.empty(E.shape[:2])
    cfa[0::2, 0::2] = E[0::2, 0::2, 0]
    cfa[0::2, 1::2] = E[0::2, 1::2, 1]
    cfa[1::2, 0::2] = E[1::2, 0::2, 1]
    cfa[1::2, 1::2] = E[1::2, 1::2, 2]
    if rng is None:
        y = np.clip(cfa, 0.0, FULL_WELL) / FULL_WELL
    else:
        e = rng.poisson(np.maximum(cfa / gain, 0.0)) + rng.normal(0.0, READ_NOISE_E, cfa.shape)
        y = e * gain / FULL_WELL
    clipped = float((y >= 1.0).mean())
    dn = np.round(y * (MAXDN - BLACK) + BLACK)
    if signed:
        return np.minimum(dn, MAXDN).astype(np.int32), clipped
    dn = np.clip(dn, 0, MAXDN)
    return dn.astype(np.uint16), clipped
