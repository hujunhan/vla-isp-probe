"""Simulated sensor + ISP applied to rendered robot-camera frames (E004, decisions/006).

Pipeline (one frame, uint8 HxWx3 in, uint8 HxWx3 out):

  render (uint8)
    -> UNPROCESS   [srgb domain only] inverse tone curve, inverse gamma        -> linear RGB
               -> sRGB->camera colour matrix (inverse of the ISP's CCM)        -> camera RGB
               -> inverse white-balance and brightness gains                   -> "RAW-like" camera RGB
    -> SENSOR      exposure scale (over-exposure only) -> clip at saturation
               -> RGGB Bayer mosaic -> shot + read noise -> quantize to bit_depth
    -> ISP         white-balance and brightness gains -> edge-aware demosaic
               -> camera->sRGB colour matrix -> [srgb only] gamma, tone curve  -> uint8
               -> false-colour suppression (chroma median) -> NLM denoise (knob B)

The unprocess/process steps and all constants follow Brooks et al., "Unprocessing Images for Learned
Raw Denoising" (CVPR 2019; lit #30, Sec. 3-4), and its reference code (google-research/unprocessing,
Apache-2.0). Differences from that code, all deliberate:
  * camera parameters are FIXED at the centre of Brooks' sampling ranges (not random per image), so
    conditions differ only in the knob being swept;
  * the forward ISP re-applies the brightness gain and the tone curve, so that with no noise and no
    mosaic the pipeline is (near-)identity on the render the policy was trained on;
  * demosaic is OpenCV's edge-aware method (Brooks: bilinear), followed by chroma-median false-colour
    suppression; a denoiser is added (Brooks has none, it trains one). Both changes were made after a
    visual check showed bilinear-only output and a weak bilateral denoiser looked worse than a real
    camera, which would have biased the probe toward finding an effect (E004 plan, change log).
Parameters come from public literature only (intake/ip-check.md).

[ai-scaffold] — core experiment code written by the advisor at the owner's request (decisions/006
D5b); owner explain-back owed before E004 results count.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------- constants (Brooks et al. code)

# XYZ -> camera matrices of the four Darmstadt cameras (Brooks unprocess.py, random_ccm).
_XYZ2CAMS = np.array([
    [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
    [[0.4913, -0.0541, -0.0202], [-0.613, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
    [[0.838, -0.263, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
    [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.097, 0.1581, 0.5181]],
])
_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]])


def _make_rgb2cam():
    xyz2cam = _XYZ2CAMS.mean(axis=0)              # equal-weight convex combination (fixed, not random)
    rgb2cam = xyz2cam @ _RGB2XYZ
    return rgb2cam / rgb2cam.sum(axis=1, keepdims=True)   # rows sum to 1: grey stays grey


RGB2CAM = _make_rgb2cam()
CAM2RGB = np.linalg.inv(RGB2CAM)
RGB_GAIN = 1.0 / 0.8      # brightness gain: Brooks samples 1/N(0.8, 0.1); we use the mean
RED_GAIN = 2.15           # white balance: Brooks samples U(1.9, 2.4)
BLUE_GAIN = 1.7           # white balance: Brooks samples U(1.5, 1.9)


def read_noise_for(shot_noise):
    """Read-noise variance paired with a shot-noise level (median of Brooks' fitted log-log line)."""
    return float(np.exp(2.18 * np.log(shot_noise) + 1.20)) if shot_noise > 0 else 0.0


# ---------------------------------------------------------------- configuration

@dataclass(frozen=True)
class CameraConfig:
    input_domain: str = "srgb"   # "srgb": render treated as display-referred camera output (D6 primary)
                                 # "linear": render treated as linear light (D6 sensitivity check)
    mosaic: bool = True          # False only for the Block-0a round-trip test
    shot_noise: float = 0.0      # lambda_shot (variance per unit signal); 0 = noiseless
    read_noise: float = -1.0     # variance; -1 = derive from shot_noise via read_noise_for()
    exposure_ev: float = 0.0     # > 0: over-exposure (signal x 2^ev, clipped, NOT compensated)
    denoise: str = "default"     # "off" | "default" | "strong" (knob B)
    denoise_h: float = 0.0       # NLM strength for "default" at this noise level (from calibration,
                                 # results/e004/denoise_calibration.json); "strong" uses 2x
    bit_depth: int = 12

    @property
    def read_var(self):
        return read_noise_for(self.shot_noise) if self.read_noise < 0 else self.read_noise


# ---------------------------------------------------------------- unprocess (render -> RAW-like)

def inverse_smoothstep(x):
    """Inverse of the tone curve 3x^2 - 2x^3."""
    x = np.clip(x, 0.0, 1.0)
    return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * x) / 3.0)


def gamma_expansion(x):
    return np.maximum(x, 1e-8) ** 2.2


def apply_ccm(img, ccm):
    """Per-pixel 3x3 colour matrix: out[..., i] = sum_j ccm[i, j] * img[..., j]."""
    return img @ ccm.T


def safe_invert_gains(img):
    """Divide out brightness + white-balance gains, but fade the division out near saturation so that
    bright pixels are not pulled below 1 (Brooks Sec. 3.3, Eq. 6 / safe_invert_gains)."""
    gains = np.array([1.0 / RED_GAIN, 1.0, 1.0 / BLUE_GAIN]) / RGB_GAIN
    gray = img.mean(axis=-1, keepdims=True)
    mask = (np.maximum(gray - 0.9, 0.0) / 0.1) ** 2.0
    safe = np.maximum(mask + (1.0 - mask) * gains, gains)
    return img * safe


def unprocess(rgb01, domain):
    x = rgb01
    if domain == "srgb":
        x = gamma_expansion(inverse_smoothstep(x))
    x = apply_ccm(x, RGB2CAM)
    x = safe_invert_gains(x)
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------- sensor

def mosaic_rggb(img):
    """Keep one colour per pixel in an RGGB pattern; returns a single-channel HxW CFA image."""
    cfa = np.empty(img.shape[:2], dtype=img.dtype)
    cfa[0::2, 0::2] = img[0::2, 0::2, 0]   # R
    cfa[0::2, 1::2] = img[0::2, 1::2, 1]   # G (red rows)
    cfa[1::2, 0::2] = img[1::2, 0::2, 1]   # G (blue rows)
    cfa[1::2, 1::2] = img[1::2, 1::2, 2]   # B
    return cfa


def add_sensor_noise(signal, shot_noise, read_var, rng):
    """Heteroscedastic Gaussian approximation of Poisson shot + Gaussian read noise:
    variance = signal * shot_noise + read_var (Brooks add_noise)."""
    if shot_noise <= 0 and read_var <= 0:
        return signal
    var = np.maximum(signal, 0.0) * shot_noise + read_var
    return signal + rng.standard_normal(signal.shape) * np.sqrt(var)


def quantize(x, bits):
    levels = 2 ** bits - 1
    return np.round(np.clip(x, 0.0, 1.0) * levels) / levels


# ---------------------------------------------------------------- ISP (RAW-like -> render)

_CFA_MASKS = {}


def _masks(h, w):
    key = (h, w)
    if key not in _CFA_MASKS:
        r = np.zeros((h, w)); g = np.zeros((h, w)); b = np.zeros((h, w))
        r[0::2, 0::2] = 1; g[0::2, 1::2] = 1; g[1::2, 0::2] = 1; b[1::2, 1::2] = 1
        _CFA_MASKS[key] = (r, g, b)
    return _CFA_MASKS[key]


_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float64) / 4.0
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 4.0


def demosaic_bilinear(cfa):
    """Textbook bilinear CFA interpolation (kept for reference/ablation)."""
    r_m, g_m, b_m = _masks(*cfa.shape)
    border = cv2.BORDER_REFLECT_101
    r = cv2.filter2D(cfa * r_m, -1, _K_RB, borderType=border)
    g = cv2.filter2D(cfa * g_m, -1, _K_G, borderType=border)
    b = cv2.filter2D(cfa * b_m, -1, _K_RB, borderType=border)
    return np.stack([r, g, b], axis=-1)


def demosaic_ea(cfa):
    """OpenCV edge-aware demosaic on a 16-bit RGGB CFA (code verified on our frames: RGGB/BG give
    the highest round-trip PSNR; GBRG/GR are wrong-phase)."""
    u16 = np.round(np.clip(cfa, 0.0, 1.0) * 65535).astype(np.uint16)
    return cv2.cvtColor(u16, cv2.COLOR_BayerRGGB2RGB_EA).astype(np.float64) / 65535


def smoothstep(x):
    return 3.0 * x ** 2 - 2.0 * x ** 3


def gamma_compression(x):
    return np.maximum(x, 1e-8) ** (1.0 / 2.2)


def process_to_display(raw, cfg):
    """ISP up to the display-referred float image. `raw` is a CFA (HxW) or 3-channel camera RGB."""
    gains = np.array([RED_GAIN, 1.0, BLUE_GAIN]) * RGB_GAIN
    if raw.ndim == 2:                                   # white balance per CFA site, then demosaic
        r_m, g_m, b_m = _masks(*raw.shape)
        x = np.clip(raw * (r_m * gains[0] + g_m * gains[1] + b_m * gains[2]), 0.0, 1.0)
        x = demosaic_ea(x)
    else:
        x = np.clip(raw * gains, 0.0, 1.0)
    x = np.clip(apply_ccm(x, CAM2RGB), 0.0, 1.0)
    if cfg.input_domain == "srgb":
        x = smoothstep(gamma_compression(x))
    return x


def false_colour_suppression(u8):
    """Median filter on the chroma channels (YCrCb), luma untouched: the standard ISP fix for the
    colour zippering that demosaic + CCM produce on fine texture."""
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    ycc[..., 1] = cv2.medianBlur(ycc[..., 1], 5)
    ycc[..., 2] = cv2.medianBlur(ycc[..., 2], 5)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


def denoise(u8, cfg):
    """Knob B: non-local-means denoising in the YUV-like output domain (OpenCV
    fastNlMeansDenoisingColored, luma strength h, chroma strength h). 'default' uses the
    PSNR-optimal h for this noise level (calibrated, i.e. an ISP tuned for its gain); 'strong' = 2x
    (over-smoothing); 'off' = none."""
    if cfg.denoise == "off" or cfg.denoise_h <= 0:
        return u8
    h = cfg.denoise_h * {"default": 1.0, "strong": 2.0}[cfg.denoise]
    return cv2.fastNlMeansDenoisingColored(u8, None, h, h, 5, 15)


def to_u8(x):
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- full camera

def simulate(frame_u8, cfg, rng):
    """Render (uint8 HxWx3) -> simulated camera output (uint8 HxWx3)."""
    x = unprocess(frame_u8.astype(np.float64) / 255.0, cfg.input_domain)
    x = np.clip(x * 2.0 ** cfg.exposure_ev, 0.0, 1.0)      # exposure; sensor saturates at 1.0
    if cfg.mosaic:
        x = mosaic_rggb(x)
    x = add_sensor_noise(x, cfg.shot_noise, cfg.read_var, rng)
    x = quantize(x, cfg.bit_depth)
    out = to_u8(process_to_display(x, cfg))
    if cfg.mosaic:
        out = false_colour_suppression(out)
    return denoise(out, cfg)


# ---------------------------------------------------------------- image-space control (S4)

def image_space_gaussian(frame_u8, sigma, rng):
    """Plain additive Gaussian noise on the 8-bit image (sigma in 0..255 units): the image-space
    degradation that S4 compares against at matched PSNR."""
    noisy = frame_u8.astype(np.float64) + rng.standard_normal(frame_u8.shape) * sigma
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def psnr(a_u8, b_u8):
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
