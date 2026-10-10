"""Unit tests for isp/camera.py on real LIBERO renders (results/e004/frames.npz). [ai-scaffold]

Run: python -m pytest -q tests/test_camera.py   (from repo root; prints a PSNR table with -s)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from isp.camera import (CameraConfig, add_sensor_noise, image_space_gaussian, psnr,  # noqa: E402
                        read_noise_for, simulate)

FRAMES = np.load(os.path.join(os.path.dirname(__file__), "..", "results", "e004", "frames.npz"))
ALL = np.concatenate([FRAMES["agentview"], FRAMES["wrist"]])


@pytest.mark.parametrize("domain", ["srgb", "linear"])
def test_round_trip_without_mosaic_is_identity(domain):
    """Block 0a: unprocess -> process with no mosaic, no noise reproduces the render, except pixels
    near saturation where safe_invert_gains deliberately does not invert exactly."""
    cfg = CameraConfig(input_domain=domain, mosaic=False, bit_depth=16)
    for f in ALL:
        out = simulate(f, cfg, np.random.default_rng(0))
        bright = f.astype(float).mean(axis=-1) / 255.0 > 0.85
        err = np.abs(out.astype(int) - f.astype(int)).max(axis=-1)
        assert err[~bright].max() <= 1, f"max err {err[~bright].max()} on non-bright pixels"


def test_clean_camera_is_isp_quality():
    """Block 0b reference: mosaic + EA demosaic + false-colour suppression, no noise."""
    cfg = CameraConfig()
    p = [psnr(f, simulate(f, cfg, np.random.default_rng(0))) for f in ALL]
    assert np.mean(p) > 32.0 and min(p) > 28.0, f"clean-camera PSNR mean {np.mean(p):.1f}, min {min(p):.1f}"


def test_noise_variance_matches_model():
    rng = np.random.default_rng(1)
    signal = np.full((400, 400), 0.3)
    shot, read = 0.01, read_noise_for(0.01)
    noisy = add_sensor_noise(signal, shot, read, rng)
    assert abs(noisy.var() / (0.3 * shot + read) - 1) < 0.02


def test_read_noise_line():
    # Brooks' fitted line at the top of the shot-noise range: exp(2.18*ln(0.012)+1.20)
    assert abs(read_noise_for(0.012) - np.exp(2.18 * np.log(0.012) + 1.20)) < 1e-12
    assert read_noise_for(0.0) == 0.0


def test_deterministic_given_seed():
    cfg = CameraConfig(shot_noise=0.005)
    a = simulate(ALL[0], cfg, np.random.default_rng(42))
    b = simulate(ALL[0], cfg, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_noise_lowers_psnr_monotonically():
    clean_cfg = CameraConfig()
    levels = [1e-4, 1e-3, 5e-3, 1.2e-2, 3e-2, 6e-2]
    means = []
    for lam in levels:
        cfg = CameraConfig(shot_noise=lam, denoise="off")
        vals = [psnr(simulate(f, clean_cfg, np.random.default_rng(0)), simulate(f, cfg, np.random.default_rng(i)))
                for i, f in enumerate(ALL)]
        means.append(np.mean(vals))
    print("\nshot_noise -> mean PSNR vs clean camera (denoise off):")
    for lam, m in zip(levels, means):
        print(f"  {lam:8.4g}  {m:6.2f} dB")
    assert all(a > b for a, b in zip(means, means[1:]))


def test_image_space_noise_psnr():
    rng = np.random.default_rng(0)
    f = ALL[0]
    assert abs(psnr(f, image_space_gaussian(f, 5.0, rng)) - 20 * np.log10(255 / 5.0)) < 0.5
