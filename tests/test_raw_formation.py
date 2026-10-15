"""Unit tests for isp/raw_formation.py, the E009 (C1) gain/noise model. Needs an ISETCam clone (ISETCAM_ROOT).
[ai-scaffold]

Run: python -m pytest -q tests/test_raw_formation.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ISETCAM = os.path.expanduser(os.environ.get("ISETCAM_ROOT", "~/research/ext/isetcam"))
if not os.path.isdir(ISETCAM):
    pytest.skip("ISETCam not found (set ISETCAM_ROOT)", allow_module_level=True)
from isp import raw_formation as rf  # noqa: E402

GAINS = {t: rf.gain_for(t * 1e-3) for t in (4, 8, 16, 32, 48)}


def test_gains_match_the_record():
    """G = 40.6 / 20.3 / 10.1 / 5.07 / 3.38 at 4 / 8 / 16 / 32 / 48 ms, and G proportional to 1/t."""
    assert [round(GAINS[t], 1) for t in (4, 8, 16, 32)] == [40.6, 20.3, 10.1, 5.1]
    assert GAINS[4] * 4 == pytest.approx(GAINS[48] * 48)


def test_noise_free_raw_is_independent_of_gain():
    """Without noise, exposure changes nothing but blur: the DN equal form_raw for every gain."""
    rng = np.random.default_rng(0)
    L = rng.uniform(0, 2e4, (32, 32, 3))
    cos4 = rf.cos4_map(32, 45.0)
    ref, _ = rf.form_raw(L, 1.0, cos4)
    for g in GAINS.values():
        dn, _ = rf.form_raw_gain(L, 1.0, cos4, g)
        assert np.array_equal(dn, ref)


def test_noise_anchor_at_4ms():
    """Total variance at normalised level 0.18 at 4 ms equals E005's lambda = 0.012 total (2.376e-3)."""
    n4 = rf.FULL_SCALE_4MS
    assert n4 == pytest.approx(246.5, abs=0.5)
    x = 0.18
    assert x / n4 + (rf.READ_NOISE_E / n4) ** 2 == pytest.approx(2.376e-3, rel=0.01)


def test_measured_variance_matches_the_model():
    """Flat scene through form_raw_gain at G(4 ms): per Bayer site, the measured variance of the normalised
    RAW matches shot + read, y / N4 + (10 e- / N4)^2, within 3% (quantisation is negligible)."""
    cos4 = np.ones((256, 256))
    L = np.ones((256, 256, 3))
    clean, _ = rf.form_raw_gain(L, 1.0, cos4, 1.0)
    t = 0.18 / ((clean[0::2, 1::2].mean() - rf.BLACK) / (rf.MAXDN - rf.BLACK))   # green at 0.18 of full scale
    clean, _ = rf.form_raw_gain(L, t, cos4, 1.0)
    noisy, _ = rf.form_raw_gain(L, t, cos4, GAINS[4], np.random.default_rng(1), signed=True)
    norm = lambda d: (d.astype(float) - rf.BLACK) / (rf.MAXDN - rf.BLACK)  # noqa: E731
    for sl in ((slice(0, None, 2), slice(0, None, 2)), (slice(0, None, 2), slice(1, None, 2)),
               (slice(1, None, 2), slice(1, None, 2))):
        y = norm(clean[sl]).mean()
        expected = y / rf.FULL_SCALE_4MS + (rf.READ_NOISE_E / rf.FULL_SCALE_4MS) ** 2
        assert np.var(norm(noisy[sl])) == pytest.approx(expected, rel=0.03)


def test_signed_keeps_values_below_zero():
    """signed=True removes only the lower clip: equal to the unsigned DN where those are > 0."""
    L = np.full((16, 16, 3), 1e-3)
    cos4 = rf.cos4_map(16, 45.0)
    a, _ = rf.form_raw_gain(L, 1.0, cos4, GAINS[4], np.random.default_rng(3), signed=True)
    b, _ = rf.form_raw_gain(L, 1.0, cos4, GAINS[4], np.random.default_rng(3))
    assert a.dtype == np.int32 and b.dtype == np.uint16
    pos = a > 0
    assert np.array_equal(a[pos], b[pos].astype(np.int32)) and (b[~pos] == 0).all()
