"""Synthetic tests run everywhere; tests on the DR4 pre-release sample run only
if tests/private/dr4_rc3_epoch_astrometry_12src.xml exists (gitignored)."""
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from ebatlas.dr4_astrometry import (decode_transit_id, fit_five_parameter_al,
                                    astrometric_periodogram, orbit_grid_fit, _kepler_XY)

PRIVATE = Path(__file__).parent / "private" / "dr4_rc3_epoch_astrometry_12src.xml"


def _fake_obs(n_tr=70, seed=0, params=(1.0, -2.0, 5.0, 30.0, -12.0), orbit=None):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(-1.4, 4.0, n_tr))
    psi = rng.uniform(0, 2 * np.pi, n_tr)
    f = rng.uniform(-1, 1, n_tr)
    t, psi, f = np.repeat(t, 9), np.repeat(psi, 9), np.repeat(f, 9)
    s, c = np.sin(psi), np.cos(psi)
    da, dd, plx, pa, pd = params
    w = s * (da + pa * t) + c * (dd + pd * t) + plx * f
    if orbit:
        P, e, T0, A, B, F, G = orbit
        X, Y = _kepler_XY(t, P, e, T0)
        w = w + s * (B * X + G * Y) + c * (A * X + F * Y)
    err = np.full(t.size, 0.1)
    return Table({"source_id": np.ones(t.size, int), "transit_id": np.repeat(np.arange(n_tr), 9),
                  "t_years": t, "w": w + rng.normal(0, 0.1, t.size), "w_err": err, "psi": psi, "f_al": f})


def test_decode_transit_id():
    fov = 1; row = 5; ac = 1234; obmt_units = 123456789
    tid = (obmt_units << 17) | ((fov - 1) << 15) | (row << 12) | ac
    d = decode_transit_id([tid])
    assert d["fov"][0] == fov and d["ccd_row"][0] == row and d["ac_pixel_af1"][0] == ac
    assert d["obmt_ns"][0] == 204800 * obmt_units


def test_five_parameter_recovery():
    f = fit_five_parameter_al(_fake_obs())
    p = f["params"]
    assert abs(p["parallax"] - 5.0) < 0.02 and abs(p["pmra"] - 30.0) < 0.02 and abs(p["pmdec"] + 12.0) < 0.02
    assert 0.9 < f["uwe"] < 1.1


def test_orbit_recovery():
    obs = _fake_obs(orbit=(400.0, 0.4, 0.3, 1.5, -0.8, 0.6, 1.2))
    assert fit_five_parameter_al(obs)["uwe"] > 5
    d, _ = astrometric_periodogram(obs, np.array([150.0, 400.0, 900.0]))
    assert np.argmax(d) == 1
    best = orbit_grid_fit(obs, np.linspace(300, 500, 41), eccs=(0.0, 0.2, 0.4, 0.6), n_t0=20)
    assert abs(best["period"] - 400) < 15 and best["e"] == 0.4 and best["uwe"] < 1.5


@pytest.mark.skipif(not PRIVATE.exists(), reason="DR4 pre-release sample not present")
def test_private_sample():
    from ebatlas.dr4_astrometry import read_epoch_astrometry, transit_times, al_observations
    tab = read_epoch_astrometry(PRIVATE)
    tt = transit_times(tab)
    assert len(np.unique(tt["source_id"])) == 12
    assert -140 < tt["btjd"].min() and tt["btjd"].max() < 1870        # inside DR4 window
    known = ~tt["bary_corr_estimated"]
    assert np.all(np.abs(tt["bary_corr_s"][known]) < 375)                 # data model: |corr| < ~370 s
    obs = al_observations(tab)
    for sid in np.unique(obs["source_id"]):
        assert 0.8 < fit_five_parameter_al(obs[obs["source_id"] == sid])["uwe"] < 1.3
