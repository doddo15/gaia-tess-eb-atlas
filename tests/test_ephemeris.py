import numpy as np
import pytest

from ebatlas import Ephemeris, gaia_time_to_btjd, btjd_to_gaia_time
from ebatlas.ephemeris import GAIA_TIME_OFFSET, BTJD_OFFSET


def test_tcb_tdb_offset_is_about_19s():
    t_gaia = np.array([2000.0])                       # ~2016
    naive = t_gaia + GAIA_TIME_OFFSET - BTJD_OFFSET   # ignores TCB vs TDB
    dt_s = (naive - gaia_time_to_btjd(t_gaia)) * 86400
    assert 18.0 < dt_s[0] < 21.0


def test_roundtrip():
    t = np.linspace(1666, 3671, 50)
    assert np.allclose(btjd_to_gaia_time(gaia_time_to_btjd(t)), t, atol=1e-9)


def test_phase_wrap_and_eclipse_masks():
    e = Ephemeris(period=2.0, t0=100.0, primary_width=0.1, secondary_width=0.1)
    t = 100.0 + 2.0 * np.array([0.0, 0.03, 0.2, 0.5, 0.97, 10.5])
    ph = e.phase(t)
    assert np.all((ph >= -0.5) & (ph < 0.5))
    assert list(e.in_primary(t)) == [True, True, False, False, True, False]
    assert list(e.in_secondary(t)) == [False, False, False, True, False, True]


def test_phase_sigma_grows_with_cycles():
    e = Ephemeris(period=1.0, t0=0.0, period_err=1e-5, t0_err=1e-4, primary_width=0.05)
    s = e.phase_sigma(np.array([0.0, -1000.0, -3000.0]))
    assert s[0] == pytest.approx(1e-4)
    assert s[0] < s[1] < s[2]
    assert e.is_reliable_at(0.0) and not e.is_reliable_at(-3000.0)


def test_secure_vs_possible():
    e = Ephemeris(period=1.0, t0=0.0, period_err=1e-5, primary_width=0.06)
    t = np.array([-2000.0 + 0.025])     # near edge, sigma = 0.02 in phase
    assert e.in_primary(t, "possible")[0] and not e.in_primary(t, "secure")[0]


def test_missing_width_raises():
    with pytest.raises(ValueError):
        Ephemeris(period=1.0, t0=0.0).in_primary(np.array([0.0]))
