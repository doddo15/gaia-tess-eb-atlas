import numpy as np
from astropy.table import Table

from ebatlas import Ephemeris, btjd_to_gaia_time
from ebatlas.gaia_epochs import standardize_dr3
from ebatlas.synthetic import eb_model
from ebatlas.verify import chi2_breakdown, phase_offset_scan
from ebatlas.templates import tess_template
from ebatlas.noise import noise_curve, interp_curve
from ebatlas.coverage import audit_target, observed_sectors

EPH = Ephemeris(1.3, 1500.0, primary_width=0.06, secondary_phase=0.5, secondary_width=0.06)


def _tess(seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(1487, 1513, 2 / 1440)
    return t, eb_model(EPH, t, 0.30, 0.12) * (1 + rng.normal(0, 0.0008, t.size))


def _gaia(d=(0.33, 0.12), noise=0.003, quoted=0.003, n=400, seed=2):
    rng = np.random.default_rng(seed)
    bt = np.sort(rng.uniform(-130, 1100, n))
    f = eb_model(EPH, bt, *d) * (1 + rng.normal(0, noise, n)) * 1e5
    return standardize_dr3(Table({"source_id": np.full(n, 1), "band": ["G"] * n, "time": btjd_to_gaia_time(bt),
                                  "flux": f, "flux_error": np.full(n, quoted * 1e5), "mag": np.full(n, 13.0)}))


def _diag(gaia):
    t, f = _tess()
    tm = tess_template(EPH, t, f)
    off = phase_offset_scan(EPH, gaia, tm)
    return chi2_breakdown(EPH, gaia, tm, off["best_shift"])


def test_underestimated_errors_show_out_of_eclipse():
    d = _diag(_gaia(d=(0.33, 0.132), noise=0.009, quoted=0.003))           # true scatter 3x the quoted error
    assert 2.5 < d["oot_scatter_ratio"] < 3.5
    assert 6 < d["chi2_red_oot"] < 12 and d["n_in"] > 10


def test_bandpass_mismatch_shows_in_eclipse():
    d = _diag(_gaia(d=(0.33, 0.25)))                       # Gaia secondary ~2x deeper relative to TESS
    assert d["chi2_red_in"] > 5 * d["chi2_red_oot"]
    assert 0.8 < d["oot_scatter_ratio"] < 1.25


def test_correct_model_and_errors():
    d = _diag(_gaia(d=(0.33, 0.132)))                      # both eclipses 10% deeper: one scale fits

    assert 0.6 < d["chi2_red_in"] < 1.8 and 0.7 < d["chi2_red_oot"] < 1.4


def test_noise_curve_bounds():
    g = np.repeat(np.arange(8, 20, 1.0), 50)
    t = Table({"phot_g_mean_mag": g, "phot_g_mean_flux": np.ones_like(g),
               "phot_g_mean_flux_error": 0.001 / np.sqrt(9 * 40) * np.ones_like(g),
               "phot_g_n_obs": np.full(g.size, 360.0), "matched_transits": np.full(g.size, 40.0)})
    c = noise_curve(t, bins=np.arange(8, 20.5, 1.0))
    lo, up = interp_curve(c, 12.5)
    assert abs(up[0] - 0.001) < 1e-6 and abs(lo[0] - 0.001 / 3) < 1e-6


def test_audit_target_logic():
    prods = [{"sector": 11, "author": "SPOC", "exptime": 120.0}, {"sector": 38, "author": "QLP", "exptime": 600.0},
             {"sector": 38, "author": "TESS-SPOC", "exptime": 600.0}, {"sector": 64, "author": "CDIPS", "exptime": 200.0}]
    st = Table({"sector": [11, 38], "status": ["ok", "no_eclipse"]})
    r = audit_target([11, 38, 64, 90], prods, st)
    assert r["n_with_product"] == 2 and r["missing_sectors"] == "90" and r["other_pipeline_only"] == "64"
    assert r["n_timed"] == 1 and r["n_no_eclipse"] == 1 and r["last_product"] == 38


def test_tess_point_real_target():
    secs = observed_sectors(212.817308, -30.884359, max_sector=70)   # TEBC's first entry; TEBC lists only sector 11
    assert 11 in secs and 38 in secs and all(s <= 70 for s in secs)


def test_sector_status_reasons():
    from ebatlas.ephemeris_fit import sector_timing
    eph = Ephemeris(30.0, 1500.0, primary_width=0.01, secondary_width=0.01)
    t = np.arange(1502, 1512, 10 / 1440)                       # phases 0.07-0.40: misses both eclipses
    f = eb_model(eph, t, 0.3, 0.1) * (1 + np.random.default_rng(0).normal(0, 1e-3, t.size))
    tt = np.arange(1490, 1531, 10 / 1440)
    tm = tess_template(eph, tt, eb_model(eph, tt, 0.3, 0.1))
    assert sector_timing(eph, tm, t, f, 0.0, 0.02)["status"] == "no_eclipse"
