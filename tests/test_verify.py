import numpy as np
from astropy.table import Table

from ebatlas import Ephemeris, btjd_to_gaia_time
from ebatlas.gaia_epochs import standardize_dr3
from ebatlas.synthetic import eb_model
from ebatlas.verify import period_agreement, tess_template, phase_offset_scan, verify_dr3

EPH = Ephemeris(period=2.5, t0=1500.0, period_err=2e-6, t0_err=1e-4,
                primary_width=0.06, secondary_phase=0.5, secondary_width=0.06)


def _tess(eph, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(eph.t0 - 13, eph.t0 + 13, 2 / 1440)
    return t, eb_model(eph, t, 0.30, 0.10) * (1 + rng.normal(0, 0.001, t.size))


def _dr3_long(eph_true, btjd, sid=7, seed=2, var_reject_deep=False):
    rng = np.random.default_rng(seed)
    f = eb_model(eph_true, btjd, 0.33, 0.11) * (1 + rng.normal(0, 0.003, btjd.size)) * 1e5
    deep = f < 0.9e5
    return Table({"source_id": np.full(btjd.size, sid), "band": ["G"] * btjd.size,
                  "time": btjd_to_gaia_time(btjd), "flux": f, "flux_error": np.full(btjd.size, 300.0),
                  "mag": np.full(btjd.size, 15.0), "rejected_by_photometry": np.zeros(btjd.size, bool),
                  "rejected_by_variability": deep if var_reject_deep else np.zeros(btjd.size, bool)})


def _gaia_times(n=150, seed=3):
    return np.sort(np.random.default_rng(seed).uniform(-130, 1860, n))


def test_period_agreement():
    assert period_agreement(2.0, 1 / 2.0)[1] == "match"
    assert period_agreement(2.0, 1 / 1.0)[1] == "half"
    assert period_agreement(2.0, 1 / 4.0)[1] == "double"
    assert period_agreement(2.0, 1 / 2.3)[1] == "mismatch"
    assert period_agreement(2.0, np.nan)[1] == "not_in_dr3_eb"


def test_offset_recovered():
    delta = 0.012
    true = Ephemeris(EPH.period, EPH.t0 + delta * EPH.period, primary_width=0.06, secondary_width=0.06)
    gaia = standardize_dr3(_dr3_long(true, _gaia_times()))
    t, f = _tess(EPH)
    res = phase_offset_scan(EPH, gaia, tess_template(EPH, t, f), span=0.05)
    assert res["constrained"]
    assert abs(res["best_shift"] - delta) < 3 * res["shift_err"] + 0.002
    assert 0.8 < res["depth_scale"] < 1.5          # Gaia depth differs from TESS: free scale


def test_zero_offset_consistent():
    gaia = standardize_dr3(_dr3_long(EPH, _gaia_times(seed=5)))
    t, f = _tess(EPH)
    res = phase_offset_scan(EPH, gaia, tess_template(EPH, t, f))
    assert res["constrained"] and abs(res["best_shift"]) < 0.004


def test_unconstrained_when_eclipses_missed():
    bt = _gaia_times(400, seed=9)
    ph = EPH.phase(bt)
    bt = bt[(np.abs(ph) > 0.1) & (np.abs(np.abs(ph) - 0.5) > 0.1)]
    gaia = standardize_dr3(_dr3_long(EPH, bt))
    t, f = _tess(EPH)
    assert not phase_offset_scan(EPH, gaia, tess_template(EPH, t, f))["constrained"]


def test_verify_dr3_offline(tmp_path):
    from astropy.table import Table as T
    cat = T({"eb_id": ["TIC1.01", "TIC2.01", "TIC3.01"], "tic_id": [1, 2, 3], "period": [2.5, 2.5, 2.5],
             "t0_btjd": [1500.0] * 3, "period_err": [2e-6] * 3, "t0_err": [1e-4] * 3,
             "primary_width": [np.nan] * 3, "secondary_phase": [np.nan] * 3, "secondary_width": [np.nan] * 3})
    xm = T({"eb_id": ["TIC1.01", "TIC2.01", "TIC3.01"], "dr3_source_id": [11, 22, 33],
            "dr3_has_epoch_photometry": ["True", "True", "False"], "dr3_eb_frequency": [1 / 2.5, np.nan, np.nan]})
    tables = {11: standardize_dr3(_dr3_long(EPH, _gaia_times(seed=1), sid=11, var_reject_deep=True)),
              22: standardize_dr3(_dr3_long(EPH, _gaia_times(seed=2), sid=22))}
    res = verify_dr3(cat, xm, tmp_path, fetcher=lambda ids: {i: tables[i] for i in ids if i in tables},
                     lc_getter=lambda tic: (*_tess(EPH), None))
    assert len(res) == 2                                         # TIC3 has no DR3 epoch photometry
    assert list(res["period_check"]) == ["match", "not_in_dr3_eb"]
    assert np.all(np.abs(res["meas_secondary_phase"] - 0.5) < 0.01)
    assert res["primary_var_rejected_in_eclipse"][0] >= 1           # flagged, but kept
    assert (tmp_path / "verification.csv").exists() and (tmp_path / "verification_summary.png").exists()


def test_verify_with_refit(tmp_path):
    """Catalog period slightly wrong; TESS spans 6 years. Refit should remove the Gaia-era drift."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("tf", pathlib.Path(__file__).parent / "test_ephemeris_fit.py")
    tf = importlib.util.module_from_spec(spec); spec.loader.exec_module(tf)
    from astropy.table import Table as T
    P, T0 = 1.7, 1330.0
    true = Ephemeris(P, T0, primary_width=0.05, secondary_phase=0.5, secondary_width=0.05)
    exps = {1: 1800.0, 27: 600.0, 54: 200.0, 80: 200.0}
    lc = tf._lc(true, list(exps), exps, seed=3)
    rng = np.random.default_rng(4)
    gb = np.sort(rng.uniform(-130, 1100, 300))
    gaia = standardize_dr3(_dr3_long(true, gb, sid=1))
    cat = T({"eb_id": ["TIC1.01"], "tic_id": [1], "period": [P + 2e-5], "t0_btjd": [T0], "period_err": [2e-6],
             "t0_err": [2e-4], "primary_width": [np.nan], "secondary_phase": [np.nan], "secondary_width": [np.nan]})
    xm = T({"eb_id": ["TIC1.01"], "dr3_source_id": [1], "dr3_has_epoch_photometry": [True], "dr3_eb_frequency": [1 / P]})
    res = verify_dr3(cat, xm, tmp_path, fetcher=lambda ids: {1: gaia}, lc_getter=lambda tic: lc, pages=True, n_boot=3)
    r = res[0]
    assert r["refit_status"] == "ok" and r["refit_n_sectors"] == 4
    assert abs(r["refit_period"] - P) < 5 * r["refit_period_err"]
    assert abs(r["catalog_drift_at_gaia"]) > 0.004                 # expected ~0.0058 for a 2e-5 d period error
    assert r["offset_constrained"] and abs(r["offset_best_shift"]) < 3 * r["offset_shift_err"] + 5e-4
    assert (tmp_path / "TIC1.01_oc.png").exists()
