import numpy as np

from ebatlas import Ephemeris
from ebatlas.ephemeris_fit import refit_ephemeris, sector_qc, linear_fit
from ebatlas.errors import jitter_ml
from ebatlas.synthetic import eb_model
from ebatlas.tess import TessLC

STARTS = {1: 1325.3, 14: 1683.4, 27: 2036.3, 40: 2390.7, 54: 2768.0, 67: 3125.0, 80: 3480.0}
P, T0 = 1.2345678, 1330.1
TRUE = Ephemeris(P, T0, primary_width=0.05, secondary_phase=0.5, secondary_width=0.05)


def _sector(s, eph=TRUE, cad=600.0, depth=(0.3, 0.12), noise=0.002, author="TESS-SPOC", seed=None, dt=0.0):
    rng = np.random.default_rng(s if seed is None else seed)
    t = np.arange(STARTS[s], STARTS[s] + 26.0, cad / 86400)
    t = t[(t - STARTS[s] < 12.5) | (t - STARTS[s] > 13.5)]
    sub = (np.arange(5) - 2) / 5 * cad / 86400
    f = np.mean([eb_model(eph, t + u - dt, *depth) for u in sub], axis=0) * (1 + rng.normal(0, noise, t.size))
    n = t.size
    return (t, f, np.full(n, noise), np.full(n, s), np.full(n, cad), np.full(n, author, dtype=object))


def _lc(parts):
    return TessLC(*[np.concatenate([p[k] for p in parts]) for k in range(6)])


def test_sector_qc():
    good = [_sector(s) for s in (1, 14, 27, 40)]
    t, f, e, sec, exp, au = _sector(54)
    garbage = (t, (f - 1.0) * 1e3 / 1e-3, e, sec, exp, au)          # "zero-centred" sector
    noisy = _sector(67, noise=0.05)
    lc = _lc(good + [garbage, noisy])
    qc = sector_qc(lc.t, lc.flux, lc.sector, lc.exptime)
    assert qc == {54: "bad_normalisation", 67: "noisy"}


def test_wide_retry_recovers_bad_catalog_period():
    cat = Ephemeris(P + 2e-4, T0, period_err=2e-6, t0_err=2e-4)     # 100x worse than its claimed error
    res = refit_ephemeris(cat, _lc([_sector(s) for s in (1, 40, 80)]))
    assert res["status"] == "ok" and res["n_sectors"] == 3 and res["n_wide"] >= 1
    assert abs(res["ephem"].period - P) < 1e-6


def test_diluted_pipeline_sectors_are_aligned():
    parts = [_sector(s, author="SPOC", cad=120.0) for s in (1, 27, 54)]
    parts += [_sector(s, author="QLP", depth=(0.15, 0.06)) for s in (14, 40, 67, 80)]   # half the depth
    res = refit_ephemeris(Ephemeris(P, T0, 1e-6, 1e-4), _lc(parts))
    tab = res["timings"]
    assert res["n_sectors"] == 7
    qlp = np.asarray(tab["depth_scale"][tab["author"] == "QLP"], float)
    assert np.all(np.abs(qlp - 0.5) < 0.06)
    assert np.all(np.abs(tab["O_C_min"]) < 0.3)                   # < 20 s
    assert res["depth_scale_spread"] < 0.05


def test_timing_jitter_recovered_without_false_period_change():
    rng = np.random.default_rng(7)
    offs = rng.normal(0, 60.0, len(STARTS)) / 86400                # 60 s random timing scatter per sector
    parts = [_sector(s, dt=o) for s, o in zip(STARTS, offs)]
    res = refit_ephemeris(Ephemeris(P, T0, 1e-6, 1e-4), _lc(parts))
    assert 25 < res["jitter_s"] < 110
    assert abs(res["quad_snr"]) < 4
    assert abs(res["ephem"].period - P) < 4 * res["ephem"].period_err


def test_single_outlier_sector_clipped():
    parts = [_sector(s, dt=(10 / 1440 if s == 40 else 0.0)) for s in STARTS]   # sector 40 off by 10 min
    res = refit_ephemeris(Ephemeris(P, T0, 1e-6, 1e-4), _lc(parts))
    st = res["sector_status"]
    assert st["status"][st["sector"] == 40][0] == "outlier_timing"
    assert res["n_sectors"] == 6 and res["n_outliers"] == 1


def test_jitter_ml():
    rng = np.random.default_rng(1)
    e = np.full(4000, 1.0)
    assert jitter_ml(rng.normal(0, 1.0, 4000), e) < 0.15
    assert abs(jitter_ml(rng.normal(0, np.hypot(1.0, 2.0), 4000), e) - 2.0) < 0.15
