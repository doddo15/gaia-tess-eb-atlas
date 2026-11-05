import numpy as np

from ebatlas import Ephemeris
from ebatlas.ephemeris_fit import refit_ephemeris, linear_fit
from ebatlas.synthetic import eb_model
from ebatlas.tess import TessLC, select_products

SECTOR_STARTS = {1: 1325.3, 14: 1683.4, 27: 2036.3, 40: 2390.7, 54: 2768.0, 67: 3125.0, 80: 3480.0}


def _lc(true, sectors, exptimes, noise=0.002, seed=0, d=(0.3, 0.12)):
    rng = np.random.default_rng(seed)
    parts = []
    for s in sectors:
        cad = exptimes[s] / 86400.0
        t = np.arange(SECTOR_STARTS[s], SECTOR_STARTS[s] + 26.0, cad)
        t = t[(t - SECTOR_STARTS[s] < 12.5) | (t - SECTOR_STARTS[s] > 13.5)]
        sub = (np.arange(7) - 3) / 7 * cad            # integrate over the exposure
        f = np.mean([eb_model(true, t + u, *d) for u in sub], axis=0) * (1 + rng.normal(0, noise, t.size))
        parts.append((t, f, np.full(t.size, noise), np.full(t.size, s), np.full(t.size, exptimes[s]),
                      np.full(t.size, "SPOC" if exptimes[s] == 120 else "TESS-SPOC", dtype=object)))
    return TessLC(*[np.concatenate([p[k] for p in parts]) for k in range(6)])


def test_select_products_priority():
    rows = [{"index": 0, "sector": 5, "author": "QLP", "exptime": 1800.0},
            {"index": 1, "sector": 5, "author": "TESS-SPOC", "exptime": 1800.0},
            {"index": 2, "sector": 40, "author": "SPOC", "exptime": 20.0},
            {"index": 3, "sector": 40, "author": "SPOC", "exptime": 120.0},
            {"index": 4, "sector": 41, "author": "QLP", "exptime": 600.0},
            {"index": 5, "sector": 41, "author": "CDIPS", "exptime": 1800.0}]
    chosen = select_products(rows)
    assert [(r["sector"], r["index"]) for r in chosen] == [(5, 1), (40, 3), (41, 4)]


def test_refit_recovers_period_from_bad_catalog():
    """Catalog period off by 2e-5 d (a PM-only fit), 7 sectors over 6 years, mixed cadences."""
    P_true, T_true = 1.2345678, 1330.1
    true = Ephemeris(P_true, T_true, primary_width=0.04, secondary_phase=0.5, secondary_width=0.04)
    cat = Ephemeris(P_true + 2e-5, T_true + 0.0005, period_err=5e-6, t0_err=3e-4)
    exps = {1: 1800.0, 14: 1800.0, 27: 600.0, 40: 120.0, 54: 200.0, 67: 120.0, 80: 200.0}
    lc = _lc(true, list(exps), exps)
    res = refit_ephemeris(cat, lc)
    assert res["status"] == "ok" and res["n_sectors"] == 7
    e = res["ephem"]
    assert abs(e.period - P_true) < 5 * e.period_err and e.period_err < 1e-6
    assert abs(res["period_shift_sigma"]) > 3                  # catalog period clearly superseded
    assert abs(res["quad_snr"]) < 4                             # no period change injected
    assert np.all(np.abs(res["timings"]["O_C_min"]) < 4 * res["timings"]["T_err_min"] + 0.2)


def test_refit_detects_period_change():
    P0, T0 = 0.6, 1330.0
    exps = {s: 120.0 for s in SECTOR_STARTS}
    rng_parts = []
    # build sector by sector with a slowly increasing period (dP/dE = 2e-9 d/cycle)
    parts = []
    for s in SECTOR_STARTS:
        Emid = (SECTOR_STARTS[s] + 13 - T0) / P0
        Pnow = P0 + 2e-9 * Emid
        Tnow = T0 + P0 * Emid + 1e-9 * Emid**2 - Pnow * Emid
        parts.append(_lc(Ephemeris(Pnow, Tnow, primary_width=0.08, secondary_width=0.08), [s], exps, seed=s))
    lc = TessLC(*[np.concatenate([p[k] for p in parts]) for k in range(6)])
    res = refit_ephemeris(Ephemeris(P0, T0, 1e-6, 1e-4), lc)
    assert res["status"] == "ok" and res["quad_snr"] > 5


def test_one_sector_keeps_catalog_period():
    true = Ephemeris(2.0, 1330.0, primary_width=0.05, secondary_width=0.05)
    cat = Ephemeris(2.0, 1330.0, 1e-5, 1e-4)
    res = refit_ephemeris(cat, _lc(true, [1], {1: 120.0}))
    assert res["status"] == "one_sector" and res["ephem"].period == cat.period


def test_linear_fit_basic():
    E = np.arange(10); T = 100 + 2.5 * E + np.random.default_rng(1).normal(0, 1e-4, 10)
    f = linear_fit(E, T, np.full(10, 1e-4))
    assert abs(f["p"][1] - 2.5) < 1e-4
