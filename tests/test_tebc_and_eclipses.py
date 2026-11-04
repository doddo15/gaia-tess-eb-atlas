from pathlib import Path

import numpy as np

from ebatlas import Ephemeris
from ebatlas.catalogs import load_catalog, row_to_ephemeris, PERIOD_ERR_FLOOR
from ebatlas.eclipses import measure_eclipses
from ebatlas.planning import phase_sigma_table, DR4_START_BTJD

SAMPLE = Path(__file__).parent / "data" / "tebc_sample.csv"


def test_tebc_reader_quirks():
    cat = load_catalog(SAMPLE, "tebc")
    assert len(cat) == 7
    r = cat[0]
    assert r["tic_id"] == 107548305 and r["eb_id"] == "TIC107548305.01"
    assert abs(r["t0_btjd"] - 1600.122599) < 1e-9          # already BTJD
    assert abs(r["glon"] - 322.487545) < 1e-6              # swapped columns fixed
    dup = cat[cat["tic_id"] == 441794509]
    assert list(dup["eb_id"]) == ["TIC441794509.01", "TIC441794509.02"]
    assert (~cat["has_ephemeris"]).sum() == 2   # missing bjd0; missing period_uncert
    assert np.isnan(cat["morph_coeff"]).sum() >= 1             # -1 sentinel -> NaN
    assert np.all(cat["period_err"][cat["has_ephemeris"]] >= PERIOD_ERR_FLOOR)
    assert np.all(np.isnan(cat["secondary_phase"]))           # not assumed 0.5


def test_phase_sigma_table():
    cat = load_catalog(SAMPLE, "tebc")
    tab = phase_sigma_table(cat)
    ok = cat["has_ephemeris"]
    assert np.all(tab["sigma_phase_start"][ok] >= tab["sigma_phase_end"][ok])
    assert DR4_START_BTJD < -130


def _synthetic(phi2=0.5, w1=0.04, w2=0.04, d1=0.3, d2=0.1, seed=1):
    e = Ephemeris(period=3.1, t0=1500.0)
    rng = np.random.default_rng(seed)
    t = np.arange(1490, 1517, 2 / 1440)
    ph = e.phase(t)
    f = np.ones_like(t)
    for c, d, w in ((0.0, d1, w1), (phi2, d2, w2)):
        x = np.abs(((ph - c + 0.5) % 1) - 0.5) / (w / 2)
        f -= d * np.clip(1 - x**2, 0, None) ** 0.6
    return e, t, f * (1 + rng.normal(0, 0.002, t.size))


def test_measure_circular():
    e, t, f = _synthetic()
    m = measure_eclipses(e, t, f)
    assert abs(m["primary_width"] - 0.04) < 0.01
    assert abs(m["primary_depth"] - 0.3) < 0.03
    assert m["secondary_detected"] and abs(m["secondary_phase"] - 0.5) < 0.005
    assert not m["contact_like"]


def test_measure_eccentric_secondary():
    e, t, f = _synthetic(phi2=0.37, w2=0.03)
    m = measure_eclipses(e, t, f)
    assert abs(m["secondary_phase"] - 0.37) < 0.005
    assert abs(m["secondary_width"] - 0.03) < 0.01
    eph = row_to_ephemeris(load_catalog(SAMPLE, "tebc")[0],
                           primary_width=m["primary_width"],
                           secondary_phase=m["secondary_phase"],
                           secondary_width=m["secondary_width"])
    assert eph.secondary_phase == m["secondary_phase"]


def test_no_secondary():
    e, t, f = _synthetic(d2=0.0)
    assert not measure_eclipses(e, t, f)["secondary_detected"]


def test_ragged_summary_rows():
    from ebatlas.atlas import _rows_to_table
    t = _rows_to_table([{"eb_id": "a", "n": 1}, {"eb_id": "b", "ruwe": 1.2, "flag": True}])
    assert t.colnames == ["eb_id", "n", "ruwe", "flag"] and np.isnan(t["ruwe"][0]) and not t["flag"][0]
