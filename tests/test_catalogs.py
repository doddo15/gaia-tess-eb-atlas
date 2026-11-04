import numpy as np
from astropy.table import Table

from ebatlas.catalogs import load_catalog, row_to_ephemeris


def test_example_mapping(tmp_path):
    p = tmp_path / "cat.csv"
    Table({"tic_id": [1, 2], "period": [1.5, 3.0], "period_err": [1e-6, np.nan],
           "t0": [1400.0, 1500.0], "t0_err": [1e-4, 2e-4], "w1": [0.05, 0.03],
           "phi2": [0.5, 0.48], "w2": [0.05, np.nan]}).write(p, format="ascii.csv")
    cat = load_catalog(p, "example", fmt="ascii.csv")
    assert cat["tic_id"].dtype == np.int64
    e = row_to_ephemeris(cat[1])
    assert e.period_err == 0.0 and e.secondary_width is None and e.secondary_phase == 0.48
