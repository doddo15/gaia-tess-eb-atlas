import numpy as np
from astropy.io.votable import from_table, writeto

from ebatlas import Ephemeris
from ebatlas.dr4_photometry import read_epoch_photometry, standardize_dr4, per_transit_g_noise_forecast
from ebatlas.synthetic import make_dr4_epoch_photometry
from ebatlas.plotting import summarize

EPH = Ephemeris(period=2.71828, t0=1500.0, period_err=3e-6, t0_err=2e-4,
                primary_width=0.045, secondary_width=0.045)


def test_standardize_columns_and_flags():
    raw = make_dr4_epoch_photometry(EPH, 123, seed=4)
    std = standardize_dr4(raw)
    n = len(raw)
    assert set(std["band"]) == {"G", "BP", "RP"} and len(std) == 3 * n
    g = std[std["band"] == "G"]; bp = std[std["band"] == "BP"]
    assert g["rejected"].sum() == 1 and g["n_ccd"][5] == 0          # g_n_obs == 0
    assert bp["rejected"].sum() == 1                                # NaN BP flux
    assert set(np.unique(g["fov"])) == {1, 2}
    # times are barycentric already: BTJD = BJD_TDB - 2457000, ~19 s TCB-TDB offset
    dt = (np.asarray(raw["g_obs_time"]) + 2455197.5 - 2457000.0 - np.asarray(g["btjd"])) * 86400
    assert np.all((dt > 17) & (dt < 22))


def test_votable_roundtrip(tmp_path):
    raw = make_dr4_epoch_photometry(EPH, 456, seed=4)
    p = tmp_path / "ep.xml"
    writeto(from_table(raw), str(p))
    d = read_epoch_photometry(p)
    assert list(d) == [456] and len(d[456]) == 3 * len(raw)


def test_summary_counts_gaia_outliers():
    std = standardize_dr4(make_dr4_epoch_photometry(EPH, 1, seed=4))
    s = summarize(EPH, std)
    assert s["n_primary_secure"] >= 1
    assert s["primary_gaia_outlier_transits"] >= 1
    assert s["primary_depth_snr"] > 10


def test_noise_forecast_monotonic():
    s = per_transit_g_noise_forecast([10, 15, 20])
    assert np.all(np.diff(s) > 0) and 0.0005 < s[1] < 0.02


def test_dr4_crossmatch_resolution():
    from astropy.table import Table, MaskedColumn
    from ebatlas.crossmatch import resolve_dr4_matches, dr4_crossmatch_query
    q = dr4_crossmatch_query()
    assert "tap_upload.ids" in q and "has_epoch_photometry" in q
    t = Table({"dr3_source_id": [1, 1, 2], "dr4_source_id": [11, 12, 21],
               "angular_distance": [0.4, 0.05, 0.1], "magnitude_difference": [0.1, 0.0, 0.0],
               "has_epoch_photometry": ["true", "false", "true"], "has_epoch_astrometry": [True] * 3,
               "gaia_source_id": MaskedColumn([11, 12, 0], mask=[False, False, True]),
               "vari_eb_id": MaskedColumn([0, 12, 0], mask=[True, False, True])})
    r = resolve_dr4_matches(t)
    assert list(r["dr4_source_id"]) == [12, 21] and list(r["n_dr4_candidates"]) == [2, 1]
    assert list(r["in_gaia_source"]) == [True, False] and list(r["in_dr4_vari_eb"]) == [True, False]
    assert list(r["has_epoch_photometry"]) == [False, True]
