import numpy as np
from astropy.table import Table

from ebatlas import Ephemeris, btjd_to_gaia_time
from ebatlas.gaia_epochs import parse_dr3_result, standardize_dr3
from ebatlas.plotting import atlas_page, normalize_band, MAX_NORM_FLUX
from ebatlas.synthetic import eb_model


def _wide_row(sid, n=6):
    t = np.linspace(1700, 1710, n)
    row = {"source_id": sid, "g_transit_time": t, "g_transit_flux": np.full(n, 1e4),
           "g_transit_flux_error": np.full(n, 10.0), "g_transit_mag": np.full(n, 15.0),
           "g_transit_n_obs": np.full(n, 9), "bp_obs_time": t, "bp_flux": np.r_[np.full(n - 1, 5e3), 6e4],
           "bp_flux_error": np.full(n, 10.0), "bp_mag": np.full(n, 15.3),
           "rp_obs_time": t, "rp_flux": np.full(n, 4e3), "rp_flux_error": np.full(n, 10.0), "rp_mag": np.full(n, 14.8),
           "photometry_flag_bp_reject": np.r_[np.zeros(n - 1, bool), True],
           "photometry_flag_rp_reject": np.zeros(n, bool), "photometry_flag_bp_unavailable": np.zeros(n, bool),
           "photometry_flag_rp_unavailable": np.zeros(n, bool),
           "variability_flag_g_reject": np.zeros(n, bool)}
    for k in range(1, 10):                    # transit 0: every AF CCD rejected -> G rejected
        row[f"photometry_flag_af{k}_reject"] = np.r_[True, np.zeros(n - 1, bool)]
        row[f"photometry_flag_af{k}_unavailable"] = np.zeros(n, bool)
    return row


def test_gaia_datamodel_flags_applied():
    out = parse_dr3_result({"EPOCH_PHOTOMETRY_DATAMODEL_GAIA.xml": [Table(rows=[_wide_row(5)])]})[5]
    bp, g = out[out["band"] == "BP"], out[out["band"] == "G"]
    assert list(bp["rejected"]) == [False] * 5 + [True]       # the 12x BP point is DR3-rejected
    assert bool(g["rejected"][0]) and not g["rejected"][1:].any()


def test_standard_format_source_from_filename():
    n = 4
    t = Table({"transit_id": np.arange(n), "g_transit_time": np.linspace(1700, 1701, n),
               "g_transit_flux": np.full(n, 1e4), "g_transit_flux_error": np.full(n, 10.0),
               "g_transit_mag": np.full(n, 15.0), "bp_obs_time": np.linspace(1700, 1701, n),
               "bp_flux": np.full(n, 5e3), "bp_flux_error": np.full(n, 10.0), "bp_mag": np.full(n, 15.3),
               "variability_flag_g_reject": [False, True, False, False],
               "rejected_by_photometry": [False, False, False, True]})
    out = parse_dr3_result({"EPOCH_PHOTOMETRY_Gaia_DR3_2037415700780608128.xml": [t]})
    std = out[2037415700780608128]
    g = std[std["band"] == "G"]
    assert list(g["rejected"]) == [False, False, False, True]
    assert list(g["rejected_by_variability"]) == [False, True, False, False]


def test_brightening_flag_and_plot_limits():
    eph = Ephemeris(1.0, 0.0, primary_width=0.05, secondary_width=0.05)
    bt = np.linspace(-100, 1000, 60)
    f = eb_model(eph, bt, 0.3, 0.1) * 1e4
    f[7] = 12e4                                               # 12x, unflagged, like the RP points you saw
    tab = Table({"band": ["RP"] * 60, "btjd": bt, "time_gaia": btjd_to_gaia_time(bt), "flux": f,
                 "flux_err": np.full(60, 10.0), "mag": np.full(60, 15.0), "rejected": np.zeros(60, bool)})
    _, fn, _, rej = normalize_band(eph, tab, "RP")
    assert rej[7] and fn[7] > MAX_NORM_FLUX and rej.sum() == 1
    fig = atlas_page(eph, tab, bands=("RP",))
    lo, hi = fig.axes[0].get_ylim()
    assert hi < 1.2 and lo > 0.5                            # scale set by real data, not the outlier


def test_double_period_refold(tmp_path):
    """Contact binary with unequal minima, catalogued at half its period; DR3 reports 2P."""
    from ebatlas.verify import verify_dr3
    P2 = 0.34
    true = Ephemeris(P2, 1500.0, primary_width=0.3, secondary_phase=0.5, secondary_width=0.3)
    rng = np.random.default_rng(0)
    tt = np.arange(1490, 1516, 2 / 1440)
    tf = eb_model(true, tt, 0.30, 0.18, ellip=0.0) * (1 + rng.normal(0, 0.001, tt.size))
    gb = np.sort(rng.uniform(-130, 1100, 90))
    gf = eb_model(true, gb, 0.33, 0.19, ellip=0.0) * (1 + rng.normal(0, 0.003, gb.size)) * 1e5
    dr3 = standardize_dr3(Table({"source_id": np.full(gb.size, 9), "band": ["G"] * gb.size,
                                 "time": btjd_to_gaia_time(gb), "flux": gf, "flux_error": np.full(gb.size, 300.0)}))
    cat = Table({"eb_id": ["TIC9.01"], "tic_id": [9], "period": [P2 / 2], "t0_btjd": [1500.0],
                 "period_err": [1e-7], "t0_err": [1e-5], "primary_width": [np.nan],
                 "secondary_phase": [np.nan], "secondary_width": [np.nan]})
    xm = Table({"eb_id": ["TIC9.01"], "dr3_source_id": [9], "dr3_has_epoch_photometry": [True],
                "dr3_eb_frequency": [1 / P2]})
    res = verify_dr3(cat, xm, tmp_path, fetcher=lambda ids: {9: dr3}, lc_getter=lambda tic: (tt, tf, None),
                     pages=False, n_boot=3)
    assert res["period_check"][0] == "double"
    assert res["adopted_period_factor"][0] == 2
    assert res["offset_chi2_red"][0] < 0.9 * res["alt_offset_chi2_red"][0]


def test_tess_and_gaia_share_normalisation():
    from ebatlas.plotting import normalize_tess
    eph = Ephemeris(0.3, 1500.0, primary_width=0.3, secondary_phase=0.5, secondary_width=0.3)
    tt = np.arange(1490, 1516, 10 / 1440)
    tf = eb_model(eph, tt, 0.25, 0.2, ellip=0.0) * 1234.0
    bt = np.linspace(-100, 1000, 200)
    gf = eb_model(eph, bt, 0.25, 0.2, ellip=0.0) * 1e4
    tab = Table({"band": ["G"] * 200, "btjd": bt, "flux": gf, "flux_err": np.full(200, 1.0),
                 "rejected": np.zeros(200, bool)})
    _, fn, _, _ = normalize_band(eph, tab, "G")
    tn = normalize_tess(eph, tt, tf)
    assert abs(np.max(tn) - 1.0) < 0.01 and abs(np.max(fn) - 1.0) < 0.01   # same out-of-eclipse level
