"""Synthetic data in DR4 formats, for tests and release-day rehearsals.

`make_dr4_epoch_photometry` writes columns exactly as named in the DR4 draft
data model (section 11.1, DATAMODEL_GAIA). Fluxes come from a simple EB model;
magnitudes use the DR3 G zero point as a stand-in for the DR4 one.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table

from .ephemeris import DR4_START_GAIA, DR4_END_GAIA, btjd_to_gaia_time, gaia_time_to_btjd

ZP = {"g": 25.6874, "bp": 25.3385, "rp": 24.7479}    # DR3 values, stand-ins


def eb_model(ephem, btjd, d1, d2, w1=None, w2=None, ellip=0.008):
    ph = ephem.phase(btjd)
    f = 1.0 - ellip * np.cos(4 * np.pi * ph)
    w1 = w1 or ephem.primary_width or 0.04
    w2 = w2 or ephem.secondary_width or w1
    for c, d, w in ((0.0, d1, w1), (ephem.secondary_phase, d2, w2)):
        x = np.abs(((ph - c + 0.5) % 1.0) - 0.5) / (w / 2)
        f = f - d * np.clip(1 - x**2, 0, None) ** 0.6
    return f


def gaia_like_times(n_scans=36, seed=0):
    """Pairs of FoV transits 106.5 min apart, scattered over the DR4 window (Gaia time)."""
    rng = np.random.default_rng(seed)
    scans = np.sort(rng.uniform(DR4_START_GAIA, DR4_END_GAIA, n_scans))
    t = np.r_[scans, scans + 106.5 / 1440.0]
    fov = np.r_[np.ones(n_scans, int), np.full(n_scans, 2)]
    o = np.argsort(t)
    return t[o], fov[o]


def make_dr4_epoch_photometry(ephem, source_id, g_mag=12.0, depths=None, seed=0,
                              n_scans=36, flag_outliers=True):
    """Return a Table in DR4 epoch_photometry (DATAMODEL_GAIA) format."""
    rng = np.random.default_rng(seed)
    depths = depths or {"g": (0.30, 0.10), "bp": (0.33, 0.08), "rp": (0.27, 0.12)}
    t_g, fov = gaia_like_times(n_scans, seed)
    n = len(t_g)
    obmt = (np.arange(n) * 7_000_000 + 1_000_000).astype(np.int64)
    transit_id = (obmt << 17) | ((fov - 1).astype(np.int64) << 15) | (3 << 12) | 777
    tab = Table()
    tab["solution_id"] = np.full(n, 375316653866487564, np.int64)
    tab["source_id"] = np.full(n, int(source_id), np.int64)
    tab["transit_id"] = transit_id
    tab["g_n_obs"] = np.full(n, 9, np.int32)
    mags = {"g": g_mag, "bp": g_mag + 0.35, "rp": g_mag - 0.55}
    for band, dt, noise in (("g", 0.0, 0.002), ("bp", 12 / 86400, 0.006), ("rp", 14 / 86400, 0.006)):
        t = t_g + dt
        rel = eb_model(ephem, gaia_time_to_btjd(t), *depths[band])
        f0 = 10 ** (-0.4 * (mags[band] - ZP[band]))
        flux = f0 * rel * (1 + rng.normal(0, noise, n))
        err = f0 * noise * np.ones(n)
        tab[f"{band}_obs_time"] = t
        tab[f"{band}_flux"] = flux.astype(np.float32)
        tab[f"{band}_flux_error"] = err.astype(np.float32)
        tab[f"{band}_flux_over_error"] = (flux / err).astype(np.float32)
        tab[f"{band}_mag"] = (ZP[band] - 2.5 * np.log10(flux)).astype(np.float32)
    tab["transit_acq_flags"] = np.full(n, 0x2000 | 0x0020, np.int16)
    tab["transit_proc_flags"] = np.full(n, 0x0024, np.int16)
    pf = np.zeros((n, 12), np.int16)
    if flag_outliers:
        # Mimic mean-photometry outlier rejection of in-eclipse CCDs (bit 7).
        rel_g = eb_model(ephem, gaia_time_to_btjd(t_g), *depths["g"])
        deep = rel_g < 0.95
        pf[deep, 1:10] |= 1 << 7
    tab["phot_ccd_proc_flags"] = pf
    tab["xp_ccd_info_flags"] = np.zeros((n, 2), np.uint8)
    # One transit with BP rejected, one with all AF fluxes discarded.
    tab["bp_flux"][3] = np.nan; tab["bp_flux_error"][3] = np.nan
    tab["g_n_obs"][5] = 0
    return tab
