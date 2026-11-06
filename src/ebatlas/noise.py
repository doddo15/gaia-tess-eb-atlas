"""Empirical per-transit G noise from (presumably) constant stars, Gaia-team style.

Gaia's per-transit flux errors are internal: they come from the scatter of the ~9 AF
CCD fluxes within one transit, so anything common to those CCDs (that transit's
calibration, background, crowding, attitude) is missed. Mean-photometry errors are
external (scatter between observations / sqrt(N)), so for constant stars they encode
the real epoch-to-epoch noise.

In DR3, phot_g_n_obs counts CCD observations, not transits, so the per-transit noise is
bracketed rather than determined:
  upper = phot_g_mean_flux_error * sqrt(phot_g_n_obs) / flux
          (all scatter is transit-to-transit systematics: CCD scatter = transit scatter)
  lower = upper / sqrt(phot_g_n_obs / matched_transits)
          (all scatter is independent per-CCD noise, which averages down within a transit)
Bright stars sit near the upper bound (systematics dominate), faint ones nearer the lower.
DR4 publishes epoch photometry for all sources, so there the noise can be measured
directly from constant stars' light curves.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table

NOISE_QUERY = """
SELECT TOP {n} phot_g_mean_mag, phot_g_mean_flux, phot_g_mean_flux_error, phot_g_n_obs, matched_transits
FROM gaiadr3.gaia_source
WHERE random_index < {max_index}
  AND phot_g_mean_mag BETWEEN {gmin} AND {gmax}
  AND phot_variable_flag != 'VARIABLE'
  AND ruwe < 1.4
  AND phot_g_n_obs > 0 AND matched_transits > 0
"""


def fetch_constant_stars(n_per_bin=1000, bins=np.arange(5.0, 20.51, 0.5)):
    """DR3 sources not flagged variable, sampled bin by bin in G.

    A single random sample is almost all faint stars (bright ones are rare), which
    left the bright end of the curve empty. Each bin gets its own query; the
    random_index cut is loosened for bright bins so they still fill up.
    """
    from astropy.table import vstack
    from astroquery.gaia import Gaia
    parts = []
    for a, b in zip(bins[:-1], bins[1:]):
        cap = 1_900_000_000 if a < 13 else (200_000_000 if a < 16 else 20_000_000)
        q = NOISE_QUERY.format(n=int(n_per_bin), gmin=a, gmax=b, max_index=cap)
        parts.append(Gaia.launch_job_async(q).get_results())
    return vstack(parts)


def noise_curve(stars, bins=np.arange(5.0, 20.51, 0.5)):
    """Median relative per-transit noise bounds per G bin -> Table(g, upper, lower, n)."""
    g = np.asarray(stars["phot_g_mean_mag"], float)
    f = np.asarray(stars["phot_g_mean_flux"], float)
    fe = np.asarray(stars["phot_g_mean_flux_error"], float)
    nobs = np.asarray(stars["phot_g_n_obs"], float)
    ntr = np.asarray(stars["matched_transits"], float)
    up = fe * np.sqrt(nobs) / f
    lo = up / np.sqrt(np.clip(nobs / ntr, 1.0, None))
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (g >= a) & (g < b) & np.isfinite(up) & np.isfinite(lo)
        if m.sum() >= 20:
            rows.append((0.5 * (a + b), float(np.median(up[m])), float(np.median(lo[m])), int(m.sum())))
    return Table(rows=rows, names=("g", "upper", "lower", "n"))


def interp_curve(curve, g):
    """(lower, upper) relative per-transit noise at magnitude(s) g."""
    g = np.atleast_1d(np.asarray(g, float))
    return (np.interp(g, curve["g"], curve["lower"]), np.interp(g, curve["g"], curve["upper"]))
