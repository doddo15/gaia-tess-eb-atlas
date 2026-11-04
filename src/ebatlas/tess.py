"""TESS light-curve retrieval via lightkurve.

Many catalogued EBs only have FFI light curves; SPOC 2-min data will be
missing for most. The fallback here takes any available author (e.g.
QLP, TESS-SPOC); a custom FFI pipeline may be needed for faint targets.
"""
from __future__ import annotations

import numpy as np


def fetch_tess_lc(tic_id, author="SPOC", quality_bitmask="default", sigma_upper=5.0):
    """Return (btjd, norm_flux, norm_flux_err) stitched across sectors, or None.

    Each sector is normalised by its median (lightkurve's stitch). Upward outliers
    beyond `sigma_upper` (robust, iterated) are removed; there is no lower clip,
    so eclipses are never touched.
    """
    import lightkurve as lk

    target = f"TIC {int(tic_id)}"
    sr = lk.search_lightcurve(target, mission="TESS", author=author)
    if len(sr) == 0:
        sr = lk.search_lightcurve(target, mission="TESS")
    if len(sr) == 0:
        return None
    lc = sr.download_all(quality_bitmask=quality_bitmask).stitch().remove_nans()
    if sigma_upper:
        lc = lc.remove_outliers(sigma_lower=float("inf"), sigma_upper=sigma_upper)
    return (np.asarray(lc.time.value, float),
            np.asarray(lc.flux.value, float),
            np.asarray(lc.flux_err.value, float))
