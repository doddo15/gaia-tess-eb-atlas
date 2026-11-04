"""TESS light-curve retrieval via lightkurve.

Many catalogued EBs only have FFI light curves; SPOC 2-min data will be
missing for most. The fallback here takes any available author (e.g.
QLP, TESS-SPOC); a custom FFI pipeline may be needed for faint targets.
"""
from __future__ import annotations

import numpy as np


def fetch_tess_lc(tic_id, author="SPOC", quality_bitmask="default"):
    """Return (btjd, norm_flux, norm_flux_err) stitched across sectors, or None."""
    import lightkurve as lk

    target = f"TIC {int(tic_id)}"
    sr = lk.search_lightcurve(target, mission="TESS", author=author)
    if len(sr) == 0:
        sr = lk.search_lightcurve(target, mission="TESS")
    if len(sr) == 0:
        return None
    lc = sr.download_all(quality_bitmask=quality_bitmask).stitch().remove_nans()
    return (np.asarray(lc.time.value, float),
            np.asarray(lc.flux.value, float),
            np.asarray(lc.flux_err.value, float))
