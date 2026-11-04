"""Pre-release planning: how well can each catalog ephemeris be projected onto Gaia epochs?

Gaia DR4 spans roughly BTJD -136.5 (2014-07-25) to 1868.5 (2020-01-20),
while TESS ephemerides are anchored at BTJD >~ 1325. Propagating back
multiplies the period error by thousands of cycles for short-period EBs.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table

from .ephemeris import DR4_START_GAIA, DR4_END_GAIA, gaia_time_to_btjd

DR4_START_BTJD = float(gaia_time_to_btjd(DR4_START_GAIA)[0])
DR4_END_BTJD = float(gaia_time_to_btjd(DR4_END_GAIA)[0])
DR4_MID_BTJD = 0.5 * (DR4_START_BTJD + DR4_END_BTJD)


def phase_sigma_table(cat):
    """sigma_phase at DR4 start / middle / end for each catalog row.

    Also returns the worst-case timing uncertainty in minutes, which is
    the number to compare with eclipse durations.
    """
    P = np.asarray(cat["period"], float)
    T0 = np.asarray(cat["t0_btjd"], float)
    sP = np.nan_to_num(np.asarray(cat["period_err"], float))
    sT = np.nan_to_num(np.asarray(cat["t0_err"], float))
    out = Table()
    out["eb_id"] = cat["eb_id"]
    for name, t in (("start", DR4_START_BTJD), ("mid", DR4_MID_BTJD), ("end", DR4_END_BTJD)):
        sig_t = np.hypot(sT, (t - T0) / P * sP)
        out[f"sigma_phase_{name}"] = sig_t / P
        if name == "start":
            out["sigma_minutes_start"] = sig_t * 1440.0
    out["n_cycles_back"] = np.abs(DR4_START_BTJD - T0) / P
    return out
