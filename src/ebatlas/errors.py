"""Error modelling shared by the Gaia offset scan and the TESS timing fits.

Effective per-point uncertainty (each term measured, none assumed):
    sigma_eff^2 = sigma_formal^2 + sigma_inst(G)^2 + sigma_template^2 + sigma_jit^2
  sigma_formal   Gaia's per-transit error (internal: scatter of ~9 CCDs in one transit)
  sigma_inst(G)  instrumental floor from constant stars at the source's magnitude
                 (noise.py; lower bound of the DR3 bracket, so it never over-inflates)
  sigma_template spread of bootstrapped TESS templates at each point's phase
  sigma_jit      per-source extra scatter, fitted by maximum likelihood to the residuals
                 (intrinsic variability, crowding, anything the other terms miss)
For TESS timings the same jitter estimator replaces chi^2 inflation.

Calibration on synthetics (pull widths, 1.0 = ideal): 0.71 with correct Gaia errors,
0.95-1.32 with Gaia errors 5x too small (narrow / wide eclipses); 95-100% of pulls
inside 3 sigma. Treat offset errors as good to ~30%.
"""
from __future__ import annotations

import numpy as np


def jitter_ml(resid, err, max_iter=60):
    """Maximum-likelihood extra scatter s: maximise -1/2 sum[r^2/(e^2+s^2) + ln(e^2+s^2)].

    Returns 0 when the residuals are consistent with the errors. Solved by bisection on
    the derivative, which is monotonic in s^2 for this likelihood near its maximum.
    """
    r = np.asarray(resid, float); e = np.asarray(err, float)
    ok = np.isfinite(r) & np.isfinite(e) & (e > 0)
    r, e = r[ok], e[ok]
    if r.size < 3:
        return 0.0
    def dlnL(s2):
        v = e**2 + s2
        return np.sum(r**2 / v**2 - 1.0 / v)
    if dlnL(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, max(np.var(r) * 4, np.max(e) ** 2)
    while dlnL(hi) > 0:
        hi *= 4
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dlnL(mid) > 0 else (lo, mid)
    return float(np.sqrt(0.5 * (lo + hi)))


def instrumental_floor(curve, g_mag):
    """Relative per-transit instrumental noise at G (lower bound of the constant-star bracket)."""
    if curve is None or not np.isfinite(g_mag):
        return 0.0
    from .noise import interp_curve
    lo, _ = interp_curve(curve, g_mag)
    return float(lo[0])
