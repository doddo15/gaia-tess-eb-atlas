"""Measure eclipse widths, depths and secondary phase from a folded light curve.

TEBC (and some other catalogs) give period and t0 but no eclipse
geometry. Without widths we can't decide which Gaia epochs fall in
eclipse, and assuming the secondary sits at phase 0.5 is wrong for
eccentric systems. This module measures both from the TESS light curve.

Method: fold on the ephemeris, median-bin, estimate a robust baseline,
then walk outward from each minimum until the flux recovers to within
max(10% of the depth, 2 sigma) of baseline. The width is therefore
close to first-to-fourth contact. It is deliberately simple; swap in a
two-Gaussian model (as used to build TEBC) if you need more precision.
"""
from __future__ import annotations

import numpy as np

from .ephemeris import _wrap


def _bin(phase, flux, nbins):
    edges = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    med = np.full(nbins, np.nan)
    cnt = np.bincount(idx, minlength=nbins)
    order = np.argsort(idx)
    split = np.split(flux[order], np.cumsum(cnt)[:-1])
    for k, chunk in enumerate(split):
        if len(chunk) >= 3:
            med[k] = np.median(chunk)
    ok = np.isfinite(med)
    return centers[ok], med[ok], 1.0 / nbins


def _walk(centers, flux, i0, level):
    """Indices of the contiguous (circular) run of bins below `level` containing i0."""
    n = len(flux)
    lo = hi = i0
    for _ in range(n // 2):
        j = (lo - 1) % n
        if flux[j] >= level:
            break
        lo = j
    for _ in range(n // 2):
        j = (hi + 1) % n
        if flux[j] >= level:
            break
        hi = j
    return lo, hi


def _extent(centers, lo, hi, binw):
    """Phase span and centre of a run of bins, handling wrap-around."""
    left, right = centers[lo] - binw / 2, centers[hi] + binw / 2
    if right < left:
        right += 1.0
    return right - left, _wrap(0.5 * (left + right))


def measure_eclipses(ephem, t, flux, nbins=None, primary_search=0.1, min_snr=5.0):
    """Return a dict of eclipse parameters (phase units; depths as fractional flux)."""
    t = np.asarray(t, float); f = np.asarray(flux, float)
    good = np.isfinite(t) & np.isfinite(f)
    t, f = t[good], f[good] / np.nanmedian(f[good])
    if nbins is None:
        nbins = int(np.clip(len(t) // 20, 200, 2000))
    c, b, binw = _bin(ephem.phase(t), f, nbins)
    res = {"nbins": nbins, "n_valid_bins": int(len(c)), "primary_width": np.nan,
           "primary_depth": np.nan, "primary_offset": np.nan, "secondary_phase": np.nan,
           "secondary_width": np.nan, "secondary_depth": np.nan,
           "secondary_detected": False, "contact_like": False}
    if len(c) < 20:
        return res

    base = np.median(b[b > np.percentile(b, 20)])
    noise = 1.4826 * np.median(np.abs(np.diff(b))) / np.sqrt(2)
    res.update(baseline=float(base), bin_noise=float(noise))

    near = np.where(np.abs(c) < primary_search)[0]
    if len(near) == 0:
        return res
    i1 = near[np.argmin(b[near])]
    d1 = base - b[i1]
    if d1 < min_snr * noise:
        return res
    lo, hi = _walk(c, b, i1, base - max(0.1 * d1, 2 * noise))
    w1, c1 = _extent(c, lo, hi, binw)
    res.update(primary_width=float(w1), primary_depth=float(d1), primary_offset=float(c1))

    # Secondary: deepest point outside the primary (with a margin).
    away = np.abs(_wrap(c - c1)) > 0.75 * w1
    if away.sum() >= 10:
        cand = np.where(away)[0]
        i2 = cand[np.argmin(b[cand])]
        d2 = base - b[i2]
        if d2 >= min_snr * noise:
            lo2, hi2 = _walk(c, b, i2, base - max(0.1 * d2, 2 * noise))
            w2, c2 = _extent(c, lo2, hi2, binw)
            res.update(secondary_phase=float(np.mod(c2, 1.0)), secondary_width=float(w2),
                       secondary_depth=float(d2), secondary_detected=True)

    # Very broad "eclipses" mean continuous variation (contact / ellipsoidal):
    # in-eclipse counting is not meaningful there.
    res["contact_like"] = bool(w1 > 0.25 or (res["secondary_detected"] and res["secondary_width"] > 0.25))
    return res
