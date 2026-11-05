"""Light-curve templates from folded TESS photometry, shared by verification and ephemeris refits.

`tess_template` builds a periodic model (Fourier series outside eclipses, running mean of
phase-sorted points inside). `smeared` averages a template over an exposure time, so a
template from short-cadence data can be compared with long-cadence (FFI) sectors.
`_scan` fits flux = a + b * (template(phase - shift) - 1) for many shifts at once.
"""
from __future__ import annotations

import numpy as np

from .ephemeris import _wrap
from .eclipses import _bin


GRID_N = 20000


class GriddedTemplate:
    """A template tabulated on a fine periodic phase grid; evaluation is one np.interp.

    Templates are evaluated thousands of times per fit (every trial shift), so
    tabulating once makes scans ~10-50x faster. Grid spacing 5e-5 in phase.
    """

    def __init__(self, template, n=GRID_N, values=None):
        self.x = (np.arange(n) + 0.5) / n - 0.5
        self.y = np.asarray(template(self.x), float) if values is None else np.asarray(values, float)
        self._xx = np.r_[self.x[-1] - 1, self.x, self.x[0] + 1]
        self._yy = np.r_[self.y[-1], self.y, self.y[0]]

    def __call__(self, ph):
        return np.interp(_wrap(np.asarray(ph, float)), self._xx, self._yy)

    def smeared(self, exptime_phase):
        """Boxcar average over an exposure of `exptime_phase` (phase units), on the periodic grid."""
        n = len(self.y)
        k = int(round(exptime_phase * n))
        if not np.isfinite(exptime_phase) or k < 2:
            return self
        k = min(k, n - 1)
        ext = np.r_[self.y[-(k // 2):], self.y, self.y[:k - k // 2]]
        cs = np.cumsum(np.r_[0.0, ext])
        vals = (cs[k:k + n] - cs[:n]) / k
        return GriddedTemplate(None, n=n, values=vals)


def gridded(template):
    return template if isinstance(template, GriddedTemplate) else GriddedTemplate(template)


def smeared(template, exptime_phase):
    """Template averaged over an exposure of `exptime_phase` (in phase units)."""
    return gridded(template).smeared(exptime_phase)


def shortest_cadence_mask(exptime, min_points=1500):
    """Points from the shortest exposure time available, if there are enough of them; else all."""
    exptime = np.asarray(exptime, float)
    if not np.isfinite(exptime).any():
        return np.ones(exptime.size, bool)
    m = exptime <= np.nanmin(exptime) * 1.01
    return m if m.sum() >= min_points else np.ones(exptime.size, bool)


def _running_mean(x, k):
    c = np.cumsum(np.r_[0.0, x])
    return (c[k:] - c[:-k]) / k


def tess_template(ephem, t, flux, nbins=None, n_harm=8, window=0.75, k=5, phase_shift=None):
    """Periodic model of the folded TESS light curve.

    Outside the eclipse windows (|phase - centre| < window * width): a Fourier series
    (n_harm harmonics) fitted to the binned curve, carrying ellipsoidal/reflection
    variation without bin noise. Inside the windows: a running mean over k
    phase-sorted TESS points (of both phase and residual), which keeps the data's own
    phase resolution; fixed bins would smear sharp contact points and bias offsets.
    Without eclipse widths on `ephem` it falls back to the plain binned curve.
    `phase_shift` (per point) is subtracted from the phases first: used to line up
    sectors by their measured timing offsets before stacking (shift-and-stack).
    """
    t = np.asarray(t, float); f = np.asarray(flux, float)
    ok = np.isfinite(t) & np.isfinite(f)
    shift = np.zeros(t.size) if phase_shift is None else np.asarray(phase_shift, float)
    t, f, shift = t[ok], f[ok] / np.nanmedian(f[ok]), shift[ok]
    if nbins is None:
        nbins = int(np.clip(len(t) // 20, 200, 4000))
    ph_all = _wrap(ephem.phase(t) - shift)
    c, b, _ = _bin(ph_all, f, nbins)

    def in_win(x):
        m = np.zeros(np.shape(x), bool)
        if ephem.primary_width:
            m |= np.abs(_wrap(x)) < window * ephem.primary_width
        if ephem.secondary_width:
            m |= np.abs(_wrap(x - ephem.secondary_phase)) < window * ephem.secondary_width
        return m

    win_b = in_win(c)
    if not win_b.any() or (~win_b).sum() < 2 * n_harm + 5:
        cc, bb = np.r_[c - 1, c, c + 1], np.r_[b, b, b]
        return GriddedTemplate(lambda ph: np.interp(_wrap(ph), cc, bb))

    def design(x):
        x = np.atleast_1d(x)
        cols = [np.ones_like(x)]
        for j in range(1, n_harm + 1):
            cols += [np.cos(2 * np.pi * j * x), np.sin(2 * np.pi * j * x)]
        return np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design(c[~win_b]), b[~win_b], rcond=None)
    w = in_win(ph_all)
    if w.sum() < 2 * k:
        return GriddedTemplate(lambda ph: design(_wrap(ph)) @ coef)
    pw = ph_all[w]; rw = f[w] - design(pw) @ coef
    o = np.argsort(pw); pw, rw = pw[o], rw[o]
    xs, ys = _running_mean(pw, k), _running_mean(rw, k)
    # zero residual just outside each window edge so the template is continuous
    edges = []
    for cen, wd in ((0.0, ephem.primary_width), (ephem.secondary_phase, ephem.secondary_width)):
        if wd:
            edges += [_wrap(cen - window * wd), _wrap(cen + window * wd)]
    xs = np.r_[xs, edges]; ys = np.r_[ys, np.zeros(len(edges))]
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    cc, rr = np.r_[xs - 1, xs, xs + 1], np.r_[ys, ys, ys]
    outside = lambda ph: ~in_win(ph)
    return GriddedTemplate(lambda ph: design(_wrap(ph)) @ coef
                           + np.where(outside(_wrap(ph)), 0.0, np.interp(_wrap(ph), cc, rr)))


def bootstrap_templates(ephem, t, flux, n=10, seed=0, **kw):
    """Templates from TESS light curves resampled with replacement (for template-noise errors)."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float); f = np.asarray(flux, float)
    out = []
    for _ in range(n):
        i = np.sort(rng.integers(0, len(t), len(t)))
        out.append(tess_template(ephem, t[i], f[i], **kw))
    return out


def _scan(ph, f, e, template, shifts):
    """Vectorised weighted fit of flux = a + b*(template(ph - s) - 1) for every shift s."""
    M = np.stack([template(ph - s) - 1.0 for s in shifts])            # (n_shift, n_pts)
    w = 1.0 / e**2
    S, Sx, Sy = w.sum(), M @ w, (w * f).sum()
    Sxx, Sxy = (M**2) @ w, M @ (w * f)
    det = S * Sxx - Sx**2
    det = np.where(np.abs(det) < 1e-300, np.nan, det)
    b = (S * Sxy - Sx * Sy) / det
    a = (Sy - b * Sx) / S
    chi2 = (((f[None, :] - a[:, None] - b[:, None] * M) ** 2) * w[None, :]).sum(axis=1)
    return np.nan_to_num(chi2, nan=np.inf), b


