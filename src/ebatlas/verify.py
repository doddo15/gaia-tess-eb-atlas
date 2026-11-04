"""DR3 verification: fold Gaia DR3 epoch photometry on TESS ephemerides and check
that the eclipses land where TESS says they should.

This validates the whole chain (cross-match, time systems, ephemeris propagation,
folding) on real data before DR4. Three checks per target:

1. Period (`period_agreement`): if DR3 classified the source as an EB
   (gaiadr3.vari_eclipsing_binary), compare 1/frequency with the catalog period.
   Classes: match, half, double, mismatch, not_in_dr3_eb.
2. Phase offset (`phase_offset_scan`): slide a template built from the folded TESS
   light curve across the Gaia G points (free offset and depth scale, since the
   bands differ) and find the shift delta that fits best. delta is the phase at
   which the Gaia-era eclipse actually falls under the catalog ephemeris. It should
   be consistent with zero given the propagated ephemeris error: z = delta / sigma.
3. Eclipse recovery: in-eclipse counts and depth S/N from `plotting.summarize`,
   plus how many in-eclipse points DR3's variability pipeline rejected.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table

from .ephemeris import _wrap
from .eclipses import _bin, measure_eclipses


def period_agreement(period, dr3_frequency, rtol=2e-3):
    """(ratio P_DR3/P_catalog, class). rtol is relative; DR3 periods come from ~34 months."""
    try:
        f = float(dr3_frequency)
    except (TypeError, ValueError):
        f = np.nan
    if not np.isfinite(f) or f <= 0:
        return np.nan, "not_in_dr3_eb"
    r = (1.0 / f) / float(period)
    for name, k in (("match", 1.0), ("half", 0.5), ("double", 2.0)):
        if abs(r / k - 1.0) < rtol:
            return r, name
    return r, "mismatch"


def _running_mean(x, k):
    c = np.cumsum(np.r_[0.0, x])
    return (c[k:] - c[:-k]) / k


def tess_template(ephem, t, flux, nbins=None, n_harm=8, window=0.75, k=5):
    """Periodic model of the folded TESS light curve.

    Outside the eclipse windows (|phase - centre| < window * width): a Fourier series
    (n_harm harmonics) fitted to the binned curve, carrying ellipsoidal/reflection
    variation without bin noise. Inside the windows: a running mean over k
    phase-sorted TESS points (of both phase and residual), which keeps the data's own
    phase resolution; fixed bins would smear sharp contact points and bias offsets.
    Without eclipse widths on `ephem` it falls back to the plain binned curve.
    """
    t = np.asarray(t, float); f = np.asarray(flux, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok] / np.nanmedian(f[ok])
    if nbins is None:
        nbins = int(np.clip(len(t) // 20, 200, 4000))
    ph_all = ephem.phase(t)
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
        return lambda ph: np.interp(_wrap(ph), cc, bb)

    def design(x):
        x = np.atleast_1d(x)
        cols = [np.ones_like(x)]
        for j in range(1, n_harm + 1):
            cols += [np.cos(2 * np.pi * j * x), np.sin(2 * np.pi * j * x)]
        return np.column_stack(cols)

    coef, *_ = np.linalg.lstsq(design(c[~win_b]), b[~win_b], rcond=None)
    w = in_win(ph_all)
    if w.sum() < 2 * k:
        return lambda ph: design(_wrap(ph)) @ coef
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
    return lambda ph: design(_wrap(ph)) @ coef + np.where(outside(_wrap(ph)), 0.0, np.interp(_wrap(ph), cc, rr))


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


def phase_offset_scan(ephem, gaia_tab, template, band="G", span=None, n=801, min_dchi2=25.0,
                      boot_templates=None):
    """Best phase shift of the TESS template against the Gaia points of one band.

    Model at each trial delta: flux = a + b * (template(phase - delta) - 1), fitted
    with the Gaia errors (inflated by sqrt(chi2_min/dof) when the fit is worse than
    they claim). shift_err combines (a) the half-width of delta-chi2 <= 4, halved,
    which suits flat-bottomed chi^2 curves from few in-eclipse points, and (b) with
    `boot_templates`, the scatter of best shifts across bootstrapped TESS templates,
    i.e. the template's own noise, which the Gaia chi^2 cannot see.
    `constrained` needs at least one Gaia point inside an eclipse of the best-fit
    template AND a preferred shift by more than `min_dchi2`; out-of-eclipse
    (ellipsoidal) signal alone does not count.
    """
    sel = (np.asarray(gaia_tab["band"]) == band) & ~np.asarray(gaia_tab["rejected"]).astype(bool)
    f = np.asarray(gaia_tab["flux"], float)[sel]; e = np.asarray(gaia_tab["flux_err"], float)[sel]
    t = np.asarray(gaia_tab["btjd"], float)[sel]
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & np.isfinite(t)
    t, f, e = t[ok], f[ok], e[ok]
    out = {"n_points": int(len(t)), "best_shift": np.nan, "shift_err": np.nan, "shift_err_template": np.nan,
           "z": np.nan, "depth_scale": np.nan, "dchi2_max": np.nan, "chi2_red": np.nan, "constrained": False,
           "sigma_phase_gaia": np.nan, "n_in_eclipse": 0}
    if len(t) < 10:
        return out
    med = np.median(f); f, e = f / med, e / med
    from .plotting import MAX_NORM_FLUX
    keep = f < MAX_NORM_FLUX                      # brightenings cannot be eclipse signal
    t, f, e = t[keep], f[keep], e[keep]
    out["n_points"] = int(len(t))
    if len(t) < 10:
        return out
    ph = ephem.phase(t)
    sig = float(np.median(ephem.phase_sigma(t)))
    out["sigma_phase_gaia"] = sig
    span = span or float(min(0.5, max(0.03, 6 * sig)))
    shifts = np.linspace(-span, span, n)
    chi2, scale = _scan(ph, f, e, template, shifts)
    k = int(np.argmin(chi2)); dof = max(len(t) - 3, 1)
    infl = max(1.0, chi2[k] / dof)
    d = chi2 - chi2[k]
    inside = d <= 4.0 * infl
    lo, hi = k, k
    while lo > 0 and inside[lo - 1]:
        lo -= 1
    while hi < n - 1 and inside[hi + 1]:
        hi += 1
    err = max(0.25 * (shifts[hi] - shifts[lo]), shifts[1] - shifts[0])
    terr = 0.0
    if boot_templates:
        bs = [shifts[int(np.argmin(_scan(ph, f, e, bt, shifts)[0]))] for bt in boot_templates]
        terr = float(np.std(bs))
    grid = np.linspace(-0.5, 0.5, 4001)
    tg = template(grid); base = np.median(tg); depth = (base - tg).max()
    n_in = int(np.sum((base - template(ph - shifts[k])) > 0.2 * depth)) if depth > 0 else 0
    tot = float(np.hypot(err, terr))
    out.update(best_shift=float(shifts[k]), shift_err=tot, shift_err_template=terr,
               depth_scale=float(scale[k]), dchi2_max=float(d.max() / infl), chi2_red=float(chi2[k] / dof),
               n_in_eclipse=n_in, constrained=bool(d.max() / infl > min_dchi2 and 0 < k < n - 1 and n_in >= 1))
    out["z"] = float(shifts[k] / np.hypot(tot, sig))
    return out


def _analyse(row, factor, lc, tab, n_boot):
    """Ephemeris (catalog period x factor), eclipse geometry from TESS, and offset scan."""
    from .catalogs import row_to_ephemeris
    from .ephemeris import Ephemeris

    base = row_to_ephemeris(row)
    if factor != 1:
        base = Ephemeris(base.period * factor, base.t0, base.period_err * factor, base.t0_err)
    if lc is None:
        return base, None, {}
    geom = measure_eclipses(base, lc[0], lc[1])
    pick = lambda k, d: geom[k] if np.isfinite(geom[k]) else d
    eph = Ephemeris(base.period, base.t0, base.period_err, base.t0_err,
                    primary_width=pick("primary_width", base.primary_width),
                    secondary_phase=pick("secondary_phase", base.secondary_phase),
                    secondary_width=pick("secondary_width", base.secondary_width))
    off = phase_offset_scan(eph, tab, tess_template(eph, lc[0], lc[1]),
                            boot_templates=bootstrap_templates(eph, lc[0], lc[1], n=n_boot))
    return eph, geom, off


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def verify_dr3(cat, xmatch, out_dir, fetcher=None, lc_getter=None, cache_dir="cache/dr3_epoch_photometry",
               pages=True, limit=None, n_boot=10):
    """Run the three checks for every catalog EB that has DR3 epoch photometry.

    cat: unified catalog; xmatch: stage-1 table (needs eb_id, dr3_source_id,
    dr3_has_epoch_photometry, dr3_eb_frequency). fetcher(ids) -> {sid: table}
    and lc_getter(tic) -> (t, f, e) | None can be swapped for offline runs.
    Writes verification.csv, verification_summary.png and (optionally) pages.
    """
    from .catalogs import row_to_ephemeris
    from .plotting import summarize, atlas_page

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if fetcher is None:
        from .gaia_epochs import fetch_dr3_epoch_photometry
        fetcher = lambda ids: fetch_dr3_epoch_photometry(ids, cache_dir=cache_dir)
    if lc_getter is None:
        from .tess import fetch_tess_lc as lc_getter

    by_eb = {r["eb_id"]: r for r in cat}
    keep = [r for r in xmatch if int(r["dr3_source_id"]) > 0 and _truthy(r["dr3_has_epoch_photometry"])
            and r["eb_id"] in by_eb]
    if limit:
        keep = keep[:limit]
    epochs = fetcher(sorted({int(r["dr3_source_id"]) for r in keep}))

    rows, lc_cache = [], {}
    for x in keep:
        row = by_eb[x["eb_id"]]; sid = int(x["dr3_source_id"]); tic = int(row["tic_id"])
        rec = {"eb_id": x["eb_id"], "tic_id": tic, "gaia_dr3_id": sid, "period": float(row["period"]),
               "has_dr3_epochs": sid in epochs, "has_tess": False, "adopted_period_factor": 1}
        ratio, pclass = period_agreement(row["period"], x["dr3_eb_frequency"] if "dr3_eb_frequency" in xmatch.colnames else np.nan)
        rec.update(dr3_period_ratio=ratio, period_check=pclass)
        if sid not in epochs:
            rows.append(rec); continue
        if tic not in lc_cache:
            try:
                lc_cache[tic] = lc_getter(tic)
            except Exception:
                lc_cache[tic] = None
        lc = lc_cache[tic]
        rec["has_tess"] = lc is not None

        eph, geom, off = _analyse(row, 1, lc, epochs[sid], n_boot)
        # DR3 says the period is twice the catalog's: a common catalog alias for contact
        # binaries with near-equal eclipses. Try 2P and adopt it if it fits clearly better.
        if pclass == "double" and lc is not None:
            eph2, geom2, off2 = _analyse(row, 2, lc, epochs[sid], n_boot)
            better = (off2.get("constrained") and np.isfinite(off2.get("chi2_red", np.nan))
                      and off2["chi2_red"] < 0.9 * off.get("chi2_red", np.inf))
            alt = (eph, geom, off) if better else (eph2, geom2, off2)
            rec.update({f"alt_offset_{k}": v for k, v in alt[2].items()})
            rec["alt_period_factor"] = 1 if better else 2
            if better:
                eph, geom, off = eph2, geom2, off2
                rec["adopted_period_factor"] = 2
        if geom is not None:
            rec.update({f"meas_{k}": geom[k] for k in ("primary_width", "primary_depth", "secondary_phase",
                                                      "secondary_width", "contact_like")})
        rec.update(summarize(eph, epochs[sid]))
        rec.update({f"offset_{k}": v for k, v in off.items()})
        if pages:
            tag = "  (2× catalog P)" if rec["adopted_period_factor"] == 2 else ""
            atlas_page(eph, epochs[sid], tess=lc, title=f"{x['eb_id']} / Gaia DR3 {sid}{tag}",
                       out=out_dir / f"{x['eb_id']}.png")
        rows.append(rec)

    from .atlas import _rows_to_table
    res = _rows_to_table(rows) if rows else Table()
    res.write(out_dir / "verification.csv", overwrite=True)
    if len(res):
        verification_figure(res, out_dir / "verification_summary.png")
    return res


def verification_figure(res, path):
    """Three panels: offset z-scores, |offset| vs ephemeris error, DR3/catalog period ratio."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    con = (np.asarray(res["offset_constrained"]).astype(bool) if "offset_constrained" in res.colnames
           else np.zeros(len(res), bool))
    if con.any():
        z = np.asarray(res["offset_z"], float)[con]
        ax[0].hist(np.clip(z, -8, 8), bins=np.linspace(-8, 8, 33), color="#1f5fbf", alpha=0.8)
        xs = np.linspace(-8, 8, 200)
        ax[0].plot(xs, con.sum() * 0.5 * np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
        ax[0].legend(fontsize=8)
        s = np.asarray(res["offset_sigma_phase_gaia"], float)[con]
        d = np.abs(np.asarray(res["offset_best_shift"], float)[con])
        ax[1].loglog(np.maximum(s, 1e-6), np.maximum(d, 1e-5), "o", ms=4, color="#1f5fbf")
        lim = [1e-6, 0.5]
        for k, ls in ((1, "-"), (3, "--")):
            ax[1].plot(lim, [k * v for v in lim], color="0.4", ls=ls, lw=0.8, label=f"{k}σ")
        ax[1].legend(fontsize=8)
    ax[0].set_xlabel("Phase offset / combined error (z)"); ax[0].set_ylabel("N targets")
    ax[0].set_title(f"Phase-offset check: {con.sum()} constrained of {len(res)}", fontsize=10)
    ax[1].set_xlabel(r"Propagated ephemeris error $\sigma_\phi$ at Gaia epochs")
    ax[1].set_ylabel("|best phase offset|"); ax[1].set_title("Offsets vs. expected drift", fontsize=10)
    r = np.asarray(res["dr3_period_ratio"], float); r = r[np.isfinite(r)]
    if len(r):
        ax[2].hist(np.log2(r), bins=np.linspace(-2.5, 2.5, 101), color="#d1495b")
        for v in (-1, 0, 1):
            ax[2].axvline(v, color="0.4", lw=0.8, ls=":")
    ax[2].set_xlabel(r"$\log_2(P_{\rm DR3}/P_{\rm TESS})$  (0 = match, ±1 = double/half)")
    ax[2].set_title(f"Period check: {len(r)} also classified EB in DR3", fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)
