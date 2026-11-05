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
from .eclipses import measure_eclipses
from .templates import tess_template, bootstrap_templates, smeared, _scan, _running_mean  # noqa: F401 (re-exported)
from .ephemeris import Ephemeris
from .ephemeris_fit import refit_ephemeris, oc_figure
from .tess import as_tesslc


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


def _analyse(base, lc, tab, n_boot, template=None):
    """Eclipse geometry, template and Gaia offset scan on the given base ephemeris."""
    from .ephemeris import Ephemeris
    from .templates import shortest_cadence_mask
    from .tess import as_tesslc

    if lc is None:
        return base, None, {}
    lc = as_tesslc(lc)
    sc = shortest_cadence_mask(lc.exptime)
    t, f = np.asarray(lc.t)[sc], np.asarray(lc.flux)[sc]
    geom = measure_eclipses(base, t, f)
    pick = lambda k, d: geom[k] if np.isfinite(geom[k]) else d
    eph = Ephemeris(base.period, base.t0, base.period_err, base.t0_err,
                    primary_width=pick("primary_width", base.primary_width),
                    secondary_phase=pick("secondary_phase", base.secondary_phase),
                    secondary_width=pick("secondary_width", base.secondary_width))
    tmpl = template if template is not None else tess_template(eph, t, f)
    boots = bootstrap_templates(eph, t, f, n=n_boot)
    off = phase_offset_scan(eph, tab, tmpl, boot_templates=boots)
    off["_diag"] = chi2_breakdown(eph, tab, tmpl, off.get("best_shift", np.nan), boots)
    return eph, geom, off


def _refit_columns(fit, cat_eph, gaia_btjd):
    e = fit["ephem"]
    rec = {"refit_status": fit["status"], "refit_n_sectors": fit["n_sectors"],
           "refit_baseline_days": fit["baseline_days"], "refit_chi2_red": fit["chi2_red"],
           "refit_quad_snr": fit["quad_snr"], "refit_period_shift_sigma": fit["period_shift_sigma"],
           "template_exptime_s": fit["template_exptime"]}
    if fit["status"] == "ok":
        rec.update(refit_period=e.period, refit_period_err=e.period_err, refit_t0=e.t0, refit_t0_err=e.t0_err)
        if gaia_btjd is not None and len(gaia_btjd):
            tg = float(np.median(gaia_btjd))
            # how far apart the catalog and refit ephemerides put the eclipses at the Gaia epochs
            rec["catalog_drift_at_gaia"] = float(_wrap(cat_eph.phase(tg) - e.phase(tg)))
    return rec


def fit_ok(off, max_chi2_red=20.0, scale_range=(0.2, 5.0)):
    """Offset fits worth trusting: constrained, sensible depth scale, template matches the Gaia points."""
    return bool(off.get("constrained") and scale_range[0] < off.get("depth_scale", np.nan) < scale_range[1]
                and off.get("chi2_red", np.inf) < max_chi2_red)


def chi2_breakdown(ephem, gaia_tab, template, shift, boot_templates=None, band="G", in_frac=0.2):
    """Where does the offset fit's chi^2 come from?

    Returns, for the best-fit shift: n_in / n_oot (Gaia points in / out of eclipse, using
    the template's own dip > in_frac of its depth), chi2_red_in / chi2_red_oot,
    oot_scatter_ratio (robust std of normalised out-of-eclipse residuals: 1 means Gaia's
    formal errors describe the scatter), the same chi2's with the TESS template's noise
    (spread of bootstrap templates at each Gaia phase) added to the errors, the median
    template-noise / Gaia-error ratio in eclipse, and the source's median G and formal
    relative per-transit error.

    Reading it: excess mostly out of eclipse -> noise (Gaia errors, crowding); excess
    mostly in eclipse and not removed by template noise -> model mismatch (bandpass,
    changes between the Gaia and TESS eras).
    """
    from .plotting import MAX_NORM_FLUX
    sel = (np.asarray(gaia_tab["band"]) == band) & ~np.asarray(gaia_tab["rejected"]).astype(bool)
    f = np.asarray(gaia_tab["flux"], float)[sel]; e = np.asarray(gaia_tab["flux_err"], float)[sel]
    t = np.asarray(gaia_tab["btjd"], float)[sel]
    mag = np.asarray(gaia_tab["mag"], float)[sel] if "mag" in gaia_tab.colnames else np.full(sel.sum(), np.nan)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & np.isfinite(t)
    t, f, e, mag = t[ok], f[ok], e[ok], mag[ok]
    out = {"n_in": 0, "n_oot": 0, "chi2_red_in": np.nan, "chi2_red_oot": np.nan, "oot_scatter_ratio": np.nan,
           "chi2_red_in_tnoise": np.nan, "chi2_red_oot_tnoise": np.nan, "tnoise_over_err_in": np.nan,
           "g_mag": float(np.nanmedian(mag)) if mag.size else np.nan, "rel_err_formal": np.nan}
    if len(t) < 10 or not np.isfinite(shift):
        return out
    med = np.median(f); f, e = f / med, e / med
    keep = f < MAX_NORM_FLUX
    t, f, e = t[keep], f[keep], e[keep]
    out["rel_err_formal"] = float(np.median(e))
    ph = ephem.phase(t)
    m = template(ph - shift) - 1.0
    A = np.column_stack([np.ones_like(m), m])
    p, *_ = np.linalg.lstsq(A / e[:, None], f / e, rcond=None)
    model = A @ p
    grid = np.linspace(-0.5, 0.5, 4001); tg = template(grid); base = np.median(tg); depth = (base - tg).max()
    inn = (base - template(ph - shift)) > in_frac * depth if depth > 0 else np.zeros(len(t), bool)
    oot = ~inn

    def red(err, mask, dof_loss):
        n = mask.sum()
        return float(np.sum(((f - model) / err)[mask] ** 2) / max(n - dof_loss, 1)) if n else np.nan

    out.update(n_in=int(inn.sum()), n_oot=int(oot.sum()), chi2_red_in=red(e, inn, 0), chi2_red_oot=red(e, oot, 2))
    if oot.sum() >= 5:
        r = ((f - model) / e)[oot]
        out["oot_scatter_ratio"] = float(1.4826 * np.median(np.abs(r - np.median(r))))
    if boot_templates:
        tv = np.std([bt(ph - shift) for bt in boot_templates], axis=0) * abs(p[1])
        e2 = np.hypot(e, tv)
        out.update(chi2_red_in_tnoise=red(e2, inn, 0), chi2_red_oot_tnoise=red(e2, oot, 2),
                   tnoise_over_err_in=float(np.median(tv[inn] / e[inn])) if inn.any() else np.nan)
    return out


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "t", "yes")


def verify_dr3(cat, xmatch, out_dir, fetcher=None, lc_getter=None, cache_dir="cache/dr3_epoch_photometry",
               pages=True, limit=None, n_boot=10, refit=True):
    """Run the three checks for every catalog EB that has DR3 epoch photometry.

    With refit=True (default) each ephemeris is first refit from all TESS sectors
    (ephemeris_fit.refit_ephemeris); the Gaia comparison then uses the refit ephemeris.

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

    rows, lc_cache, sector_rows = [], {}, []
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
                lc_cache[tic] = as_tesslc(lc_getter(tic))
            except Exception:
                lc_cache[tic] = None
        lc = lc_cache[tic]
        rec["has_tess"] = lc is not None
        if lc is not None:
            rec["tess_sectors"] = int(len(np.unique(lc.sector)))
            rec["tess_authors"] = ",".join(sorted(set(map(str, lc.author))))
        g_btjd = np.asarray(epochs[sid]["btjd"][np.asarray(epochs[sid]["band"]) == "G"], float)

        def attempt(factor):
            cat = row_to_ephemeris(row)
            if factor != 1:
                cat = Ephemeris(cat.period * factor, cat.t0, cat.period_err * factor, cat.t0_err)
            fit = refit_ephemeris(cat, lc) if (refit and lc is not None) else None
            base = fit["ephem"] if fit is not None and fit["status"] == "ok" else cat
            tmpl = fit.get("template") if fit is not None and fit["status"] == "ok" else None
            eph, geom, off = _analyse(base, lc, epochs[sid], n_boot, template=tmpl)
            return eph, geom, off, fit, cat

        eph, geom, off, fit, cat = attempt(1)
        # DR3 says the period is twice the catalog's: a common catalog alias for contact
        # binaries with near-equal eclipses. Try 2P and adopt it if it fits clearly better.
        if pclass == "double" and lc is not None:
            alt = attempt(2)
            better = (alt[2].get("constrained") and np.isfinite(alt[2].get("chi2_red", np.nan))
                      and alt[2]["chi2_red"] < 0.9 * off.get("chi2_red", np.inf))
            other = (eph, geom, off, fit, cat) if better else alt
            rec.update({f"alt_offset_{k}": v for k, v in other[2].items() if not k.startswith("_")})
            rec["alt_period_factor"] = 1 if better else 2
            if better:
                eph, geom, off, fit, cat = alt
                rec["adopted_period_factor"] = 2
        if fit is not None:
            rec.update(_refit_columns(fit, cat, g_btjd))
            for srow in fit.get("sector_status", []):
                sector_rows.append({"eb_id": x["eb_id"], "tic_id": tic, **{k: srow[k] for k in srow.colnames}})
        if geom is not None:
            rec.update({f"meas_{k}": geom[k] for k in ("primary_width", "primary_depth", "secondary_phase",
                                                      "secondary_width", "contact_like")})
        rec.update(summarize(eph, epochs[sid]))
        rec.update({f"diag_{k}": v for k, v in off.pop("_diag", {}).items()})
        rec.update({f"offset_{k}": v for k, v in off.items()})
        rec["offset_fit_ok"] = fit_ok(off) if off else False
        if pages:
            tag = "  (2× catalog P)" if rec["adopted_period_factor"] == 2 else ""
            tag += "  [refit]" if fit is not None and fit["status"] == "ok" else ""
            atlas_page(eph, epochs[sid], tess=lc, title=f"{x['eb_id']} / Gaia DR3 {sid}{tag}",
                       out=out_dir / f"{x['eb_id']}.png")
            if fit is not None and len(fit["timings"]):
                gp = None
                if off and off.get("constrained") and len(g_btjd):
                    from astropy.time import Time
                    gy = float(Time(np.median(g_btjd) + 2457000.0, format="jd", scale="tdb").decimalyear)
                    gp = (gy, off["best_shift"] * eph.period * 1440.0, off["shift_err"] * eph.period * 1440.0)
                oc_figure(fit, out_dir / f"{x['eb_id']}_oc.png", title=x["eb_id"], gaia_point=gp)
        rows.append(rec)

    from .atlas import _rows_to_table
    res = _rows_to_table(rows) if rows else Table()
    res.write(out_dir / "verification.csv", overwrite=True)
    if sector_rows:
        _rows_to_table(sector_rows).write(out_dir / "sector_status.csv", overwrite=True)
    if len(res):
        verification_figure(res, out_dir / "verification_summary.png")
    return res


def verification_figure(res, path):
    """Three panels: offset z-scores, |offset| vs ephemeris error, DR3/catalog period ratio."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    col = "offset_fit_ok" if "offset_fit_ok" in res.colnames else "offset_constrained"
    con = np.asarray(res[col]).astype(bool) if col in res.colnames else np.zeros(len(res), bool)
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
    ax[0].set_title(f"Phase-offset check: {con.sum()} good fits of {len(res)}", fontsize=10)
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
