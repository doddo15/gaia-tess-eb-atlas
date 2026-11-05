"""Refit linear ephemerides from per-sector TESS eclipse timings.

Catalog ephemerides (e.g. TEBC, from primary-mission sectors 1-26) drift when folded
over many years of TESS data, smearing the fold and biasing the Gaia comparison.
`refit_ephemeris` measures one eclipse time per sector and fits T(E) = T0 + P*E:

1. Template from the sector nearest the catalog epoch (where the catalog ephemeris is
   most accurate); eclipse geometry measured there.
2. Sectors are timed in order of distance from that epoch, refitting after each one, so
   every prediction is close enough that the cycle count cannot slip.
3. Iterate (shift-and-stack): line up every timed sector by its own measured offset,
   stack the shortest-cadence data into a sharp template, and re-time every sector
   (template smeared by each sector's exposure time). Stacking on the linear ephemeris
   alone would blur the template whenever the period changes, biasing the timings.
4. Report P, T0 (at the weighted-mean epoch, so their covariance is ~0), uncertainties
   inflated by sqrt(chi2_red) when timings scatter more than their errors, O-C per
   sector, and the significance of a quadratic term (a period change).

A sector counts only if its data cover an eclipse; long-period systems may have few.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table

from .ephemeris import Ephemeris, _wrap
from .eclipses import measure_eclipses
from .tess import as_tesslc
from .templates import tess_template, smeared, shortest_cadence_mask, _scan

# Systematic floor added in quadrature to each sector timing. Statistical errors from
# tens of thousands of points reach ~0.1 s, but template mismatch, spots and TESS
# timing systematics are at the seconds level; on synthetic data the residual
# scatter after shift-and-stack is ~1-3 s.
TIMING_FLOOR_S = 5.0


def _noise(f):
    d = np.diff(f)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) > 3 else np.nan


def sector_timing(ephem, template, t, f, exptime_s, span, n=401, min_dchi2=25.0):
    """Phase shift of the template that best fits one sector. Returns dict (ok, delta, err, ...).

    Coarse scan over +-span, then a fine scan around the minimum (step ~2e-5 phase).
    """
    out = {"ok": False, "delta": np.nan, "err": np.nan, "n_points": int(len(t)), "n_in_eclipse": 0}
    if len(t) < 20:
        return out
    sig = _noise(f)
    if not np.isfinite(sig) or sig <= 0:
        return out
    e = np.full(len(t), sig)
    tm = smeared(template, float(exptime_s) / 86400.0 / ephem.period)
    ph = ephem.phase(t)
    coarse = np.linspace(-span, span, n)
    c_chi2, _ = _scan(ph, f, e, tm, coarse)
    kc = int(np.argmin(c_chi2))
    if not (0 < kc < n - 1) or not np.isfinite(c_chi2[kc]):
        return out
    dc = c_chi2 - c_chi2[kc]
    dof0 = max(len(t) - 3, 1); infl0 = max(1.0, c_chi2[kc] / dof0)
    lo = kc
    while lo > 0 and dc[lo - 1] <= 25 * infl0:
        lo -= 1
    hi = kc
    while hi < n - 1 and dc[hi + 1] <= 25 * infl0:
        hi += 1
    half = max(coarse[min(hi + 1, n - 1)] - coarse[kc], coarse[kc] - coarse[max(lo - 1, 0)], 3 * (coarse[1] - coarse[0]))
    shifts = np.linspace(coarse[kc] - half, coarse[kc] + half, n)
    chi2, scale = _scan(ph, f, e, tm, shifts)
    k = int(np.argmin(chi2))
    if not (0 < k < n - 1) or not np.isfinite(chi2[k]):
        return out
    out["dchi2_coarse"] = float(dc.max() / infl0)
    dof = max(len(t) - 3, 1); infl = max(1.0, chi2[k] / dof)
    d = chi2 - chi2[k]
    lo = hi = k
    while lo > 0 and d[lo - 1] <= 4 * infl:
        lo -= 1
    while hi < n - 1 and d[hi + 1] <= 4 * infl:
        hi += 1
    step = shifts[1] - shifts[0]
    # sub-grid refinement: parabola through the minimum and its neighbours
    c0, c1, c2 = chi2[k - 1], chi2[k], chi2[k + 1]
    den = c0 - 2 * c1 + c2
    best = shifts[k] + (0.5 * step * (c0 - c2) / den if den > 0 else 0.0)
    grid = np.linspace(-0.5, 0.5, 4001); tg = tm(grid); base = np.median(tg); depth = (base - tg).max()
    n_in = int(np.sum((base - tm(ph - best)) > 0.2 * depth)) if depth > 0 else 0
    out.update(delta=float(best), err=float(max(0.25 * (shifts[hi] - shifts[lo]), step / 2)),
               n_in_eclipse=n_in, scale=float(scale[k]), chi2_red=float(chi2[k] / dof),
               ok=bool(out["dchi2_coarse"] > min_dchi2 and n_in >= 3 and scale[k] > 0))
    return out


def linear_fit(E, T, err, quadratic=False):
    """Weighted least squares T = T0 + P*(E - Eref) [+ Q/2 (E - Eref)^2], Eref = weighted mean epoch (integer)."""
    E = np.asarray(E, float); T = np.asarray(T, float); w = 1.0 / np.asarray(err, float) ** 2
    Eref = float(np.round(np.sum(w * E) / np.sum(w)))
    x = E - Eref
    cols = [np.ones_like(x), x] + ([0.5 * x**2] if quadratic else [])
    A = np.column_stack(cols)
    Aw = A * np.sqrt(w)[:, None]
    p, *_ = np.linalg.lstsq(Aw, T * np.sqrt(w), rcond=None)
    r = T - A @ p
    chi2 = float(np.sum(w * r**2)); dof = len(T) - A.shape[1]
    cov = np.linalg.inv(Aw.T @ Aw)
    infl = max(1.0, chi2 / dof) if dof > 0 else 1.0
    return {"p": p, "cov": cov * infl, "chi2": chi2, "dof": dof, "Eref": Eref, "resid": r}


def _fit_ephem(ts, eph):
    """Linear fit to absolute timings; epochs re-numbered against the current ephemeris."""
    Ts = np.array([x["T"] for x in ts]); errs = np.array([x["T_err"] for x in ts])
    Es = np.round((Ts - eph.t0) / eph.period)
    fit = linear_fit(Es, Ts, errs)
    return Ephemeris(float(fit["p"][1]), float(fit["p"][0]), float(np.sqrt(fit["cov"][1, 1])),
                     float(np.sqrt(fit["cov"][0, 0])), eph.primary_width, eph.secondary_phase,
                     eph.secondary_width)


def _with_geometry(ephem, geom):
    pick = lambda k, d: geom[k] if np.isfinite(geom.get(k, np.nan)) else d
    return Ephemeris(ephem.period, ephem.t0, ephem.period_err, ephem.t0_err,
                     primary_width=pick("primary_width", ephem.primary_width),
                     secondary_phase=pick("secondary_phase", ephem.secondary_phase),
                     secondary_width=pick("secondary_width", ephem.secondary_width))


def refit_ephemeris(catalog_ephem, lc, n_iter=2, max_span=0.25):
    """Refit the ephemeris from all TESS sectors. Returns a dict:

    status ('ok' | 'one_sector' | 'no_timings'), ephem (Ephemeris incl. geometry),
    timings (Table per sector), n_sectors, baseline_days, chi2_red, quad_snr,
    period_shift_sigma (refit vs catalog period, in combined sigma), template_exptime.
    """
    lc = as_tesslc(lc)
    t, f, sec, exp = np.asarray(lc.t), np.asarray(lc.flux), np.asarray(lc.sector), np.asarray(lc.exptime, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, sec, exp = t[ok], f[ok], sec[ok], exp[ok]
    authors = np.asarray(lc.author)[ok]
    sectors = np.unique(sec)
    tmid = {s: float(np.median(t[sec == s])) for s in sectors}
    result = {"status": "no_timings", "ephem": catalog_ephem, "timings": Table(), "n_sectors": 0,
              "baseline_days": 0.0, "chi2_red": np.nan, "quad_snr": np.nan, "period_shift_sigma": np.nan,
              "template_exptime": float(np.nanmin(exp)) if exp.size else np.nan}
    if len(sectors) == 0:
        return result

    # 1. reference sector, geometry and template
    ref = min(sectors, key=lambda s: abs(tmid[s] - catalog_ephem.t0))
    m = sec == ref
    eph = _with_geometry(catalog_ephem, measure_eclipses(catalog_ephem, t[m], f[m]))
    template = tess_template(eph, t[m], f[m])
    tmpl_exp = float(np.median(exp[m]))

    order = sorted(sectors, key=lambda s: abs(tmid[s] - catalog_ephem.t0))
    timings = {}
    for it in range(n_iter + 1):
        timings = {}
        for s in order:
            mm = sec == s
            sig = float(eph.phase_sigma(tmid[s]))
            span = float(np.clip(5 * sig, 0.01, max_span)) if it == 0 else float(np.clip(5 * sig, 0.005, 0.05))
            extra = max(float(np.median(exp[mm])) - tmpl_exp, 0.0)        # smear only beyond the template's own
            r = sector_timing(eph, template, t[mm], f[mm], extra, span)
            if not r["ok"]:
                continue
            E = int(np.round((tmid[s] - eph.t0) / eph.period))
            T = eph.t0 + eph.period * (E + r["delta"])
            timings[s] = {"sector": int(s), "author": str(authors[mm][0]), "exptime": float(np.median(exp[mm])),
                          "n_points": r["n_points"], "n_in_eclipse": r["n_in_eclipse"], "tmid": tmid[s],
                          "E": E, "T": T, "T_err": float(np.hypot(r["err"] * eph.period, TIMING_FLOOR_S / 86400.0))}
            if it == 0 and len(timings) >= 2:                                  # progressive update
                eph = _fit_ephem(list(timings.values()), eph)
        if len(timings) < 2:
            break
        eph = _fit_ephem(list(timings.values()), eph)
        if it < n_iter:                                               # shift-and-stack template
            resid = {s_: (x["T"] - (eph.t0 + eph.period * np.round((x["T"] - eph.t0) / eph.period))) / eph.period
                     for s_, x in timings.items()}
            timed = np.isin(sec, list(timings))
            sc = shortest_cadence_mask(np.where(timed, exp, np.inf)) & timed
            shift = np.array([resid.get(s_, 0.0) for s_ in sec[sc]])
            geom = measure_eclipses(Ephemeris(eph.period, eph.t0), t[sc] - shift * eph.period, f[sc])
            eph = _with_geometry(eph, geom)
            template = tess_template(eph, t[sc], f[sc], phase_shift=shift)
            tmpl_exp = float(np.nanmin(exp[sc]))

    if len(timings) == 0:
        return result
    ts = list(timings.values())
    tab = Table(rows=ts)
    tab["E"] = np.round((tab["T"] - eph.t0) / eph.period).astype(int)
    tab["O_C_min"] = (tab["T"] - (eph.t0 + eph.period * tab["E"])) * 1440.0
    tab["T_err_min"] = tab["T_err"] * 1440.0
    result.update(template=template, template_ephem=eph)
    result.update(timings=tab, n_sectors=len(ts), template_exptime=tmpl_exp,
                  baseline_days=float(np.ptp(tab["T"])) if len(ts) > 1 else 0.0)
    if len(ts) < 2:
        result.update(status="one_sector")
        return result
    fit = linear_fit(np.asarray(tab["E"]), np.asarray(tab["T"]), np.asarray(tab["T_err"]))
    result["chi2_red"] = fit["chi2"] / fit["dof"] if fit["dof"] > 0 else np.nan
    if len(ts) >= 4:
        q = linear_fit(np.asarray(tab["E"]), np.asarray(tab["T"]), np.asarray(tab["T_err"]), quadratic=True)
        result["quad_snr"] = float(q["p"][2] / np.sqrt(q["cov"][2, 2]))
    sp = np.hypot(catalog_ephem.period_err, eph.period_err)
    result.update(status="ok", ephem=eph,
                  period_shift_sigma=float((eph.period - catalog_ephem.period) / sp) if sp > 0 else np.nan)
    return result


def oc_figure(refit, path, title="", gaia_point=None):
    """O-C diagram of TESS sector timings (and optionally the Gaia-era point)."""
    import matplotlib.pyplot as plt
    from astropy.time import Time

    tab = refit["timings"]
    fig, ax = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    if len(tab):
        yr = Time(np.asarray(tab["T"]) + 2457000.0, format="jd", scale="tdb").decimalyear
        for author, mk in (("SPOC", "o"), ("TESS-SPOC", "s"), ("QLP", "D")):
            sel = np.asarray(tab["author"]) == author
            if sel.any():
                ax.errorbar(yr[sel], tab["O_C_min"][sel], yerr=tab["T_err_min"][sel], fmt=mk, ms=5,
                            label=f"TESS {author}", capsize=2)
        other = ~np.isin(np.asarray(tab["author"]), ["SPOC", "TESS-SPOC", "QLP"])
        if other.any():
            ax.errorbar(yr[other], tab["O_C_min"][other], yerr=tab["T_err_min"][other], fmt="^", ms=5, label="TESS other")
    if gaia_point is not None:
        gy, goc, gerr = gaia_point
        ax.errorbar([gy], [goc], yerr=[gerr], fmt="*", ms=13, color="#d1495b", mec="k", label="Gaia DR3 (fold offset)")
    ax.axhline(0, color="0.5", lw=0.8)
    e = refit["ephem"]
    ax.set_xlabel("Year"); ax.set_ylabel("O − C [min]")
    ax.set_title(f"{title}  P = {e.period:.7f} ± {e.period_err:.1e} d", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120); plt.close(fig)
