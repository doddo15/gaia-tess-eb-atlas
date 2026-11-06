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
from .ephemeris import _wrap

# Systematic floor added in quadrature to each sector timing. Statistical errors from
# tens of thousands of points reach ~0.1 s, but template mismatch, spots and TESS
# timing systematics are at the seconds level; on synthetic data the residual
# scatter after shift-and-stack is ~1-3 s.
TIMING_FLOOR_S = 5.0


def _noise(f):
    d = np.diff(f)
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) > 3 else np.nan


def sector_timing(ephem, template, t, f, exptime_s, span, n=401, min_dchi2=25.0):
    """Phase shift of the template that best fits one sector.

    Model: flux = a + b * (template(phase - shift) - 1) with b > 0 (anti-correlated
    alignments are rejected, not fitted). Coarse scan over +-span, then a fine scan around
    the minimum. Status: ok | flat (no shift preferred) | edge (best fit at the window
    edge: offset larger than predicted) | no_fit (no positive-depth alignment) |
    no_eclipse | too_few_points | bad_noise. With span >= 0.5 the search covers the whole
    orbit and has no edge.
    """
    out = {"ok": False, "delta": np.nan, "err": np.nan, "n_points": int(len(t)), "n_in_eclipse": 0,
           "status": "too_few_points", "scale": np.nan, "offset": np.nan, "dchi2_coarse": np.nan}
    if len(t) < 20:
        return out
    sig = _noise(f)
    if not np.isfinite(sig) or sig <= 0:
        out["status"] = "bad_noise"
        return out
    e = np.full(len(t), sig)
    tm = smeared(template, float(exptime_s) / 86400.0 / ephem.period)
    ph = ephem.phase(t)
    periodic = span >= 0.5
    span = min(span, 0.5)
    coarse = np.linspace(-span, span, n)
    c_chi2, _ = _scan(ph, f, e, tm, coarse)
    if not np.isfinite(c_chi2).any():
        out["status"] = "no_fit"
        return out
    kc = int(np.argmin(c_chi2))
    fin = np.isfinite(c_chi2)
    dof0 = max(len(t) - 3, 1); infl0 = max(1.0, c_chi2[kc] / dof0)
    dc = np.where(fin, c_chi2 - c_chi2[kc], np.nan)
    out["dchi2_coarse"] = float(np.nanmax(dc) / infl0)
    if not periodic and kc in (0, n - 1):
        out.update(status="edge", delta=float(coarse[kc]))
        return out
    lo = hi = kc
    while lo > 0 and fin[lo - 1] and dc[lo - 1] <= 25 * infl0:
        lo -= 1
    while hi < n - 1 and fin[hi + 1] and dc[hi + 1] <= 25 * infl0:
        hi += 1
    step_c = coarse[1] - coarse[0]
    half = max(coarse[hi] - coarse[kc] + step_c, coarse[kc] - coarse[lo] + step_c, 3 * step_c)
    shifts = np.linspace(coarse[kc] - half, coarse[kc] + half, n)
    chi2, scale, offs = _scan(ph, f, e, tm, shifts, return_offset=True)
    k = int(np.argmin(chi2))
    if not (0 < k < n - 1) or not np.isfinite(chi2[k]):
        out.update(status="edge" if np.isfinite(chi2[k]) else "no_fit")
        return out
    dof = max(len(t) - 3, 1); infl = max(1.0, chi2[k] / dof)
    d = chi2 - chi2[k]
    lo = hi = k
    while lo > 0 and d[lo - 1] <= 4 * infl:
        lo -= 1
    while hi < n - 1 and d[hi + 1] <= 4 * infl:
        hi += 1
    step = shifts[1] - shifts[0]
    c0, c1, c2 = chi2[k - 1], chi2[k], chi2[k + 1]
    den = c0 - 2 * c1 + c2
    best = shifts[k] + (0.5 * step * (c0 - c2) / den if np.isfinite(den) and den > 0 else 0.0)
    grid = np.linspace(-0.5, 0.5, 4001); tg = tm(grid); base = np.median(tg); depth = (base - tg).max()
    n_in = int(np.sum((base - tm(ph - best)) > 0.2 * depth)) if depth > 0 else 0
    ok = bool(out["dchi2_coarse"] > min_dchi2 and n_in >= 3)
    status = "ok" if ok else ("no_eclipse" if n_in < 3 else "flat")
    out.update(delta=float(_wrap(best)), err=float(max(0.25 * (shifts[hi] - shifts[lo]), step / 2)),
               n_in_eclipse=n_in, scale=float(scale[k]), offset=float(offs[k]), chi2_red=float(chi2[k] / dof),
               ok=ok, status=status)
    return out


def linear_fit(E, T, err, quadratic=False, jitter=True, n_iter=5):
    """Weighted least squares T = T0 + P*(E - Eref) [+ Q/2 (E - Eref)^2], Eref = weighted mean epoch.

    With `jitter`, an extra timing scatter s is fitted by maximum likelihood and added in
    quadrature to every error (instead of inflating by sqrt(chi2_red)); covariances use the
    effective errors. `chi2` is computed with the input errors (before jitter).
    """
    from .errors import jitter_ml
    E = np.asarray(E, float); T = np.asarray(T, float); err = np.asarray(err, float)
    s = 0.0
    for _ in range(n_iter if jitter else 1):
        ee = np.hypot(err, s); w = 1.0 / ee**2
        Eref = float(np.round(np.sum(w * E) / np.sum(w)))
        x = E - Eref
        A = np.column_stack([np.ones_like(x), x] + ([0.5 * x**2] if quadratic else []))
        Aw = A * np.sqrt(w)[:, None]
        p, *_ = np.linalg.lstsq(Aw, T * np.sqrt(w), rcond=None)
        r = T - A @ p
        if not jitter or len(T) <= A.shape[1]:
            break
        s_new = jitter_ml(r * np.sqrt(len(T) / max(len(T) - A.shape[1], 1)), err)   # dof-corrected residuals
        if abs(s_new - s) <= 1e-3 * max(s, np.min(err)):
            s = s_new
            break
        s = s_new
    ee = np.hypot(err, s); w = 1.0 / ee**2
    Aw = A * np.sqrt(w)[:, None]
    cov = np.linalg.inv(Aw.T @ Aw)
    chi2 = float(np.sum((r / err) ** 2)); dof = len(T) - A.shape[1]
    if not jitter and dof > 0:
        cov = cov * max(1.0, chi2 / dof)
    return {"p": p, "cov": cov, "chi2": chi2, "dof": dof, "Eref": Eref, "resid": r, "jitter": s, "err_eff": ee}


def _with_geometry(ephem, geom):
    pick = lambda k, d: geom[k] if np.isfinite(geom.get(k, np.nan)) else d
    return Ephemeris(ephem.period, ephem.t0, ephem.period_err, ephem.t0_err,
                     primary_width=pick("primary_width", ephem.primary_width),
                     secondary_phase=pick("secondary_phase", ephem.secondary_phase),
                     secondary_width=pick("secondary_width", ephem.secondary_width))


def _fit_ephem(ts, eph, jitter=True):
    """Linear fit to absolute timings; epochs re-numbered against the current ephemeris."""
    Ts = np.array([x["T"] for x in ts]); errs = np.array([x["T_err"] for x in ts])
    Es = np.round((Ts - eph.t0) / eph.period)
    fit = linear_fit(Es, Ts, errs, jitter=jitter and len(ts) >= 4)
    return Ephemeris(float(fit["p"][1]), float(fit["p"][0]), float(np.sqrt(fit["cov"][1, 1])),
                     float(np.sqrt(fit["cov"][0, 0])), eph.primary_width, eph.secondary_phase,
                     eph.secondary_width)


def sector_qc(t, f, sec, exp, max_noise_ratio=8.0, max_wild_frac=0.05):
    """{sector: reason} for sectors to drop before timing (others absent).

    bad_normalisation: >max_wild_frac of points more than 50% from the median (e.g. a light
    curve divided by a near-zero median); noisy: per-point noise, scaled to 120 s cadence,
    above max_noise_ratio x the target's median sector (needs >= 3 sectors).
    """
    bad, noise = {}, {}
    for s in np.unique(sec):
        m = sec == s
        fs = f[m]
        if np.mean(np.abs(fs - np.median(fs)) > 0.5) > max_wild_frac or not (0.5 < np.median(fs) < 1.5):
            bad[int(s)] = "bad_normalisation"
            continue
        noise[int(s)] = _noise(fs) * np.sqrt(np.median(exp[m]) / 120.0)
    if len(noise) >= 3:
        ref = np.median(list(noise.values()))
        for s, v in noise.items():
            if not np.isfinite(v) or v > max_noise_ratio * ref:
                bad[s] = "noisy"
    return bad


def refit_ephemeris(catalog_ephem, lc, n_iter=2, max_span=0.25, outlier_sigma=5.0):
    """Refit the ephemeris from all TESS sectors. Returns a dict:

    status ('ok' | 'one_sector' | 'no_timings'), ephem (Ephemeris incl. geometry),
    timings (Table per timed sector), sector_status (every sector, with reason),
    n_sectors, baseline_days, chi2_red (timings vs linear model, before jitter),
    jitter_s (fitted timing scatter, seconds), quad_snr (period-change significance
    against that jitter), period_shift_sigma, n_outliers, n_wide, depth_scale_spread,
    template_exptime.
    """
    lc = as_tesslc(lc)
    t, f, sec, exp = np.asarray(lc.t), np.asarray(lc.flux), np.asarray(lc.sector), np.asarray(lc.exptime, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f, sec, exp = t[ok], f[ok], sec[ok], exp[ok]
    authors = np.asarray(lc.author)[ok]
    result = {"status": "no_timings", "ephem": catalog_ephem, "timings": Table(), "sector_status": Table(),
              "n_sectors": 0, "baseline_days": 0.0, "chi2_red": np.nan, "jitter_s": np.nan, "quad_snr": np.nan,
              "period_shift_sigma": np.nan, "n_outliers": 0, "n_wide": 0, "depth_scale_spread": np.nan,
              "template_exptime": float(np.nanmin(exp)) if exp.size else np.nan}
    qc = sector_qc(t, f, sec, exp)
    status = {}
    for s_, why in qc.items():
        m = sec == s_
        status[s_] = {"sector": int(s_), "author": str(authors[m][0]), "exptime": float(np.median(exp[m])),
                      "n_points": int(m.sum()), "status": why, "n_in_eclipse": 0, "depth_scale": np.nan, "wide": False}
    keep = ~np.isin(sec, list(qc))
    t, f, sec, exp, authors = t[keep], f[keep], sec[keep], exp[keep], authors[keep]
    sectors = np.unique(sec)
    if len(sectors) == 0:
        result["sector_status"] = Table(rows=list(status.values())) if status else Table()
        return result
    tmid = {s_: float(np.median(t[sec == s_])) for s_ in sectors}

    ref = min(sectors, key=lambda s_: abs(tmid[s_] - catalog_ephem.t0))
    m = sec == ref
    eph = _with_geometry(catalog_ephem, measure_eclipses(catalog_ephem, t[m], f[m]))
    template = tess_template(eph, t[m], f[m])
    tmpl_exp = float(np.median(exp[m]))
    order = sorted(sectors, key=lambda s_: abs(tmid[s_] - catalog_ephem.t0))

    timings = {}
    n_wide_first = 0
    for it in range(n_iter + 1):
        timings, st_it = {}, {}
        for s_ in order:
            mm = sec == s_
            sig = float(eph.phase_sigma(tmid[s_]))
            span = float(np.clip(5 * sig, 0.01, max_span)) if it == 0 else float(np.clip(5 * sig, 0.005, 0.05))
            extra = max(float(np.median(exp[mm])) - tmpl_exp, 0.0)
            r = sector_timing(eph, template, t[mm], f[mm], extra, span)
            wide = False
            if r["status"] in ("edge", "no_fit"):                  # larger offset than predicted: search the orbit
                r2 = sector_timing(eph, template, t[mm], f[mm], extra, 0.5)
                if r2["ok"]:
                    r, wide = r2, True
                    n_wide_first += int(it == 0)
            st_it[s_] = {"sector": int(s_), "author": str(authors[mm][0]), "exptime": float(np.median(exp[mm])),
                         "n_points": int(mm.sum()), "status": r["status"], "n_in_eclipse": r["n_in_eclipse"],
                         "depth_scale": r["scale"], "wide": wide}
            if not r["ok"]:
                continue
            E = int(np.round((tmid[s_] - eph.t0) / eph.period))
            T = eph.t0 + eph.period * (E + r["delta"])
            timings[s_] = {"sector": int(s_), "author": str(authors[mm][0]), "exptime": float(np.median(exp[mm])),
                           "n_points": r["n_points"], "n_in_eclipse": r["n_in_eclipse"], "tmid": tmid[s_],
                           "E": E, "T": T, "T_err": float(np.hypot(r["err"] * eph.period, TIMING_FLOOR_S / 86400.0)),
                           "depth_scale": r["scale"], "flux_offset": r["offset"], "wide": wide}
            if it == 0 and len(timings) >= 2:
                eph = _fit_ephem(list(timings.values()), eph, jitter=False)
        # drop outlier timings one at a time. Each timing is judged against a fit made
        # WITHOUT it (incl. that fit's jitter): otherwise one outlier inflates the jitter
        # and hides itself.
        dropped = set()
        while len(timings) >= 5 and len(dropped) < 0.2 * (len(timings) + len(dropped)):
            keys = sorted(timings)
            Ts = np.array([timings[k]["T"] for k in keys]); errs = np.array([timings[k]["T_err"] for k in keys])
            Es = np.round((Ts - eph.t0) / eph.period)
            z = np.zeros(len(keys))
            for j in range(len(keys)):
                o = np.arange(len(keys)) != j
                fj = linear_fit(Es[o], Ts[o], errs[o])
                x = Es[j] - fj["Eref"]
                pred = fj["p"][0] + fj["p"][1] * x
                pvar = fj["cov"][0, 0] + x**2 * fj["cov"][1, 1] + 2 * x * fj["cov"][0, 1]
                z[j] = abs(Ts[j] - pred) / np.sqrt(errs[j] ** 2 + fj["jitter"] ** 2 + pvar)
            j = int(np.argmax(z))
            if z[j] <= outlier_sigma:
                break
            dropped.add(keys[j]); st_it[keys[j]]["status"] = "outlier_timing"; timings.pop(keys[j])
        if len(timings) < 2:
            break
        eph = _fit_ephem(list(timings.values()), eph)
        if it < n_iter:                                                 # shift-and-stack, depth-aligned
            resid = {s_: (x["T"] - (eph.t0 + eph.period * np.round((x["T"] - eph.t0) / eph.period))) / eph.period
                     for s_, x in timings.items()}
            timed = np.isin(sec, list(timings))
            sc = shortest_cadence_mask(np.where(timed, exp, np.inf)) & timed
            shift = np.array([resid.get(s_, 0.0) for s_ in sec[sc]])
            a_s = np.array([timings[s_]["flux_offset"] for s_ in sec[sc]])
            b_s = np.array([timings[s_]["depth_scale"] for s_ in sec[sc]])
            f_al = 1.0 + (f[sc] - a_s) / b_s                             # each sector mapped onto the template's depth
            geom = measure_eclipses(Ephemeris(eph.period, eph.t0), t[sc] - shift * eph.period, f_al)
            eph = _with_geometry(eph, geom)
            template = tess_template(eph, t[sc], f_al, phase_shift=shift)
            tmpl_exp = float(np.nanmin(exp[sc]))
        status.update(st_it)

    result["sector_status"] = Table(rows=[status[k] for k in sorted(status)]) if status else Table()
    if len(timings) == 0:
        return result
    ts = [timings[k] for k in sorted(timings)]
    tab = Table(rows=ts)
    tab["E"] = np.round((tab["T"] - eph.t0) / eph.period).astype(int)
    tab["O_C_min"] = (tab["T"] - (eph.t0 + eph.period * tab["E"])) * 1440.0
    tab["T_err_min"] = tab["T_err"] * 1440.0
    st_vals = list(status.values())
    result.update(template=template, template_ephem=eph, timings=tab, n_sectors=len(ts), template_exptime=tmpl_exp,
                  baseline_days=float(np.ptp(tab["T"])) if len(ts) > 1 else 0.0,
                  n_outliers=sum(1 for v in st_vals if v["status"] == "outlier_timing"),
                  n_wide=int(n_wide_first))   # sectors whose first-pass offset exceeded the predicted window
    # depth-scale spread within a pipeline (contamination is fixed per pipeline; real depth changes are not)
    spreads = []
    for au in set(tab["author"]):
        sc_ = np.asarray(tab["depth_scale"][tab["author"] == au], float)
        if len(sc_) >= 3:
            spreads.append(1.4826 * np.median(np.abs(sc_ - np.median(sc_))) / np.median(sc_))
    result["depth_scale_spread"] = float(max(spreads)) if spreads else np.nan
    if len(ts) < 2:
        result.update(status="one_sector")
        return result
    fit = linear_fit(np.asarray(tab["E"]), np.asarray(tab["T"]), np.asarray(tab["T_err"]), jitter=len(ts) >= 4)
    result["chi2_red"] = fit["chi2"] / fit["dof"] if fit["dof"] > 0 else np.nan
    result["jitter_s"] = float(fit["jitter"] * 86400.0)
    if len(ts) >= 5:
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
