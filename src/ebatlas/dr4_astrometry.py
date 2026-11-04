"""Gaia DR4 epoch astrometry (DataLink retrieval type EPOCH_ASTROMETRY).

Written against the DR4 draft data model (CU9, section 3.1) and a
DR4_RC3 pre-release VOTable. Key facts, each checked on that sample:

* One row per (source_id, transit_id); per-CCD quantities are length-10
  arrays indexed SM=0, AF1=1, ..., AF9=9.
* obs_time_tcb: int64 nanoseconds since J2010.0 (TCB) **at Gaia**, per CCD.
  J2010.0 = JD 2455197.5, i.e. the same origin as DR3 "Gaia time".
* obs_time_bary_corr: TCB(barycentre) - TCB(Gaia) in ns, for AF4.
  On the sample it matches astropy's Roemer delay to <4.4 s (Gaia vs
  geocentre), confirming its meaning. ~10% of values are NaN; we fill
  those with the geocentric estimate and flag them.
* The VOTable TIMESYS element claims refposition="BARYCENTER". That
  conflicts with the data model text; we follow the data model.
* centroid_pos_ac is empty in DR4 (1-D astrometry only).
* centroid_pos_al has rare gross outliers (data-model known issue);
  use used_by_agis_al and a robust clip.
* SM centroid errors are underestimated (data model); AF-only by default.
"""
from __future__ import annotations

import warnings

import numpy as np
from astropy.table import Table

from .ephemeris import BTJD_OFFSET, GAIA_TIME_OFFSET

NS_PER_DAY = 86400e9
AF4 = 4


def decode_transit_id(transit_id):
    """Decode transit_id bit fields (data model 3.1)."""
    tid = np.asarray(transit_id, dtype=np.int64)
    return {
        "obmt_ns": 204800 * ((tid >> 17) & 0x000003FFFFFFFFFF),
        "fov": 1 + ((tid >> 15) & 0x03),          # 1 preceding, 2 following
        "ccd_row": (tid >> 12) & 0x07,
        "ac_pixel_af1": tid & 0x0FFF,
    }


def _stack(col, dtype=float, width=10):
    """Turn a column of variable-length arrays into an (N, width) array."""
    out = np.full((len(col), width), np.nan if dtype is float else 0, dtype=dtype)
    for i, v in enumerate(col):
        a = np.ma.filled(np.ma.asarray(v), np.nan if dtype is float else 0).astype(dtype)
        out[i, : min(width, a.size)] = a[:width]
    return out


def read_epoch_astrometry(src):
    """Read a DR4 EPOCH_ASTROMETRY VOTable (path or astropy Table)."""
    if isinstance(src, Table):
        return src
    from astropy.io.votable import parse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parse(src).get_first_table().to_table()


def geocentric_bary_corr_s(t_gaia_days, ra, dec):
    """Approximate TCB(bary) - TCB(Gaia) [s] using the geocentre (good to ~5 s)."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time

    t = Time(np.full(np.shape(t_gaia_days), GAIA_TIME_OFFSET), t_gaia_days, format="jd", scale="tcb")
    geo = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    c = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    return t.light_travel_time(c, kind="barycentric", location=geo).to_value(u.s)


def transit_times(tab):
    """Per-transit barycentric times at AF4.

    Returns Table(source_id, transit_id, fov, t_gaia_bary, btjd, bary_corr_s, bary_corr_estimated)
    where t_gaia_bary is days since J2010.0 TCB at the barycentre (= DR3-style Gaia time).
    """
    from .ephemeris import gaia_time_to_btjd

    ns = _stack(tab["obs_time_tcb"], dtype=np.int64)[:, AF4]
    t_at_gaia = ns / NS_PER_DAY
    bc = np.ma.filled(np.ma.asarray(tab["obs_time_bary_corr"]).astype(float), np.nan) / 1e9
    est = ~np.isfinite(bc)
    if est.any():
        bc[est] = geocentric_bary_corr_s(t_at_gaia[est], np.asarray(tab["ra0"])[est],
                                         np.asarray(tab["dec0"])[est])
    t_bary = t_at_gaia + bc / 86400.0
    out = Table()
    out["source_id"] = np.asarray(tab["source_id"])
    out["transit_id"] = np.asarray(tab["transit_id"])
    out["fov"] = decode_transit_id(tab["transit_id"])["fov"]
    out["t_gaia_bary"] = t_bary
    out["btjd"] = gaia_time_to_btjd(t_bary)
    out["bary_corr_s"] = bc
    out["bary_corr_estimated"] = est
    return out


def al_observations(tab, ccds=range(1, 10), only_used=True, max_abs_w_mas=1e5, include_excess_noise=True):
    """Flatten to one row per usable AF CCD observation.

    Columns: source_id, transit_id, t_years (TCB at Gaia, years since J2016.0
    for conditioning), w (mas), w_err (mas, incl. source excess noise),
    psi (rad), f_al (parallax factor AL).

    include_excess_noise=False drops agis_source_excess_noise from the
    weights, so that unmodelled motion (e.g. orbital wobble) raises the UWE
    instead of being absorbed. Use this for binarity diagnostics.
    """
    ccds = list(ccds)
    t_ns = _stack(tab["obs_time_tcb"], dtype=np.int64)[:, ccds]
    w = _stack(tab["centroid_pos_al"])[:, ccds]
    werr = _stack(tab["centroid_pos_error_al"])[:, ccds]
    psi = np.deg2rad(_stack(tab["scan_pos_angle"])[:, ccds])
    used = _stack(tab["used_by_agis_al"], dtype=bool)[:, ccds]
    exc = np.nan_to_num(np.ma.filled(np.ma.asarray(tab["agis_source_excess_noise"]).astype(float), 0.0))
    if not include_excess_noise:
        # Without the source excess noise, misfit from binarity/blending shows up in UWE.
        exc = np.zeros_like(exc)
    f_al = np.ma.filled(np.ma.asarray(tab["parallax_factor_al"]).astype(float), np.nan)

    n, k = w.shape
    rep = lambda a: np.repeat(np.asarray(a), k)
    ok = np.isfinite(w) & np.isfinite(werr) & (np.abs(w) < max_abs_w_mas)
    if only_used:
        ok &= used
    ok = ok.ravel()
    J2016_DAYS = 2457389.0 - GAIA_TIME_OFFSET
    out = Table()
    out["source_id"] = rep(tab["source_id"])[ok]
    out["transit_id"] = rep(tab["transit_id"])[ok]
    out["t_years"] = ((t_ns.ravel() / NS_PER_DAY) - J2016_DAYS)[ok] / 365.25
    out["w"] = w.ravel()[ok]
    out["w_err"] = np.hypot(werr.ravel(), rep(exc))[ok]
    out["psi"] = psi.ravel()[ok]
    out["f_al"] = rep(f_al)[ok]
    return out


def fit_five_parameter_al(obs, clip=5.0, n_iter=3):
    """Weighted linear 5-parameter (single-star) fit to AL data for ONE source.

    Model: w = sin(psi) * (da* + mua* t) + cos(psi) * (dd + mud t) + plx * f_al
    (data model: sin/cos of the scan angle are dw/dalpha*, dw/ddec.)
    Returns dict with params [mas, mas/yr], uwe (sqrt of reduced chi^2),
    n_obs, n_transits, and per-observation residuals.
    """
    s, c, t = np.sin(obs["psi"]), np.cos(obs["psi"]), np.asarray(obs["t_years"])
    A = np.column_stack([s, c, np.asarray(obs["f_al"]), s * t, c * t])
    y = np.asarray(obs["w"], float); e = np.asarray(obs["w_err"], float)
    keep = np.isfinite(y) & np.isfinite(e) & (e > 0) & np.all(np.isfinite(A), axis=1)
    for _ in range(n_iter):
        Aw, yw = A[keep] / e[keep, None], y[keep] / e[keep]
        p, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
        r = (y - A @ p) / e
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        new = keep & (np.abs(r) < clip * max(mad, 1.0))
        if new.sum() == keep.sum():
            break
        keep = new
    dof = keep.sum() - 5
    chi2 = float(np.sum(r[keep] ** 2))
    cov = np.linalg.inv(Aw.T @ Aw)
    names = ["ra_offset", "dec_offset", "parallax", "pmra", "pmdec"]
    return {
        "params": dict(zip(names, p)), "errors": dict(zip(names, np.sqrt(np.diag(cov)))),
        "uwe": float(np.sqrt(chi2 / dof)) if dof > 0 else np.nan,
        "n_obs": int(keep.sum()), "n_clipped": int((~keep).sum()),
        "n_transits": int(len(np.unique(np.asarray(obs["transit_id"])[keep]))),
        "residuals": y - A @ p, "keep": keep,
    }


def _design(obs, period_days=None, e=0.0):
    s, c, t = np.sin(obs["psi"]), np.cos(obs["psi"]), np.asarray(obs["t_years"])
    cols = [s, c, np.asarray(obs["f_al"]), s * t, c * t]
    if period_days is not None:
        if e != 0.0:
            raise NotImplementedError("Eccentric orbits: scan e and T as nonlinear parameters.")
        ph = 2 * np.pi * t * 365.25 / period_days
        X, Y = np.cos(ph), np.sin(ph)
        # Thiele-Innes (circular): dra* = B X + G Y, ddec = A X + F Y
        cols += [s * X, s * Y, c * X, c * Y]
    return np.column_stack(cols)


def astrometric_periodogram(obs, periods_days):
    """Delta-chi^2 of a 9-parameter (5 + circular Thiele-Innes) fit vs the 5-parameter fit.

    Linear at fixed period, so it is fast and avoids the projection
    problems of running Lomb-Scargle on 1-D AL residuals. Returns
    (delta_chi2 array, chi2_5p). Peaks are candidates, not detections:
    check against Gaia scanning-law aliases (~63 d precession, 1 yr) and
    the DR4 NSS tables.
    """
    y = np.asarray(obs["w"], float); e = np.asarray(obs["w_err"], float)
    def chi2(A):
        Aw = A / e[:, None]
        p, *_ = np.linalg.lstsq(Aw, y / e, rcond=None)
        return float(np.sum(((y - A @ p) / e) ** 2)), p
    c5, _ = chi2(_design(obs))
    dchi = np.array([c5 - chi2(_design(obs, P))[0] for P in periods_days])
    return dchi, c5


def photocentre_orbit_fit(obs, period_days):
    """Best 9-parameter circular-orbit fit at a fixed period. Returns dict incl. semi-major axis (mas)."""
    y = np.asarray(obs["w"], float); e = np.asarray(obs["w_err"], float)
    A = _design(obs, period_days)
    p, *_ = np.linalg.lstsq(A / e[:, None], y / e, rcond=None)
    B, G, A_, F = p[5], p[6], p[7], p[8]
    # For a circular orbit, X/Y amplitudes give the semi-major axis: a^2 = (A^2+B^2+F^2+G^2)/2 * (1 + ...);
    # report the rms photocentre amplitude, which equals a for face-on and is bounded by a otherwise.
    amp = np.sqrt((A_**2 + B**2 + F**2 + G**2) / 2)
    dof = len(y) - 9
    return {"params": p, "rms_photocentre_amp_mas": float(amp),
            "uwe": float(np.sqrt(np.sum(((y - A @ p) / e) ** 2) / dof)), "model": A @ p}


def _kepler_XY(t_years, P_days, e, T0_phase):
    M = 2 * np.pi * (np.asarray(t_years) * 365.25 / P_days - T0_phase)
    E = M.copy()
    for _ in range(40):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)


def orbit_grid_fit(obs, periods_days, eccs=(0.0, 0.2, 0.4, 0.6, 0.8), n_t0=24):
    """Brute-force (P, e, T0) grid with linear Thiele-Innes constants at each node.

    Returns the best node as a dict (period, e, t0_phase, uwe, chi2, model).
    Coarse by design: use it to seed a proper nonlinear fit, and treat
    periods longer than the ~5.5-yr DR4 baseline as poorly constrained.
    """
    y = np.asarray(obs["w"], float); err = np.asarray(obs["w_err"], float)
    s, c = np.sin(obs["psi"]), np.cos(obs["psi"])
    base = _design(obs)
    best = {"chi2": np.inf}
    for P in periods_days:
        for e in eccs:
            for T0 in np.linspace(0, 1, n_t0, endpoint=False):
                X, Y = _kepler_XY(obs["t_years"], P, e, T0)
                A = np.column_stack([base, s * X, s * Y, c * X, c * Y])
                p, *_ = np.linalg.lstsq(A / err[:, None], y / err, rcond=None)
                chi2 = float(np.sum(((y - A @ p) / err) ** 2))
                if chi2 < best["chi2"]:
                    best = {"chi2": chi2, "period": float(P), "e": float(e), "t0_phase": float(T0),
                            "params": p, "model": A @ p}
    best["uwe"] = float(np.sqrt(best["chi2"] / (len(y) - 9)))
    return best


def fetch_dr4_epoch_astrometry(source_ids, data_release="Gaia DR4", chunk=500,
                               structures=("DATAMODEL_GAIA", "DATAMODEL_STANDARD")):
    """DataLink fetch (EPOCH_ASTROMETRY). The reader follows the DATAMODEL_GAIA layout
    documented in the draft data model. TODO(relay): confirm data_release on release day."""
    from astropy.table import vstack
    from . import datalink

    out = []
    ids = [int(s) for s in source_ids]
    for i in range(0, len(ids), chunk):
        res, _ = datalink.load(ids[i:i + chunk], "EPOCH_ASTROMETRY", data_release, structures=structures)
        out += [t for _k, t in datalink.iter_tables(res)]
    return vstack(out) if out else Table()
