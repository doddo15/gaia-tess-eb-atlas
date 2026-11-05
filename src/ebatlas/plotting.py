"""Atlas pages: Gaia epochs phase-folded on the TESS ephemeris."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .ephemeris import _wrap
from astropy.time import Time

BAND_STYLE = {"G": dict(color="#1f5fbf", ms=6, zorder=5),
              "BP": dict(color="#2a9d8f", ms=4, zorder=4),
              "RP": dict(color="#d1495b", ms=4, zorder=4)}


def _oot_mask(ephem, t):
    """Points safely out of eclipse (or all points if widths unknown)."""
    m = np.ones(len(t), bool)
    if ephem.primary_width:
        m &= ~ephem.in_primary(t, "possible")
    if ephem.secondary_width:
        m &= ~ephem.in_secondary(t, "possible")
    return m if m.sum() >= 5 else np.ones(len(t), bool)


# Normalised fluxes above this are treated as rejected. Eclipses only remove light, so a
# point at twice the median cannot be eclipse signal; such points are usually BP/RP
# contamination or failed photometry that slipped past the pipeline flags.
MAX_NORM_FLUX = 2.0


def normalize_band(ephem, tab, band, max_norm=MAX_NORM_FLUX):
    """Normalised (t, flux, err, rejected) for one band.

    Flux is divided by the median of non-rejected points outside eclipse (all
    non-rejected points when eclipse widths are unknown or cover most of the orbit).
    Points brighter than `max_norm` after normalising are added to `rejected`.
    """
    sel = (tab["band"] == band) & np.isfinite(tab["flux"])
    t = np.asarray(tab["btjd"][sel]); f = np.asarray(tab["flux"][sel])
    e = np.asarray(tab["flux_err"][sel]); rej = np.asarray(tab["rejected"][sel]).astype(bool)
    oot = _oot_mask(ephem, t) & ~rej
    ref = np.median(f[oot]) if oot.any() else (np.median(f[~rej]) if (~rej).any() else np.median(f))
    fn = f / ref
    return t, fn, e / ref, rej | (fn > max_norm)


def normalize_tess(ephem, t, flux):
    """TESS flux divided by its out-of-eclipse median: the same reference as the Gaia bands.

    (A plain median would include the eclipses, pushing the curve above the Gaia
    points for deep or contact systems.)
    """
    ok = np.isfinite(flux)
    oot = _oot_mask(ephem, t) & ok
    ref = np.nanmedian(flux[oot]) if oot.sum() >= 5 else np.nanmedian(flux[ok])
    return flux / ref


def _display_limits(tess_flux, gaia_good):
    """y-limits from the TESS light curve (0.2-99.8 percentiles) and all good Gaia points."""
    vals = []
    if tess_flux is not None and np.isfinite(tess_flux).any():
        vals += list(np.nanpercentile(tess_flux, [0.2, 99.8]))
    for f in gaia_good:
        if len(f):
            vals += [np.nanmin(f), np.nanmax(f)]
    if not vals:
        return 0.0, 1.5
    lo, hi = min(vals), max(vals)
    pad = 0.06 * max(hi - lo, 0.02)
    return lo - pad, hi + pad


def summarize(ephem, tab, band="G"):
    """Per-target atlas statistics: how many Gaia epochs caught an eclipse, and how strongly."""
    t, f, e, rej = normalize_band(ephem, tab, band)
    good = ~rej
    row = {"n_epochs": int(good.sum()), "n_rejected": int(rej.sum()),
           "max_phase_sigma": float(np.max(ephem.phase_sigma(t))) if len(t) else np.nan,
           "ephem_reliable_frac": float(np.mean(ephem.is_reliable_at(t))) if len(t) else np.nan}
    oot = _oot_mask(ephem, t) & good
    scatter = 1.4826 * np.median(np.abs(f[oot] - np.median(f[oot]))) if oot.sum() > 3 else np.nan
    for name, fn in (("primary", ephem.in_primary), ("secondary", ephem.in_secondary)):
        try:
            sec = fn(t, "secure"); pos = fn(t, "possible")
        except ValueError:
            row.update({f"n_{name}_secure": -1, f"n_{name}_possible": -1,
                        f"{name}_rej_in_eclipse": -1, f"{name}_depth_snr": np.nan})
            continue
        row[f"n_{name}_secure"] = int((sec & good).sum())
        row[f"n_{name}_possible"] = int((pos & good).sum())
        # In-eclipse points thrown out by Gaia's outlier rejection: a known pitfall.
        row[f"{name}_rej_in_eclipse"] = int((pos & rej).sum())
        # DR4: in-eclipse G transits whose AF CCDs Gaia rejected from the *mean* photometry
        # (phot_ccd_proc_flags bit 7). The epoch flux is kept; this records that Gaia saw an outlier.
        if "n_ccd_outlier" in tab.colnames:
            selb = (tab["band"] == band) & np.isfinite(tab["flux"])
            nout = np.asarray(tab["n_ccd_outlier"][selb])
            row[f"{name}_gaia_outlier_transits"] = int(((nout > 0) & pos & good).sum())
        # DR3: in-eclipse points the variability pipeline rejected as outliers (kept in our analysis).
        if "rejected_by_variability" in tab.colnames:
            selb = (tab["band"] == band) & np.isfinite(tab["flux"])
            vr = np.asarray(tab["rejected_by_variability"][selb]).astype(bool)
            row[f"{name}_var_rejected_in_eclipse"] = int((vr & pos & good).sum())
        ins = sec & good
        if ins.any() and np.isfinite(scatter) and scatter > 0:
            depth = 1.0 - np.median(f[ins])
            row[f"{name}_depth_snr"] = float(depth / (scatter / np.sqrt(ins.sum())))
        else:
            row[f"{name}_depth_snr"] = np.nan
    return row


def atlas_page(ephem, gaia_tab, tess=None, title="", bands=("G", "BP", "RP"), out=None):
    """Three-row atlas page.

    Row 1: TESS light curve folded on the ephemeris, Gaia epochs on top.
    Row 2: zooms on primary and secondary eclipse.
    Row 3: Gaia G epochs vs calendar time, coloured by phase (shows the sampling).

    The y-range comes from the TESS light curve and the good Gaia points; anything
    outside it (usually rejected points) is drawn as a triangle at the plot edge.
    """
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 0.8])
    ax_full = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0], sharey=ax_full)
    ax_s = fig.add_subplot(gs[1, 1], sharey=ax_full)
    ax_t = fig.add_subplot(gs[2, :])
    sec = ephem.secondary_phase
    panels = lambda ph: ((ax_full, ph), (ax_p, ph), (ax_s, _wrap(ph - sec)))

    tnorm = None
    if tess is not None:
        tt, tf = np.asarray(tess[0], float), np.asarray(tess[1], float)
        tnorm = normalize_tess(ephem, tt, tf)
    data = {}
    for band in bands:
        if np.any(gaia_tab["band"] == band):
            data[band] = normalize_band(ephem, gaia_tab, band)
    lo, hi = _display_limits(tnorm, [d[1][~d[3]] for d in data.values()])

    if tnorm is not None:
        tph = ephem.phase(tt)
        inr = (tnorm > lo) & (tnorm < hi)
        for ax, x in panels(tph):
            ax.plot(x[inr], tnorm[inr], ",", color="0.6", alpha=0.5, rasterized=True,
                    label="TESS" if ax is ax_full else None)

    n_off = 0
    for band, (t, f, e, rej) in data.items():
        ph, sph = ephem.phase(t), ephem.phase_sigma(t)
        st = BAND_STYLE.get(band, {})
        inr = (f > lo) & (f < hi)
        good, bad = ~rej & inr, rej & inr
        off_hi, off_lo = f >= hi, f <= lo
        n_off += int(off_hi.sum() + off_lo.sum())
        for ax, x in panels(ph):
            ax.errorbar(x[good], f[good], yerr=e[good], xerr=sph[good], fmt="o", lw=0.8,
                        mec="k", mew=0.4, color=st.get("color"), ms=st.get("ms", 4),
                        zorder=st.get("zorder", 3),
                        label=f"Gaia {band} ({(~rej).sum()})" if ax is ax_full else None)
            if bad.any():
                ax.plot(x[bad], f[bad], "o", mfc="none", mec=st.get("color"), ms=st.get("ms", 4),
                        label=f"{band} rejected ({rej.sum()})" if ax is ax_full else None)
            edge = 0.015 * (hi - lo)
            if off_hi.any():
                ax.plot(x[off_hi], np.full(off_hi.sum(), hi - edge), "^", color=st.get("color"), ms=6,
                        mec="k", mew=0.4, zorder=7, clip_on=False)
            if off_lo.any():
                ax.plot(x[off_lo], np.full(off_lo.sum(), lo + edge), "v", color=st.get("color"), ms=6,
                        mec="k", mew=0.4, zorder=7, clip_on=False)
            if band == "G" and "rejected_by_variability" in gaia_tab.colnames:
                sel_g = (gaia_tab["band"] == "G") & np.isfinite(gaia_tab["flux"])
                vr = np.asarray(gaia_tab["rejected_by_variability"][sel_g]).astype(bool) & inr
                if vr.any():
                    ax.plot(x[vr], f[vr], "o", mfc="none", mec="k", ms=st.get("ms", 6) + 5, mew=1.0, zorder=6,
                            label=f"G var.-pipeline outlier ({vr.sum()})" if ax is ax_full else None)
    if n_off:
        ax_full.plot([], [], "^", color="0.5", mec="k", mew=0.4, label=f"off-scale ({n_off})")
    ax_full.set_ylim(lo, hi)

    w1 = ephem.primary_width or 0.05
    w2 = ephem.secondary_width or w1
    ax_p.set_xlim(-min(2 * w1, 0.5), min(2 * w1, 0.5))
    ax_s.set_xlim(-min(2 * w2, 0.5), min(2 * w2, 0.5))
    ax_s.set_xlabel(f"Phase − {sec:.3f}"); ax_p.set_xlabel("Phase")
    ax_full.set_xlim(-0.5, 0.5)
    ax_full.set_ylabel("Normalised flux")
    ax_full.set_xlabel("Phase")
    ax_p.set_title("Primary", fontsize=10); ax_s.set_title("Secondary", fontsize=10)
    ax_full.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.85)
    ax_full.set_title(f"{title}   P = {ephem.period:.6f} d", fontsize=11)

    if "G" in data:
        t, f, e, rej = data["G"]
        yr = Time(t + 2457000.0, format="jd", scale="tdb").decimalyear
        sc = ax_t.scatter(yr[~rej], f[~rej], c=ephem.phase(t)[~rej], cmap="twilight",
                          vmin=-0.5, vmax=0.5, s=18, edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax_t, label="Phase", pad=0.01)
        ax_t.set_xlabel("Year"); ax_t.set_ylabel("Gaia G (norm.)")

    if out:
        _save(fig, out, title)
        plt.close(fig)
    return fig


def _save(fig, out, title=""):
    """Save; if constrained_layout gives up (axes collapse), retry without it and say which page."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out, dpi=130)
    if any("constrained_layout not applied" in str(w.message) for w in caught):
        fig.set_layout_engine(None)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        warnings.warn(f"layout fallback used for page: {title or out}")
