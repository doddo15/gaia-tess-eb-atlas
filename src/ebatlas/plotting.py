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


def normalize_band(ephem, tab, band):
    sel = (tab["band"] == band) & np.isfinite(tab["flux"])
    t = np.asarray(tab["btjd"][sel]); f = np.asarray(tab["flux"][sel])
    e = np.asarray(tab["flux_err"][sel]); rej = np.asarray(tab["rejected"][sel])
    oot = _oot_mask(ephem, t) & ~rej
    ref = np.median(f[oot]) if oot.any() else np.median(f)
    return t, f / ref, e / ref, rej


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
    """
    fig = plt.figure(figsize=(10, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1, 0.8])
    ax_full = fig.add_subplot(gs[0, :])
    ax_p = fig.add_subplot(gs[1, 0], sharey=ax_full)
    ax_s = fig.add_subplot(gs[1, 1], sharey=ax_full)
    ax_t = fig.add_subplot(gs[2, :])

    if tess is not None:
        tt, tf = tess[0], tess[1]
        tph = ephem.phase(tt)
        for ax, x in ((ax_full, tph), (ax_p, tph), (ax_s, _wrap(tph - ephem.secondary_phase))):
            ax.plot(x, tf / np.nanmedian(tf), ",", color="0.6", alpha=0.5, rasterized=True,
                    label="TESS" if ax is ax_full else None)

    for band in bands:
        if not np.any(gaia_tab["band"] == band):
            continue
        t, f, e, rej = normalize_band(ephem, gaia_tab, band)
        ph, sph = ephem.phase(t), ephem.phase_sigma(t)
        st = BAND_STYLE.get(band, {})
        for ax, x in ((ax_full, ph), (ax_p, ph), (ax_s, _wrap(ph - ephem.secondary_phase))):
            ax.errorbar(x[~rej], f[~rej], yerr=e[~rej], xerr=sph[~rej], fmt="o", lw=0.8,
                        mec="k", mew=0.4, color=st.get("color"), ms=st.get("ms", 4),
                        zorder=st.get("zorder", 3),
                        label=f"Gaia {band} ({(~rej).sum()})" if ax is ax_full else None)
            if rej.any():
                ax.plot(x[rej], f[rej], "o", mfc="none", mec=st.get("color"), ms=st.get("ms", 4),
                        label=f"{band} rejected ({rej.sum()})" if ax is ax_full else None)
            if band == "G" and "rejected_by_variability" in gaia_tab.colnames:
                sel_g = (gaia_tab["band"] == "G") & np.isfinite(gaia_tab["flux"])
                vr = np.asarray(gaia_tab["rejected_by_variability"][sel_g]).astype(bool)
                if vr.any():
                    ax.plot(x[vr], f[vr], "o", mfc="none", mec="k", ms=st.get("ms", 6) + 5, mew=1.0, zorder=6,
                            label=f"G var.-pipeline outlier ({vr.sum()})" if ax is ax_full else None)

    w1 = ephem.primary_width or 0.05
    w2 = ephem.secondary_width or w1
    ax_p.set_xlim(-2 * w1, 2 * w1)
    ax_s.set_xlim(-2 * w2, 2 * w2)
    ax_s.set_xlabel(f"Phase − {ephem.secondary_phase:.3f}"); ax_p.set_xlabel("Phase")
    ax_full.set_xlim(-0.5, 0.5)
    ax_full.set_ylabel("Normalised flux")
    ax_full.set_xlabel("Phase")
    ax_p.set_title("Primary", fontsize=10); ax_s.set_title("Secondary", fontsize=10)
    ax_full.legend(fontsize=8, ncol=2, loc="lower left", framealpha=0.85)
    ax_full.set_title(f"{title}   P = {ephem.period:.6f} d", fontsize=11)

    if np.any(gaia_tab["band"] == "G"):
        t, f, e, rej = normalize_band(ephem, gaia_tab, "G")
        yr = Time(t + 2457000.0, format="jd", scale="tdb").decimalyear
        sc = ax_t.scatter(yr[~rej], f[~rej], c=ephem.phase(t)[~rej], cmap="twilight",
                          vmin=-0.5, vmax=0.5, s=18, edgecolor="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax_t, label="Phase", pad=0.01)
        ax_t.set_xlabel("Year"); ax_t.set_ylabel("Gaia G (norm.)")

    if out:
        fig.savefig(out, dpi=130)
        plt.close(fig)
    return fig
