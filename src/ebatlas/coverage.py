"""TESS coverage audit: which sectors a star fell on, what MAST has, and what we used.

Three layers per target:
  observed  -- sectors the star landed on a CCD (tess-point; offline pointing tables,
               which also include *planned* sectors, so we cut at `max_sector`);
  available -- light-curve products at MAST per sector and pipeline (lightkurve search,
               no download);
  used      -- what the refit did with each downloaded sector (ephemeris_fit sector
               status: timed, or why not).
The gaps between layers explain short timing baselines: a sector can be observed but
have no product (faint star, no 2-min target, pipeline backlog), or have a product but
no usable timing (no eclipse in the sector, bad data).
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table

from .tess import _sector_of, AUTHOR_PRIORITY


def observed_sectors(ra, dec, max_sector=None):
    """Sectors whose footprint contains (ra, dec), from tess-point. Up to max_sector if given."""
    import tess_stars2px as t2p
    r = t2p.tess_stars2px_function_entry(np.array([1]), np.array([float(ra)]), np.array([float(dec)]))
    secs = sorted({int(s) for s in np.atleast_1d(r[3]) if int(s) > 0})
    return [s for s in secs if max_sector is None or s <= max_sector]


def available_products(tic_id):
    """[{sector, author, exptime}] for every TESS light-curve product at MAST (no download)."""
    import lightkurve as lk
    sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if len(sr) == 0:
        return []
    exps = np.atleast_1d(getattr(sr.exptime, "value", sr.exptime)).astype(float)
    return [{"sector": _sector_of(sr.mission[i]), "author": str(sr.author[i]), "exptime": float(exps[i])}
            for i in range(len(sr))]


def audit_target(observed, products, sector_status=None, authors=AUTHOR_PRIORITY):
    """Summarise one target's coverage. All inputs are plain lists/tables (testable offline).

    observed: list of sector numbers; products: list of dicts (sector, author, exptime);
    sector_status: optional rows (sector, status) from a refit.
    """
    obs = set(int(s) for s in observed)
    by_sector = {}
    for p in products:
        by_sector.setdefault(int(p["sector"]), set()).add(str(p["author"]))
    usable = {s for s, a in by_sector.items() if a & set(authors)}
    other_only = {s for s, a in by_sector.items() if not (a & set(authors))}
    rec = {
        "n_observed": len(obs), "first_observed": min(obs) if obs else -1, "last_observed": max(obs) if obs else -1,
        "n_with_product": len(usable), "first_product": min(usable) if usable else -1,
        "last_product": max(usable) if usable else -1,
        "missing_sectors": ",".join(map(str, sorted(obs - usable - other_only))),
        "other_pipeline_only": ",".join(map(str, sorted(other_only))),
        "n_missing": len(obs - usable),
    }
    for a in authors:
        rec[f"n_{a.replace('-', '_')}"] = sum(1 for s, au in by_sector.items() if a in au)
    if sector_status is not None and len(sector_status):
        st = {int(r["sector"]): str(r["status"]) for r in sector_status}
        timed = sorted(s for s, v in st.items() if v == "ok")
        rec.update(n_timed=len(timed), first_timed=min(timed) if timed else -1, last_timed=max(timed) if timed else -1)
        for reason in ("no_eclipse", "unconstrained", "too_few_points", "bad_noise", "negative_scale"):
            rec[f"n_{reason}"] = sum(1 for v in st.values() if v == reason)
    return rec


def coverage_figure(audit, path, max_sector):
    """Per-target sector map: observed / product available / timed."""
    import matplotlib.pyplot as plt

    n = len(audit)
    fig, ax = plt.subplots(figsize=(11, 0.22 * n + 1.6), constrained_layout=True)
    for i, r in enumerate(audit):
        obs = [int(s) for s in str(r.get("observed", "")).split(",") if s]
        prod = [int(s) for s in str(r.get("product_sectors", "")).split(",") if s]
        timed = [int(s) for s in str(r.get("timed_sectors", "")).split(",") if s]
        ax.scatter(obs, [i] * len(obs), marker="s", s=22, facecolor="none", edgecolor="0.6", lw=0.8)
        ax.scatter(prod, [i] * len(prod), marker="s", s=22, color="#9ec5e8", lw=0)
        ax.scatter(timed, [i] * len(timed), marker="s", s=8, color="#1f5fbf", lw=0)
    ax.set_yticks(range(n)); ax.set_yticklabels([r["eb_id"] for r in audit], fontsize=7)
    ax.set_xlim(0, max_sector + 1); ax.set_xlabel("TESS sector"); ax.invert_yaxis()
    ax.scatter([], [], marker="s", facecolor="none", edgecolor="0.6", label="observed (tess-point)")
    ax.scatter([], [], marker="s", color="#9ec5e8", label="product at MAST")
    ax.scatter([], [], marker="s", color="#1f5fbf", s=8, label="timed in refit")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0 + 1.2 / (0.22 * n + 1.6)), ncol=3, fontsize=8, frameon=False)
    fig.savefig(path, dpi=120); plt.close(fig)
