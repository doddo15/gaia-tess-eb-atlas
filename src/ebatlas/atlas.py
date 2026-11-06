"""Atlas builders.

DR3 (now):     ebatlas-build data/TEBC.csv --catalog tebc --release dr3 --out atlas_dr3/
DR4 (release): ebatlas-build data/TEBC.csv --catalog tebc --release dr4 \
                   --xmatch data/tebc_dr3_xmatch.ecsv --out atlas_dr4/

`build_pages` does the per-target work and is shared by both flows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

from .catalogs import load_catalog, row_to_ephemeris
from .eclipses import measure_eclipses
from .planning import phase_sigma_table
from .plotting import atlas_page, summarize


def _rows_to_table(rows):
    """Rows may have different keys; fill gaps so every column exists everywhere."""
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    def fill(k):
        sample = next(r[k] for r in rows if k in r)
        if isinstance(sample, str):
            return ""
        if isinstance(sample, (bool, np.bool_)):
            return False
        return np.nan
    return Table({k: [r.get(k, fill(k)) for r in rows] for k in keys})


def _needs_geometry(row):
    return not np.isfinite(row["primary_width"]) or not np.isfinite(row["secondary_phase"])


def build_pages(cat, gaia_ids, epochs, out_dir, context=None, lc_getter=None, release="DR4"):
    """Per-target atlas pages and summary.

    gaia_ids:  {eb_id: source_id} in the release being plotted
    epochs:    {source_id: standard epoch table}
    context:   {eb_id: dict of extra summary columns} (cross-match info, flags)
    lc_getter: callable(tic_id) -> (btjd, flux, flux_err) or None; None disables TESS.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    context = context or {}
    sig = {r["eb_id"]: r for r in phase_sigma_table(cat)}
    lc_cache, rows = {}, []
    for row in cat:
        eid = row["eb_id"]; tic = int(row["tic_id"]); sid = gaia_ids.get(eid)
        s = sig[eid]
        rec = {"eb_id": eid, "tic_id": tic, f"gaia_{release.lower()}_id": sid if sid else -1,
               "has_epoch_phot": sid in epochs,
               "sigma_phase_start": float(s["sigma_phase_start"]),
               "sigma_phase_mid": float(s["sigma_phase_mid"])}
        rec.update(context.get(eid, {}))

        lc = None
        if lc_getter is not None and (sid in epochs or _needs_geometry(row)):
            if tic not in lc_cache:
                lc_cache[tic] = lc_getter(tic)
            lc = lc_cache[tic]

        overrides = {}
        if lc is not None and _needs_geometry(row):
            geom = measure_eclipses(row_to_ephemeris(row), lc[0], lc[1])
            rec.update({f"meas_{k}": v for k, v in geom.items()
                        if k in ("primary_width", "primary_depth", "secondary_phase",
                                 "secondary_width", "secondary_depth", "contact_like")})
            overrides = {k: geom[k] for k in ("primary_width", "secondary_phase", "secondary_width")
                         if np.isfinite(geom[k])}

        if sid in epochs:
            eph = row_to_ephemeris(row, **overrides)
            rec.update(summarize(eph, epochs[sid]))
            atlas_page(eph, epochs[sid], tess=lc, title=f"{eid} / Gaia {release} {sid}",
                       out=out_dir / f"{eid}.png")
        rows.append(rec)

    summary = _rows_to_table(rows) if rows else Table()
    summary.write(out_dir / "summary.csv", overwrite=True)
    return summary


def build_atlas_dr3(cat, out_dir, with_tess=True):
    """DR3 flow (works now): cross-match, DR3 epoch photometry, pages."""
    from . import crossmatch, gaia_epochs, tess
    cat = cat[cat["has_ephemeris"]] if "has_ephemeris" in cat.colnames else cat
    xm = crossmatch.prepare_dr3_crossmatch(cat)
    ids = {r["eb_id"]: int(r["dr3_source_id"]) for r in xm if r["dr3_source_id"] > 0}
    epochs = gaia_epochs.fetch_dr3_epoch_photometry(sorted(set(ids.values())))
    ctx = {r["eb_id"]: {k: r[k] for k in xm.colnames if k not in ("eb_id", "tic_id")} for r in xm}
    return build_pages(cat, ids, epochs, out_dir, ctx, tess.fetch_tess_lc if with_tess else None, "DR3")


def build_atlas_dr4(cat, xmatch_dr3, out_dir, with_tess=True, cache_dir="cache/dr4_epoch_photometry"):
    """DR4 flow (release day): stage-1 file -> DR4 IDs -> epoch photometry -> pages."""
    from . import crossmatch, tess
    from .dr4_photometry import fetch_dr4_epoch_photometry
    cat = cat[cat["has_ephemeris"]] if "has_ephemeris" in cat.colnames else cat
    d3 = sorted({int(x) for x in xmatch_dr3["dr3_source_id"] if x > 0})
    m4 = crossmatch.dr3_to_dr4(d3)
    by3 = {int(r["dr3_source_id"]): r for r in m4}
    ids, ctx = {}, {}
    for r in xmatch_dr3:
        m = by3.get(int(r["dr3_source_id"]))
        c = {k: r[k] for k in xmatch_dr3.colnames if k not in ("eb_id", "tic_id")}
        if m is not None:
            c.update(dr4_source_id=int(m["dr4_source_id"]), n_dr4_candidates=int(m["n_dr4_candidates"]),
                     in_gaia_source=bool(m["in_gaia_source"]), in_dr4_vari_eb=bool(m["in_dr4_vari_eb"]),
                     dr4_has_epoch_photometry=bool(m["has_epoch_photometry"]))
            if m["has_epoch_photometry"]:
                ids[r["eb_id"]] = int(m["dr4_source_id"])
        ctx[r["eb_id"]] = c
    epochs = fetch_dr4_epoch_photometry(sorted(set(ids.values())), cache_dir=cache_dir)
    return build_pages(cat, ids, epochs, out_dir, ctx, tess.fetch_tess_lc if with_tess else None, "DR4")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalog"); p.add_argument("--catalog", dest="name", default="tebc")
    p.add_argument("--out", default="atlas"); p.add_argument("--release", choices=["dr3", "dr4"], default="dr3")
    p.add_argument("--xmatch", help="stage-1 cross-match file (required for --release dr4)")
    p.add_argument("--limit", type=int, default=None); p.add_argument("--no-tess", action="store_true")
    p.add_argument("--refit", default=None, help="refit ephemerides from scripts/04_refit_tebc.py (used where ok)")
    a = p.parse_args(argv)
    cat = load_catalog(a.catalog, a.name)
    if a.refit:
        from .catalogs import apply_refit
        cat = apply_refit(cat, a.refit)
        print(f"ephemerides: {np.sum(cat['ephem_source'] == 'refit')} refit, {np.sum(cat['ephem_source'] == 'catalog')} catalog")
    if a.limit:
        cat = cat[: a.limit]
    if a.release == "dr3":
        s = build_atlas_dr3(cat, a.out, with_tess=not a.no_tess)
    else:
        if not a.xmatch:
            p.error("--release dr4 needs --xmatch (run scripts/01_prepare_dr3_xmatch.py first)")
        xm = Table.read(a.xmatch)
        xm = xm[np.isin(xm["eb_id"], cat["eb_id"])]
        s = build_atlas_dr4(cat, xm, a.out, with_tess=not a.no_tess)
    print(f"Wrote {len(s)} rows to {a.out}/summary.csv")


if __name__ == "__main__":
    main()
