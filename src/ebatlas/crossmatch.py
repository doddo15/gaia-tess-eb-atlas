"""TIC -> Gaia cross-matching and Gaia-side context queries.

Two stages:
  1. Pre-release (runs now): TIC -> DR3 from the published Hardegree-Ullman et al. (2026)
     table when available, else TIC v8.2 (Gaia DR2 IDs) -> gaiadr3.dr2_neighbourhood -> DR3,
     plus DR3 context. `prepare_dr3_crossmatch` writes this to a file.
  2. Release day: DR3 -> DR4 via the DR4 `dr3_neighbourhood` table, joined with
     `all_source_flags.has_epoch_photometry`. `dr3_to_dr4` does it in one
     uploaded-table query.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table, vstack

from .gaia_epochs import _chunks


def tic_to_gaia_dr2(tic_ids, chunk=500):
    """Return Table(tic_id, gaia_dr2_id) from the TIC at MAST."""
    from astroquery.mast import Catalogs

    rows = []
    ids = [str(int(t)) for t in tic_ids]
    for block in _chunks(ids, chunk):
        tab = Catalogs.query_criteria(catalog="TIC", ID=block)
        for r in tab:
            g = r["GAIA"]
            if g is not None and not np.ma.is_masked(g) and str(g).strip():
                rows.append((int(r["ID"]), int(g)))
    return Table(rows=rows, names=("tic_id", "gaia_dr2_id"), dtype=(np.int64, np.int64))


def _adql(query):
    from astroquery.gaia import Gaia
    return Gaia.launch_job_async(query).get_results()


def dr2_to_dr3(dr2_ids, chunk=2000):
    """All DR3 counterparts per DR2 ID, with a `best` flag (closest match).

    DR2 sources can split into several DR3 sources; keep all and flag
    ambiguous cases rather than silently choosing one.
    """
    parts = []
    ids = [int(i) for i in dr2_ids]
    for block in _chunks(ids, chunk):
        q = f"""
        SELECT dr2_source_id, dr3_source_id, angular_distance, magnitude_difference
        FROM gaiadr3.dr2_neighbourhood
        WHERE dr2_source_id IN ({",".join(map(str, block))})
        """
        parts.append(_adql(q))
    t = vstack(parts) if parts else Table()
    if len(t):
        t.sort(["dr2_source_id", "angular_distance"])
        first = np.r_[True, t["dr2_source_id"][1:] != t["dr2_source_id"][:-1]]
        t["best"] = first
    return t


def dr3_context(source_ids, chunk=2000):
    """RUWE, NSS flag, and whether Gaia DR3 already flagged the source as an EB."""
    parts = []
    ids = [int(i) for i in source_ids]
    for block in _chunks(ids, chunk):
        q = f"""
        SELECT s.source_id, s.ra, s.dec, s.phot_g_mean_mag, s.bp_rp, s.ruwe,
               s.non_single_star, s.phot_variable_flag, s.has_epoch_photometry,
               c.best_class_name, e.frequency AS dr3_eb_frequency
        FROM gaiadr3.gaia_source AS s
        LEFT JOIN gaiadr3.vari_classifier_result AS c ON s.source_id = c.source_id
        LEFT JOIN gaiadr3.vari_eclipsing_binary AS e ON s.source_id = e.source_id
        WHERE s.source_id IN ({",".join(map(str, block))})
        """
        parts.append(_adql(q))
    return vstack(parts) if parts else Table()


def gaia_neighbours(ra, dec, radius_arcsec=42.0, gmax=None):
    """All DR3 sources within radius (default ~2 TESS pixels).

    Any of these could be the true host of a TESS eclipse signal; with DR4
    epoch photometry for all of them, Gaia can say which one actually dips.
    """
    cut = f"AND phot_g_mean_mag < {gmax}" if gmax is not None else ""
    q = f"""
    SELECT source_id, ra, dec, phot_g_mean_mag, bp_rp, ruwe,
           DISTANCE(POINT({ra}, {dec}), POINT(ra, dec)) * 3600 AS sep_arcsec
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_arcsec / 3600.0}))
    {cut}
    ORDER BY phot_g_mean_mag
    """
    return _adql(q)


def _launch_with_upload(query, ids, colname, table_name="ids"):
    """Run ADQL against an uploaded one-column table (avoids huge IN (...) lists)."""
    import tempfile
    from astropy.io.votable import from_table, writeto
    from astroquery.gaia import Gaia

    up = Table({colname: np.asarray(ids, dtype=np.int64)})
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fh:
        writeto(from_table(up), fh.name)
        job = Gaia.launch_job_async(query, upload_resource=fh.name, upload_table_name=table_name)
    return job.get_results()


def positional_dr3_match(ra, dec, radius_arcsec=2.0):
    """Fallback for TIC entries without a Gaia DR2 ID: nearest DR3 source within radius.

    TIC positions are J2000; for high proper-motion stars widen the radius.
    """
    q = f"""
    SELECT TOP 1 source_id, phot_g_mean_mag,
           DISTANCE(POINT({ra}, {dec}), POINT(ra, dec)) * 3600 AS sep_arcsec
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_arcsec / 3600.0}))
    ORDER BY sep_arcsec
    """
    t = _adql(q)
    return (int(t["source_id"][0]), float(t["sep_arcsec"][0])) if len(t) else (None, np.nan)


# --- Stage 1 option: the published TIC / Gaia DR3 cross-match ------------------------
# Hardegree-Ullman et al. 2026, RNAAS 10, 77 (ExoFOP, doi:10.26134/exofop6).
# Built from TIC v8.2 + gaiadr3.dr2_neighbourhood with cuts: angular distance < 100 mas
# and |delta G| <= 0.2 mag. Column names in the download are not known here, so they are
# auto-detected; pass `columns=` to override.
HU26_CUTS = {"max_dist_mas": 100.0, "max_abs_dmag": 0.2}
_HU26_PATTERNS = {
    "tic_id": ("tic",),
    "gaia_dr2_id": ("dr2",),
    "dr3_source_id": ("dr3",),
    "angular_distance_mas": ("dist", "angular", "sep"),
    "g_mag_difference": ("mag", "dg", "delta"),
}


def _detect_columns(names):
    low = {n: n.lower().replace(" ", "_") for n in names}
    out = {}
    for std, keys in _HU26_PATTERNS.items():
        for n, l in low.items():
            if n in out.values():
                continue
            if any(k in l for k in keys):
                out[std] = n
                break
    missing = {"tic_id", "dr3_source_id"} - set(out)
    if missing:
        raise ValueError(f"Could not find columns {missing} among {list(names)}; pass columns=")
    return out


def load_tic_dr3_table(path, tic_ids, columns=None):
    """Rows of the TIC/DR3 cross-match table for the given TIC IDs, without loading 33 GB.

    Uses Polars' lazy scan (CSV, gzipped CSV, Parquet; globs allowed), so only
    matching rows are materialised. Returns Table(tic_id, gaia_dr2_id, dr3_source_id,
    angular_distance_mas, g_mag_difference).
    """
    import polars as pl

    path = str(path)
    lf = pl.scan_parquet(path) if path.endswith(".parquet") else pl.scan_csv(path, infer_schema_length=10000)
    names = lf.collect_schema().names()
    cols = columns or _detect_columns(names)
    tics = [int(t) for t in np.unique(np.asarray(tic_ids))]
    df = (lf.select([pl.col(c).alias(k) for k, c in cols.items()])
            .filter(pl.col("tic_id").cast(pl.Int64).is_in(tics))
            .collect())
    t = Table({k: df[k].to_numpy() for k in df.columns})
    for k in ("tic_id", "gaia_dr2_id", "dr3_source_id"):
        if k in t.colnames:
            t[k] = np.asarray(t[k]).astype(np.int64)
    return t


def _best_per(key, t, dist_col):
    """{key: (best_row, n_rows)} choosing the smallest distance."""
    out = {}
    for r in t:
        k = int(r[key]); d = float(r[dist_col]) if dist_col in t.colnames else 0.0
        if k not in out:
            out[k] = (r, 1, d)
        else:
            best, n, bd = out[k]
            out[k] = (r, n + 1, d) if d < bd else (best, n + 1, bd)
    return {k: (v[0], v[1]) for k, v in out.items()}


def prepare_dr3_crossmatch(cat, positional_fallback=True, tic_dr3=None):
    """Stage 1: one row per catalog EB with its DR3 source_id and DR3 context.

    Match order, recorded in `match_method`:
      'hu26'        -- the published TIC/DR3 table (pass a path as `tic_dr3`),
      'tic_dr2'     -- TIC -> DR2 (MAST) -> gaiadr3.dr2_neighbourhood, closest match, no cuts,
      'positional'  -- nearest DR3 source within 2 arcsec of the TIC position,
      'none'.
    `passes_hu26_cuts` applies the paper's quality cuts to every match that has
    distance and magnitude difference, so query matches can be compared fairly.
    `shared_dr3_id` marks DR3 sources claimed by more than one TIC (TIC v7->v8 artifact).
    """
    tics = np.unique(np.asarray(cat["tic_id"]))
    found = {}   # tic -> dict
    if tic_dr3 is not None:
        hu = tic_dr3 if isinstance(tic_dr3, Table) else load_tic_dr3_table(tic_dr3, tics)
        for tic, (r, n) in _best_per("tic_id", hu, "angular_distance_mas").items():
            found[tic] = {"gaia_dr2_id": int(r["gaia_dr2_id"]) if "gaia_dr2_id" in hu.colnames else -1,
                          "dr3_source_id": int(r["dr3_source_id"]), "n_dr3_candidates": n,
                          "angular_distance_mas": float(r["angular_distance_mas"]) if "angular_distance_mas" in hu.colnames else np.nan,
                          "g_mag_difference": float(r["g_mag_difference"]) if "g_mag_difference" in hu.colnames else np.nan,
                          "match_method": "hu26"}

    todo = [int(t) for t in tics if int(t) not in found]
    if todo:
        tic2 = tic_to_gaia_dr2(todo)
        t2 = {int(r["tic_id"]): int(r["gaia_dr2_id"]) for r in tic2}
        nb = dr2_to_dr3(list(t2.values())) if t2 else Table()
        per = _best_per("dr2_source_id", nb, "angular_distance") if len(nb) else {}
        for tic, d2 in t2.items():
            if d2 in per:
                r, n = per[d2]
                found[tic] = {"gaia_dr2_id": d2, "dr3_source_id": int(r["dr3_source_id"]), "n_dr3_candidates": n,
                              "angular_distance_mas": float(r["angular_distance"]),
                              "g_mag_difference": float(r["magnitude_difference"]), "match_method": "tic_dr2"}

    rows = []
    for r in cat:
        tic = int(r["tic_id"])
        rec = {"eb_id": r["eb_id"], "tic_id": tic, "gaia_dr2_id": -1, "dr3_source_id": -1, "n_dr3_candidates": 0,
               "angular_distance_mas": np.nan, "g_mag_difference": np.nan, "match_method": "none"}
        rec.update(found.get(tic, {}))
        if rec["dr3_source_id"] < 0 and positional_fallback and np.isfinite(r["ra"]):
            d3, sep = positional_dr3_match(float(r["ra"]), float(r["dec"]))
            if d3 is not None:
                rec.update(dr3_source_id=d3, n_dr3_candidates=1, angular_distance_mas=sep * 1000.0,
                           match_method="positional")
        rows.append(rec)
    xm = Table(rows=rows)
    dist, dmag = np.asarray(xm["angular_distance_mas"], float), np.asarray(xm["g_mag_difference"], float)
    xm["passes_hu26_cuts"] = (dist < HU26_CUTS["max_dist_mas"]) & (np.abs(dmag) <= HU26_CUTS["max_abs_dmag"])
    d3 = np.asarray(xm["dr3_source_id"])
    tic_per_d3 = {}
    for t, d in zip(np.asarray(xm["tic_id"]), d3):
        if d > 0:
            tic_per_d3.setdefault(int(d), set()).add(int(t))
    xm["shared_dr3_id"] = [d > 0 and len(tic_per_d3[int(d)]) > 1 for d in d3]

    ids = sorted({int(x) for x in d3 if x > 0})
    ctx = dr3_context(ids)
    by = {int(c["source_id"]): c for c in ctx}
    for col in ("ruwe", "non_single_star", "has_epoch_photometry", "best_class_name", "dr3_eb_frequency"):
        vals = []
        for d in d3:
            c = by.get(int(d))
            v = c[col] if c is not None and col in c.colnames else None
            missing = v is None or np.ma.is_masked(v)
            default = {"best_class_name": "", "has_epoch_photometry": False}.get(col, np.nan)
            vals.append(default if missing else v)
        xm[f"dr3_{col}" if not col.startswith("dr3_") else col] = vals
    return xm


# --- Stage 2: release day -------------------------------------------------------------
# TODO(relay): confirm these against the published DR4 schema. Only
# `all_source_flags.has_epoch_photometry` is confirmed by the draft data model.
DR4_XMATCH = {
    "schema": "gaiadr4",
    "table": "dr3_neighbourhood",
    "dr3_col": "dr3_source_id",
    "dr4_col": "dr4_source_id",
    "dist_col": "angular_distance",
    "dmag_col": "magnitude_difference",
}


def dr4_crossmatch_query(cfg=None):
    c = {**DR4_XMATCH, **(cfg or {})}
    s = c["schema"]
    return f"""
    SELECT u.dr3_source_id,
           n.{c['dr4_col']} AS dr4_source_id,
           n.{c['dist_col']} AS angular_distance,
           n.{c['dmag_col']} AS magnitude_difference,
           f.has_epoch_photometry, f.has_epoch_astrometry,
           g.source_id AS gaia_source_id,
           e.source_id AS vari_eb_id
    FROM tap_upload.ids AS u
    JOIN {s}.{c['table']} AS n ON n.{c['dr3_col']} = u.dr3_source_id
    LEFT JOIN {s}.all_source_flags AS f ON f.source_id = n.{c['dr4_col']}
    LEFT JOIN {s}.gaia_source AS g ON g.source_id = n.{c['dr4_col']}
    LEFT JOIN {s}.vari_eclipsing_binary AS e ON e.source_id = n.{c['dr4_col']}
    """


def resolve_dr4_matches(t):
    """Keep the closest DR4 counterpart per DR3 ID; add n_dr4_candidates and booleans."""
    if len(t) == 0:
        return t
    t = t.copy()
    t.sort(["dr3_source_id", "angular_distance"])
    d3 = np.asarray(t["dr3_source_id"])
    first = np.r_[True, d3[1:] != d3[:-1]]
    counts = dict(zip(*np.unique(d3, return_counts=True)))
    out = t[first]
    out["n_dr4_candidates"] = [int(counts[int(x)]) for x in out["dr3_source_id"]]
    out["in_gaia_source"] = ~np.ma.getmaskarray(np.ma.asarray(out["gaia_source_id"]))
    out["in_dr4_vari_eb"] = ~np.ma.getmaskarray(np.ma.asarray(out["vari_eb_id"]))
    hp = np.ma.filled(np.ma.asarray(out["has_epoch_photometry"]), False)
    out["has_epoch_photometry"] = np.array([str(x).lower() in ("true", "1", "t") for x in hp])
    return out


def dr3_to_dr4(dr3_ids, cfg=None):
    """Stage 2: map DR3 IDs to DR4 and flag epoch-photometry availability (one query)."""
    raw = _launch_with_upload(dr4_crossmatch_query(cfg), dr3_ids, "dr3_source_id")
    return resolve_dr4_matches(raw)
