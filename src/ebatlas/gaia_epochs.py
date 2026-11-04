"""Fetch and standardise Gaia epoch photometry.

DR3 epoch photometry exists only for sources Gaia classified as variable,
so it is the *biased* baseline. DR4 publishes epoch photometry for all
sources; the DR4 adapter is a stub to fill in once the DR4 data model and
DataLink products are documented.

Standard output: astropy Table with columns
    band, time_gaia, btjd, flux, flux_err, mag, rejected
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table, vstack

from .ephemeris import gaia_time_to_btjd

STANDARD_COLUMNS = ["band", "time_gaia", "btjd", "flux", "flux_err", "mag", "rejected"]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _as_table(obj):
    return obj.to_table() if hasattr(obj, "to_table") else Table(obj)


def standardize_dr3(tab):
    """Convert a DR3 epoch-photometry table (long or wide format) to the standard schema.

    The DataLink VOTable is (as of DR3) in long format: one row per
    band x transit with `band`, `time`, `flux`, `flux_error`, `mag`,
    `rejected_by_photometry`, `rejected_by_variability`. The bulk CSV
    files are wide (g_transit_time, bp_obs_time, ...). Verify against
    the live archive; this function tries both.

    Only `rejected_by_photometry` (or a non-finite flux) sets `rejected`.
    `rejected_by_variability` is kept as its own column: the variability
    pipeline's outlier rejection often removes exactly the in-eclipse points,
    so they stay in the analysis and are counted separately.
    """
    cols = set(tab.colnames)
    if {"band", "time", "flux"} <= cols:
        rej = (np.ma.filled(np.ma.asarray(tab["rejected_by_photometry"]), False).astype(bool)
               if "rejected_by_photometry" in cols else np.zeros(len(tab), bool))
        var = (np.ma.filled(np.ma.asarray(tab["rejected_by_variability"]), False).astype(bool)
               if "rejected_by_variability" in cols else np.zeros(len(tab), bool))
        t = np.asarray(tab["time"], float)
        out = Table({
            "band": np.asarray(tab["band"]).astype(str),
            "time_gaia": t,
            "flux": np.asarray(tab["flux"], float),
            "flux_err": np.asarray(tab["flux_error"], float) if "flux_error" in cols else np.full(len(t), np.nan),
            "mag": np.asarray(tab["mag"], float) if "mag" in cols else np.full(len(t), np.nan),
            "rejected": rej,
            "rejected_by_variability": var,
        })
    elif "g_transit_time" in cols:
        parts = []
        spec = {"G": ("g_transit_time", "g_transit_flux", "g_transit_flux_error", "g_transit_mag", "variability_flag_g_reject"),
                "BP": ("bp_obs_time", "bp_flux", "bp_flux_error", "bp_mag", "variability_flag_bp_reject"),
                "RP": ("rp_obs_time", "rp_flux", "rp_flux_error", "rp_mag", "variability_flag_rp_reject")}
        row = tab[0]
        for band, (tc, fc, ec, mc, rc) in spec.items():
            t = np.asarray(row[tc], float)
            f = np.asarray(row[fc], float)
            parts.append(Table({
                "band": [band] * len(t), "time_gaia": t,
                "flux": f, "flux_err": np.asarray(row[ec], float),
                "mag": np.asarray(row[mc], float),
                "rejected": ~np.isfinite(f),
                "rejected_by_variability": np.asarray(row[rc]).astype(bool) if rc in cols else np.zeros(len(t), bool),
            }))
        out = vstack(parts)
    else:
        raise ValueError(f"Unrecognised epoch-photometry columns: {sorted(cols)[:12]} ...")

    out["rejected"] = np.asarray(out["rejected"]) | ~np.isfinite(np.asarray(out["flux"], float))
    good = np.isfinite(out["time_gaia"])
    out = out[good]
    out["btjd"] = gaia_time_to_btjd(out["time_gaia"])
    return out[STANDARD_COLUMNS + ["rejected_by_variability"]]


def fetch_dr3_epoch_photometry(source_ids, chunk=500, cache_dir=None, retries=3, pause=5.0):
    """Return {source_id: standard Table} for DR3 sources that have epoch photometry.

    Sources without DR3 epoch photometry (i.e. not classified variable) are
    simply absent from the result. With `cache_dir`, each source is saved as
    <cache_dir>/<source_id>.ecsv and empty answers are remembered in
    _empty.txt, so reruns and interrupted runs skip finished work.
    `valid_data=False` (astroquery's default) keeps rejected rows, which we need.
    """
    import time
    import warnings
    from pathlib import Path
    from astroquery.gaia import Gaia

    ids = [int(s) for s in source_ids]
    cache = Path(cache_dir) if cache_dir else None
    empty = set()
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        ef = cache / "_empty.txt"
        empty = set(int(x) for x in ef.read_text().split()) if ef.exists() else set()
        todo = [s for s in ids if s not in empty and not (cache / f"{s}.ecsv").exists()]
    else:
        todo = ids

    out = {}
    for block in _chunks(todo, chunk):
        for attempt in range(retries):
            try:
                res = Gaia.load_data(ids=block, data_release="Gaia DR3", retrieval_type="EPOCH_PHOTOMETRY",
                                     data_structure="INDIVIDUAL", format="votable", valid_data=False)
                break
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                warnings.warn(f"DataLink request failed ({exc}); retrying")
                time.sleep(pause * (attempt + 1))
        got = set()
        for _key, tables in res.items():
            tab = _as_table(tables[0])
            sid = int(tab["source_id"][0])
            std = standardize_dr3(tab)
            got.add(sid)
            if cache:
                std.write(cache / f"{sid}.ecsv", overwrite=True)
            else:
                out[sid] = std
        if cache and set(block) - got:
            empty |= set(block) - got
            (cache / "_empty.txt").write_text("\n".join(map(str, sorted(empty))))
    if cache:
        for s in ids:
            p = cache / f"{s}.ecsv"
            if p.exists():
                out[s] = Table.read(p)
    return out


def fetch_dr4_epoch_photometry(source_ids, **kwargs):
    """See ebatlas.dr4_photometry.fetch_dr4_epoch_photometry."""
    from .dr4_photometry import fetch_dr4_epoch_photometry as _f
    return _f(source_ids, **kwargs)
