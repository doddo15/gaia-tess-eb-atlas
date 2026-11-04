"""Fetch and standardise Gaia epoch photometry.

DR3 epoch photometry exists only for sources Gaia classified as variable,
so it is the *biased* baseline. DR4 lives in dr4_photometry.py.

Standard output: astropy Table with columns
    band, time_gaia, btjd, flux, flux_err, mag, rejected  (+ rejected_by_variability for DR3)

DataLink formats handled (see datalink.py for the access side):
  * long  -- one row per band x transit (band, time, flux, ...); column names
             are matched through aliases because DATAMODEL_STANDARD's exact
             names are not documented here;
  * wide  -- one row per source with per-transit arrays (g_transit_time,
             g_transit_flux, bp_obs_time, ...), the DR3 Gaia data model.
"""
from __future__ import annotations

import numpy as np
from astropy.table import Table, vstack

from .ephemeris import gaia_time_to_btjd

STANDARD_COLUMNS = ["band", "time_gaia", "btjd", "flux", "flux_err", "mag", "rejected"]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


_ALIASES = {
    "band": ("band", "passband", "filter", "photometric_band"),
    "time": ("time", "obs_time", "g_transit_time_bjd", "bjd_tcb"),
    "flux": ("flux", "flux_value"),
    "flux_error": ("flux_error", "flux_err", "flux_uncertainty", "error"),
    "mag": ("mag", "magnitude"),
    "rejected_by_photometry": ("rejected_by_photometry",),
    "rejected_by_variability": ("rejected_by_variability",),
}


_WIDE_SPEC = {
    "G": ("g_transit_time", "g_transit_flux", "g_transit_flux_error", "g_transit_mag", "variability_flag_g_reject"),
    "BP": ("bp_obs_time", "bp_flux", "bp_flux_error", "bp_mag", "variability_flag_bp_reject"),
    "RP": ("rp_obs_time", "rp_flux", "rp_flux_error", "rp_mag", "variability_flag_rp_reject"),
}


def _find(cols, key):
    low = {c.lower(): c for c in cols}
    for a in _ALIASES[key]:
        if a in low:
            return low[a]
    return None


def _bool(col, n):
    if col is None:
        return np.zeros(n, bool)
    a = np.ma.filled(np.ma.asarray(col), False)
    if a.dtype.kind in "US":
        return np.char.lower(a.astype(str)) == "true"
    return a.astype(bool)


def standardize_dr3(tab):
    """Convert ONE source's DR3 epoch photometry (long or wide format) to the standard schema.

    `rejected` comes from DR3's photometric processing: rejected_by_photometry (long and
    DATAMODEL_STANDARD formats) or the per-transit photometry_flag_*_reject /
    *_unavailable arrays (DATAMODEL_GAIA; G is rejected only when no AF CCD
    contributed), plus non-finite or non-positive fluxes.
    `rejected_by_variability` is kept as its own column: the variability
    pipeline's outlier rejection often removes exactly the in-eclipse points,
    so they stay in the analysis and are counted separately.
    """
    cols = tab.colnames
    cb, ct, cf = _find(cols, "band"), _find(cols, "time"), _find(cols, "flux")
    if cb and ct and cf:
        n = len(tab)
        ce, cm = _find(cols, "flux_error"), _find(cols, "mag")
        num = lambda c: np.ma.filled(np.ma.asarray(tab[c]).astype(float), np.nan) if c else np.full(n, np.nan)
        band = np.char.upper(np.char.strip(np.asarray(tab[cb]).astype(str)))
        band = np.where(np.isin(band, ["G", "BP", "RP"]), band,
                        np.char.replace(np.char.replace(band, "GAIA.", ""), "GAIA_", ""))
        out = Table({
            "band": band, "time_gaia": num(ct), "flux": num(cf), "flux_err": num(ce), "mag": num(cm),
            "rejected": _bool(tab[c] if (c := _find(cols, "rejected_by_photometry")) else None, n),
            "rejected_by_variability": _bool(tab[c] if (c := _find(cols, "rejected_by_variability")) else None, n),
        })
    elif "g_transit_time" in cols and tab["g_transit_time"].dtype != object and np.ndim(tab["g_transit_time"][0]) == 0:
        # DATAMODEL_STANDARD: one row per transit for ONE source, bands side by side as scalars.
        n = len(tab)
        num = lambda c: np.ma.filled(np.ma.asarray(tab[c]).astype(float), np.nan) if c in cols else np.full(n, np.nan)
        rbp = _bool(tab["rejected_by_photometry"] if "rejected_by_photometry" in cols else None, n)
        parts = []
        for band, (tc, fc, ec, mc, rc) in _WIDE_SPEC.items():
            if tc not in cols:
                continue
            parts.append(Table({
                "band": [band] * n, "time_gaia": num(tc), "flux": num(fc), "flux_err": num(ec), "mag": num(mc),
                "rejected": rbp.copy(),
                "rejected_by_variability": _bool(tab[rc] if rc in cols else None, n),
            }))
        out = vstack(parts)
    elif "g_transit_time" in cols:
        # DATAMODEL_GAIA: one row per source, per-transit arrays.
        parts = []
        for row in tab:
            arr = lambda c: np.ma.filled(np.ma.asarray(row[c]).astype(float), np.nan)
            barr = lambda c: np.ma.filled(np.ma.asarray(row[c]), False).astype(bool)
            nt = len(arr("g_transit_time"))
            flag = lambda c: barr(c) if c in cols else np.zeros(nt, bool)
            # G: rejected when no AF CCD contributed (all AF CCDs rejected or unavailable).
            af_bad = np.ones(nt, bool)
            for k in range(1, 10):
                af_bad &= flag(f"photometry_flag_af{k}_reject") | flag(f"photometry_flag_af{k}_unavailable")
            if not any(f"photometry_flag_af{k}_reject" in cols for k in range(1, 10)):
                af_bad = np.zeros(nt, bool)
            if "g_transit_n_obs" in cols:
                af_bad |= np.ma.filled(np.ma.asarray(row["g_transit_n_obs"]), 0).astype(int) == 0
            hard = {"G": af_bad,
                    "BP": flag("photometry_flag_bp_reject") | flag("photometry_flag_bp_unavailable"),
                    "RP": flag("photometry_flag_rp_reject") | flag("photometry_flag_rp_unavailable")}
            for band, (tc, fc, ec, mc, rc) in _WIDE_SPEC.items():
                if tc not in cols:
                    continue
                t = arr(tc); f = arr(fc)
                parts.append(Table({
                    "band": [band] * len(t), "time_gaia": t, "flux": f,
                    "flux_err": arr(ec) if ec in cols else np.full(len(t), np.nan),
                    "mag": arr(mc) if mc in cols else np.full(len(t), np.nan),
                    "rejected": hard[band],
                    "rejected_by_variability": flag(rc),
                }))
        out = vstack(parts) if parts else Table(names=STANDARD_COLUMNS[:2])
    else:
        raise ValueError("Unrecognised epoch-photometry columns: " + ", ".join(cols[:20]))

    fl = np.asarray(out["flux"], float)
    out["rejected"] = np.asarray(out["rejected"]) | ~np.isfinite(fl) | (fl <= 0)
    good = np.isfinite(np.asarray(out["time_gaia"], float))
    out = out[good]
    out["btjd"] = gaia_time_to_btjd(out["time_gaia"])
    return out[STANDARD_COLUMNS + ["rejected_by_variability"]]


def parse_dr3_result(res, unparsed_dir=None):
    """{source_id: standard table} from a DataLink result (any number of files/tables/sources).

    Tables that cannot be parsed are written to `unparsed_dir` (if given) before the
    error is raised, so the real format can be inspected.
    """
    from .datalink import iter_tables, split_by_source

    out = {}
    for key, tab in iter_tables(res):
        try:
            for sid, sub in split_by_source(tab, key).items():
                out[sid] = standardize_dr3(sub)
        except Exception as exc:
            if unparsed_dir:
                from pathlib import Path
                d = Path(unparsed_dir); d.mkdir(parents=True, exist_ok=True)
                safe = Table({c: [str(v) for v in tab[c]] for c in tab.colnames})
                safe.write(d / (Path(str(key)).stem + ".ecsv"), overwrite=True)
            raise RuntimeError(f"Could not parse DataLink table {key!r} ({exc}). Columns: {tab.colnames}. "
                               "Run scripts/inspect_datalink.py and send the output.") from exc
    return out


def fetch_dr3_epoch_photometry(source_ids, chunk=500, cache_dir=None, retries=3, pause=5.0,
                               structures=None):
    """Return {source_id: standard Table} for DR3 sources that have epoch photometry.

    Sources without DR3 epoch photometry (not classified variable) are simply
    absent. With `cache_dir`, each source is saved as <cache_dir>/<source_id>.ecsv
    and empty answers are remembered in _empty.txt, so reruns skip finished work.
    `structures` defaults to datalink.STRUCTURES (DATAMODEL_GAIA, then DATAMODEL_STANDARD).
    """
    from pathlib import Path
    from . import datalink

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
        res, _ = datalink.load(block, "EPOCH_PHOTOMETRY", "Gaia DR3",
                               structures=structures or datalink.STRUCTURES, retries=retries, pause=pause)
        got = parse_dr3_result(res, unparsed_dir=(cache / "_unparsed") if cache else None)
        for sid, std in got.items():
            if cache:
                std.write(cache / f"{sid}.ecsv", overwrite=True)
            else:
                out[sid] = std
        if cache and set(block) - set(got):
            empty |= set(block) - set(got)
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
