"""Gaia DR4 epoch photometry (DataLink retrieval type EPOCH_PHOTOMETRY).

Written against the DR4 draft data model, section 11.1, which documents the
DATAMODEL_GAIA serialisation (requested first; DATAMODEL_STANDARD is the fallback,
see datalink.py). Facts from the data model:

* One row per (source_id, transit_id), wide format: g_*, bp_*, rp_* columns.
* g_obs_time, bp_obs_time, rp_obs_time are **already barycentric**: BJD(TCB)
  - 2455197.5 days (unlike epoch_astrometry.obs_time_tcb, which is at Gaia).
  G time is the FoV-transit average over ~9 AF CCDs spanning ~44 s.
* g_n_obs in [0, 9] (SM not used); 0 means every AF flux was discarded.
* BP/RP flux and errors are NaN when rejected or unavailable.
* phot_ccd_proc_flags: short[12] (SM, AF1..AF9, BP, RP). Bit 7 = rejected
  in the *mean* (source) accumulation only; the transit-level flux still
  uses all calibrated CCD data. So eclipses are NOT removed from epoch
  photometry by outlier rejection, but bit 7 tells us Gaia's pipeline
  regarded the in-eclipse CCDs as outliers. We count those.
* The variability pipeline's own epoch flags are a separate product
  (retrieval type EPOCH_FLAGS_VARI).

The output follows ebatlas.gaia_epochs.STANDARD_COLUMNS (long format, one
row per band per transit) plus: transit_id, fov, n_ccd, n_ccd_outlier, suspect.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from .ephemeris import gaia_time_to_btjd
from .dr4_astrometry import decode_transit_id, _stack

# TODO(relay): confirm the data_release string on release day.
DR4_DATALINK = {"data_release": "Gaia DR4", "retrieval_type": "EPOCH_PHOTOMETRY",
                "structures": ("DATAMODEL_GAIA", "DATAMODEL_STANDARD")}

PHOT_BITS = {
    "saturation_corrected": 0, "predicted_position_failed": 1, "ac_centring_clamped": 2,
    "no_calibration": 3, "calibration_replaced": 4, "calibration_failure": 5,
    "prefiltered_from_mean": 6, "outlier_in_mean": 7, "satellite_event": 8,
    "xp_charge_release": 9, "xp_straylight": 10, "xp_knn_background": 11,
    "xp_crowding_info": 12, "xp_truncation_unrecoverable": 13, "xp_truncation_treated": 14,
}
IDX_AF = slice(1, 10)
IDX_BP, IDX_RP = 10, 11
ACQ_BP_TRUNC, ACQ_RP_TRUNC, ACQ_NO_BP, ACQ_NO_RP = 0x0002, 0x0001, 0x0008, 0x0004

EXTRA_COLUMNS = ["transit_id", "fov", "n_ccd", "n_ccd_outlier", "suspect"]


def _bit(a, name):
    return (np.asarray(a, dtype=np.int64) >> PHOT_BITS[name]) & 1


def _col(tab, name, fill=np.nan):
    if name not in tab.colnames:
        return np.full(len(tab), fill)
    return np.ma.filled(np.ma.asarray(tab[name]).astype(float), fill)


def standardize_dr4(tab):
    """DR4 DATAMODEL_GAIA epoch photometry -> long standard table."""
    from .gaia_epochs import standardize_dr3, STANDARD_COLUMNS

    cols = set(tab.colnames)
    if {"band", "time", "flux"} <= cols:          # IVOA-style long format: reuse DR3 path
        return standardize_dr3(tab)
    if "g_obs_time" not in cols:
        raise ValueError(f"Not DR4 epoch photometry: {sorted(cols)[:10]} ...")

    n = len(tab)
    tid = np.asarray(tab["transit_id"], dtype=np.int64)
    fov = decode_transit_id(tid)["fov"]
    pf = (_stack(tab["phot_ccd_proc_flags"], dtype=np.int64, width=12)
          if "phot_ccd_proc_flags" in cols else np.zeros((n, 12), np.int64))
    acq = np.ma.filled(np.ma.asarray(tab["transit_acq_flags"]), 0).astype(np.int64) if "transit_acq_flags" in cols else np.zeros(n, np.int64)
    xpi = (_stack(tab["xp_ccd_info_flags"], dtype=np.int64, width=2)
           if "xp_ccd_info_flags" in cols else np.zeros((n, 2), np.int64))
    n_g = np.ma.filled(np.ma.asarray(tab["g_n_obs"]), 0).astype(int) if "g_n_obs" in cols else np.full(n, 9)

    af = pf[:, IDX_AF]
    g_outlier = _bit(af, "outlier_in_mean").sum(axis=1)
    g_suspect = (_bit(af, "satellite_event") | _bit(af, "no_calibration") | _bit(af, "calibration_failure")).any(axis=1)

    parts = []
    for band, idx, trunc, missing in (("G", None, 0, 0), ("BP", IDX_BP, ACQ_BP_TRUNC, ACQ_NO_BP),
                                      ("RP", IDX_RP, ACQ_RP_TRUNC, ACQ_NO_RP)):
        p = band.lower()
        t = _col(tab, f"{p}_obs_time"); f = _col(tab, f"{p}_flux")
        e = _col(tab, f"{p}_flux_error"); m = _col(tab, f"{p}_mag")
        if band == "G":
            rejected = (n_g == 0) | ~np.isfinite(f)
            n_ccd, n_out, suspect = n_g, g_outlier, g_suspect
        else:
            flags = pf[:, idx]
            unrec = _bit(flags, "xp_truncation_unrecoverable").astype(bool)
            rejected = ~np.isfinite(f) | unrec | ((acq & missing) > 0)
            n_ccd = np.isfinite(f).astype(int)
            n_out = _bit(flags, "outlier_in_mean")
            suspect = (((acq & trunc) > 0) | ((xpi[:, idx - IDX_BP] & 0b11) > 0)
                       | _bit(flags, "satellite_event").astype(bool))
        parts.append(Table({
            "band": [band] * n, "time_gaia": t, "flux": f, "flux_err": e, "mag": m,
            "rejected": rejected, "transit_id": tid, "fov": fov, "n_ccd": n_ccd,
            "n_ccd_outlier": n_out, "suspect": suspect,
        }))
    out = vstack(parts)
    out = out[np.isfinite(out["time_gaia"])]
    out["btjd"] = gaia_time_to_btjd(out["time_gaia"])
    return out[STANDARD_COLUMNS + EXTRA_COLUMNS]


def read_epoch_photometry(src):
    """Read a DR4 EPOCH_PHOTOMETRY file (VOTable/ECSV/FITS) or Table -> {source_id: standard table}."""
    if isinstance(src, Table):
        raw = src
    else:
        src = str(src)
        if src.endswith((".xml", ".vot")):
            from astropy.io.votable import parse
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = parse(src).get_first_table().to_table()
        else:
            raw = Table.read(src)
    out = {}
    for sid in np.unique(np.asarray(raw["source_id"])):
        out[int(sid)] = standardize_dr4(raw[np.asarray(raw["source_id"]) == sid])
    return out


def fetch_dr4_epoch_photometry(source_ids, cache_dir="cache/dr4_epoch_photometry", chunk=200,
                               retries=3, pause=5.0, **datalink_overrides):
    """Fetch DR4 epoch photometry with per-source caching and retries.

    Each source is cached as <cache_dir>/<source_id>.ecsv, so a crashed or
    interrupted run resumes where it stopped. Sources the archive returns
    nothing for are recorded in <cache_dir>/_empty.txt and not re-requested.
    Returns {source_id: standard table}.
    """
    from . import datalink
    from .datalink import iter_tables, split_by_source

    p = {**DR4_DATALINK, **datalink_overrides}
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    empty_file = cache / "_empty.txt"
    empty = set(int(x) for x in empty_file.read_text().split()) if empty_file.exists() else set()
    ids = [int(s) for s in source_ids]
    todo = [s for s in ids if s not in empty and not (cache / f"{s}.ecsv").exists()]

    for i in range(0, len(todo), chunk):
        block = todo[i:i + chunk]
        res, _ = datalink.load(block, p["retrieval_type"], p["data_release"], structures=p["structures"],
                               retries=retries, pause=pause)
        got = set()
        for _key, raw in iter_tables(res):
            for sid, sub in split_by_source(raw).items():
                standardize_dr4(sub).write(cache / f"{sid}.ecsv", overwrite=True)
                got.add(sid)
        new_empty = set(block) - got
        if new_empty:
            empty |= new_empty
            empty_file.write_text("\n".join(map(str, sorted(empty))))

    out = {}
    for s in ids:
        q = cache / f"{s}.ecsv"
        if q.exists():
            out[s] = Table.read(q)
    return out


def per_transit_g_noise_forecast(gmag, n_transits=70, release="dr4"):
    """Rough per-transit G uncertainty [mag] from PyGaia's mean-magnitude error model.

    sigma_transit ~ sigma_mean * sqrt(N). Ignores calibration floors that do not
    average down, so treat as a forecast and compare with real g_flux_error.
    """
    from pygaia.errors.photometric import magnitude_uncertainty
    g = np.clip(np.atleast_1d(np.asarray(gmag, float)), 4.0, 21.0)
    return magnitude_uncertainty("g", g, release=release) / 1000.0 * np.sqrt(n_transits)
