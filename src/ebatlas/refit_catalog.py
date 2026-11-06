"""Refit every ephemeris in a catalog from all TESS sectors: resumable, parallel, crash-tolerant.

Outputs (CSV, appended target by target so an interrupted run resumes where it stopped):
  <out>              one row per EB: catalog and refit ephemerides, quality numbers, flags
  <out>_timings.csv  every sector timing (for O-C work)
  <out>_sectors.csv  every sector's status (timed, or why not)
Flags: period_change (|quad_snr| > 5 against the fitted timing jitter), catalog_superseded
(refit period > 3 sigma from the catalog's), variable_depth (depth-scale spread within one
pipeline > 10%), catalog_far_off (first pass needed a whole-orbit search).
"""
from __future__ import annotations

import csv
import time
import traceback
from pathlib import Path

import numpy as np

from .ephemeris import Ephemeris, _wrap
from .planning import DR4_MID_BTJD

SUMMARY_FIELDS = [
    "eb_id", "tic_id", "status", "error",
    "cat_period", "cat_period_err", "cat_t0", "cat_t0_err",
    "refit_period", "refit_period_err", "refit_t0", "refit_t0_err",
    "n_sectors_available", "n_sectors", "first_sector_timed", "last_sector_timed", "baseline_days",
    "chi2_red", "jitter_s", "quad_snr", "period_shift_sigma", "n_outliers", "n_wide", "depth_scale_spread",
    "drift_dr4_mid", "drift_last_sector", "primary_width", "secondary_phase", "secondary_width",
    "authors", "flag_period_change", "flag_catalog_superseded", "flag_variable_depth", "flag_catalog_far_off",
    "elapsed_s",
]
TIMING_FIELDS = ["eb_id", "sector", "author", "exptime", "n_points", "n_in_eclipse", "E", "T", "T_err",
                 "O_C_min", "T_err_min", "depth_scale", "wide"]
SECTOR_FIELDS = ["eb_id", "sector", "author", "exptime", "n_points", "status", "n_in_eclipse", "depth_scale", "wide"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def refit_one(row, tess_cache="cache/tess", keep_raw=False, lc_getter=None, oc_dir=None):
    """Refit one catalog row (a dict). Returns (summary, timing_rows, sector_rows); never raises."""
    from .ephemeris_fit import refit_ephemeris, oc_figure
    from .tess import fetch_tess_lc

    t_start = time.time()
    s = {k: "" for k in SUMMARY_FIELDS}
    s.update(eb_id=row["eb_id"], tic_id=int(row["tic_id"]), cat_period=_f(row["period"]),
             cat_period_err=_f(row["period_err"]), cat_t0=_f(row["t0_btjd"]), cat_t0_err=_f(row["t0_err"]))
    try:
        cat = Ephemeris(_f(row["period"]), _f(row["t0_btjd"]), np.nan_to_num(_f(row["period_err"])),
                        np.nan_to_num(_f(row["t0_err"])))
        lc = (lc_getter or (lambda tic: fetch_tess_lc(tic, cache_dir=tess_cache, keep_raw=keep_raw)))(int(row["tic_id"]))
        if lc is None:
            s.update(status="no_tess", elapsed_s=time.time() - t_start)
            return s, [], []
        s["n_sectors_available"] = int(len(np.unique(lc.sector)))
        s["authors"] = ";".join(sorted(set(map(str, lc.author))))
        fit = refit_ephemeris(cat, lc)
        e = fit["ephem"]
        s.update(status=fit["status"], n_sectors=fit["n_sectors"], baseline_days=fit["baseline_days"],
                 chi2_red=fit["chi2_red"], jitter_s=fit["jitter_s"], quad_snr=fit["quad_snr"],
                 period_shift_sigma=fit["period_shift_sigma"], n_outliers=fit["n_outliers"], n_wide=fit["n_wide"],
                 depth_scale_spread=fit["depth_scale_spread"], primary_width=e.primary_width,
                 secondary_phase=e.secondary_phase, secondary_width=e.secondary_width)
        tim = fit["timings"]
        if len(tim):
            s.update(first_sector_timed=int(np.min(tim["sector"])), last_sector_timed=int(np.max(tim["sector"])))
        if fit["status"] == "ok":
            s.update(refit_period=e.period, refit_period_err=e.period_err, refit_t0=e.t0, refit_t0_err=e.t0_err,
                     drift_dr4_mid=float(_wrap(cat.phase(DR4_MID_BTJD) - e.phase(DR4_MID_BTJD))),
                     drift_last_sector=float(_wrap(cat.phase(np.max(tim["T"])) - e.phase(np.max(tim["T"])))))
            s.update(flag_period_change=bool(abs(_f(fit["quad_snr"])) > 5),
                     flag_catalog_superseded=bool(abs(_f(fit["period_shift_sigma"])) > 3),
                     flag_variable_depth=bool(_f(fit["depth_scale_spread"]) > 0.10),
                     flag_catalog_far_off=bool(fit["n_wide"] > 0))
            if oc_dir and (s["flag_period_change"] or s["flag_variable_depth"] or s["flag_catalog_far_off"]):
                Path(oc_dir).mkdir(parents=True, exist_ok=True)
                oc_figure(fit, Path(oc_dir) / f"{row['eb_id']}_oc.png", title=row["eb_id"])
        timing_rows = [{"eb_id": row["eb_id"], **{k: r[k] for k in TIMING_FIELDS[1:] if k in tim.colnames}} for r in tim]
        st = fit["sector_status"]
        sector_rows = [{"eb_id": row["eb_id"], **{k: r[k] for k in SECTOR_FIELDS[1:] if k in st.colnames}} for r in st]
        s["elapsed_s"] = time.time() - t_start
        return s, timing_rows, sector_rows
    except Exception as exc:
        s.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300], elapsed_s=time.time() - t_start)
        s["error"] += " | " + traceback.format_exc().strip().splitlines()[-2][:200]
        return s, [], []


class _Appender:
    def __init__(self, path, fields):
        self.path = Path(path); self.fields = fields
        new = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.fh, fieldnames=fields, extrasaction="ignore")
        if new:
            self.w.writeheader()

    def write(self, rows):
        for r in rows:
            self.w.writerow(r)
        self.fh.flush()

    def close(self):
        self.fh.close()


def done_ids(path):
    p = Path(path)
    if not p.exists():
        return set()
    with open(p, newline="") as fh:
        return {r["eb_id"] for r in csv.DictReader(fh)}


def refit_catalog(cat, out, workers=4, tess_cache="cache/tess", keep_raw=False, lc_getter=None,
                  oc_dir=None, progress_every=25):
    """Refit every row of a unified catalog (rows with has_ephemeris only). Resumes from `out`."""
    out = Path(out)
    stem = out.with_suffix("")
    todo_rows = [{k: r[k] for k in cat.colnames} for r in cat
                 if ("has_ephemeris" not in cat.colnames or r["has_ephemeris"])]
    done = done_ids(out)
    todo_rows = [r for r in todo_rows if r["eb_id"] not in done]
    print(f"{len(done)} already done, {len(todo_rows)} to refit")
    summ = _Appender(out, SUMMARY_FIELDS)
    tim = _Appender(f"{stem}_timings.csv", TIMING_FIELDS)
    sec = _Appender(f"{stem}_sectors.csv", SECTOR_FIELDS)
    t0 = time.time()
    try:
        if workers <= 1 or lc_getter is not None:
            results = (refit_one(r, tess_cache, keep_raw, lc_getter, oc_dir) for r in todo_rows)
            for i, (s, tr, sr) in enumerate(results, 1):
                summ.write([s]); tim.write(tr); sec.write(sr)
                if i % progress_every == 0:
                    _progress(i, len(todo_rows), t0)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(refit_one, r, tess_cache, keep_raw, None, oc_dir) for r in todo_rows]
                for i, fu in enumerate(as_completed(futs), 1):
                    s, tr, sr = fu.result()
                    summ.write([s]); tim.write(tr); sec.write(sr)
                    if i % progress_every == 0:
                        _progress(i, len(todo_rows), t0)
    finally:
        summ.close(); tim.close(); sec.close()
    return out


def _progress(i, n, t0):
    el = time.time() - t0
    print(f"  {i}/{n} done, {el / 60:.1f} min elapsed, ~{el / i * (n - i) / 60:.0f} min left", flush=True)
