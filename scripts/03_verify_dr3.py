"""Verification on Gaia DR3 (needs network): fold DR3 epoch photometry on TESS ephemerides.

    python scripts/03_verify_dr3.py data/TEBC.csv data/tebc_dr3_xmatch.ecsv --out verify_dr3/ [--limit 20] [--no-pages]
        [--sample 50 --seed 1] [--min-period 0.5] [--max-period 20]

TEBC is sorted by period, so --limit alone picks the shortest-period (mostly contact)
systems. --sample draws a random subset instead; --min/--max-period filter first.

Needs the stage-1 file from scripts/01_prepare_dr3_xmatch.py. Downloads are cached
(cache/dr3_epoch_photometry/, plus lightkurve's own cache), so reruns are cheap.
Start with --limit 20 and look at the pages before running everything.
"""
import argparse

import numpy as np
from astropy.table import Table

from ebatlas.catalogs import load_catalog
from ebatlas.verify import verify_dr3

ap = argparse.ArgumentParser()
ap.add_argument("catalog"); ap.add_argument("xmatch")
ap.add_argument("--out", default="verify_dr3"); ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--no-pages", action="store_true"); ap.add_argument("--cache", default="cache/dr3_epoch_photometry")
ap.add_argument("--sample", type=int, default=None, help="random subset of this size (after period cuts)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-period", type=float, default=None); ap.add_argument("--max-period", type=float, default=None)
ap.add_argument("--no-refit", action="store_true", help="use catalog ephemerides as published")
ap.add_argument("--noise-curve", default="data/dr3_noise_curve.ecsv",
                help="instrumental noise floor (from diagnose_errors.py --noise-curve); used if the file exists")
ap.add_argument("--tess-cache", default="cache/tess")
a = ap.parse_args()

cat = load_catalog(a.catalog, "tebc")
cat = cat[cat["has_ephemeris"]]
if a.min_period is not None:
    cat = cat[cat["period"] >= a.min_period]
if a.max_period is not None:
    cat = cat[cat["period"] <= a.max_period]
xm = Table.read(a.xmatch)
xm = xm[np.isin(xm["eb_id"], cat["eb_id"])]
has = np.array([str(v).strip().lower() in ("true", "1") for v in xm["dr3_has_epoch_photometry"]])
xm = xm[has & (xm["dr3_source_id"] > 0)]
if a.sample and a.sample < len(xm):
    xm = xm[np.sort(np.random.default_rng(a.seed).choice(len(xm), a.sample, replace=False))]
from pathlib import Path
nc = a.noise_curve if a.noise_curve and Path(a.noise_curve).exists() else None
print(f"instrumental noise floor: {nc or 'none (per-source jitter absorbs it)'}")
res = verify_dr3(cat, xm, a.out, cache_dir=a.cache, pages=not a.no_pages, limit=a.limit, refit=not a.no_refit,
                 noise_curve=nc, tess_cache=a.tess_cache)

n = len(res)
print(f"{n} TEBC systems with DR3 epoch photometry checked")
if n:
    print(f"  with TESS light curve: {np.sum(res['has_tess'])}")
    if "refit_status" in res.colnames:
        st = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(res["refit_status"]).astype(str), return_counts=True)) if k}
        print(f"  ephemeris refit: {st}")
        ok_r = np.asarray(res["refit_status"]).astype(str) == "ok"
        if ok_r.any():
            print(f"    median sectors {np.median(np.asarray(res['refit_n_sectors'])[ok_r]):.0f}, "
                  f"median baseline {np.median(np.asarray(res['refit_baseline_days'])[ok_r]) / 365.25:.1f} yr; "
                  f"catalog period off by >3σ: {np.sum(np.abs(np.asarray(res['refit_period_shift_sigma'], float)[ok_r]) > 3)}; "
                  f"period change (|quad|>5σ): {np.sum(np.abs(np.asarray(res['refit_quad_snr'], float)[ok_r]) > 5)}")
            dr = np.abs(np.asarray(res["catalog_drift_at_gaia"], float)[ok_r])
            print(f"    catalog vs refit at Gaia epochs: median |Δphase| {np.nanmedian(dr):.4f}, >0.01 for {np.sum(dr > 0.01)}")
    if "offset_constrained" in res.colnames:
        con = np.asarray(res["offset_constrained"]).astype(bool)
        good = np.asarray(res["offset_fit_ok"]).astype(bool) if "offset_fit_ok" in res.colnames else con
        print(f"  phase offset constrained: {con.sum()}; good fits (sane depth scale, >=2 in-eclipse transits, "
              f"no single-transit leverage): {good.sum()}")
        if con.any():
            jit = np.asarray(res["offset_jitter"], float)[con]
            print(f"    median χ²_red under formal errors {np.nanmedian(np.asarray(res['offset_chi2_red_formal'], float)[con]):.1f}; "
                  f"median fitted jitter {np.nanmedian(jit) * 1e3:.1f} mmag")
        if good.any():
            z = np.asarray(res["offset_z"], float)[good]
            print(f"    good fits with |z| < 3: {np.mean(np.abs(z) < 3):.0%}")
            for r in res[good][np.abs(z) >= 3]:
                print(f"    |z|>=3: {r['eb_id']}  offset {r['offset_best_shift']:+.4f} ± {r['offset_shift_err']:.4f}  "
                      f"(ephemeris σ {r['offset_sigma_phase_gaia']:.4f}, P = {r['period']:.5f} d)")
    pc = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(res["period_check"]), return_counts=True))}
    print(f"  period check: {pc}")
    if "adopted_period_factor" in res.colnames:
        n2 = int(np.sum(np.asarray(res["adopted_period_factor"]) == 2))
        print(f"  refolded at 2x catalog period (DR3 'double' and clearly better fit): {n2}")
    if "primary_var_rejected_in_eclipse" in res.colnames:
        v = np.asarray(res["primary_var_rejected_in_eclipse"], float)
        print(f"  targets where DR3's variability pipeline rejected in-eclipse points: {np.sum(v > 0)}")
print(f"wrote {a.out}/verification.csv and verification_summary.png")
