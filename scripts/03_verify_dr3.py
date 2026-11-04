"""Verification on Gaia DR3 (needs network): fold DR3 epoch photometry on TESS ephemerides.

    python scripts/03_verify_dr3.py data/TEBC.csv data/tebc_dr3_xmatch.ecsv --out verify_dr3/ [--limit 20] [--no-pages]

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
a = ap.parse_args()

cat = load_catalog(a.catalog, "tebc")
cat = cat[cat["has_ephemeris"]]
res = verify_dr3(cat, Table.read(a.xmatch), a.out, cache_dir=a.cache, pages=not a.no_pages, limit=a.limit)

n = len(res)
print(f"{n} TEBC systems with DR3 epoch photometry checked")
if n:
    print(f"  with TESS light curve: {np.sum(res['has_tess'])}")
    if "offset_constrained" in res.colnames:
        con = np.asarray(res["offset_constrained"]).astype(bool)
        z = np.asarray(res["offset_z"], float)[con]
        print(f"  phase offset constrained: {con.sum()};  |z| < 3: {np.mean(np.abs(z) < 3):.0%}" if con.any() else "  no constrained offsets")
    pc = dict(zip(*np.unique(np.asarray(res["period_check"]), return_counts=True)))
    print(f"  period check: {pc}")
    if "primary_var_rejected_in_eclipse" in res.colnames:
        v = np.asarray(res["primary_var_rejected_in_eclipse"], float)
        print(f"  targets where DR3's variability pipeline rejected in-eclipse points: {np.sum(v > 0)}")
print(f"wrote {a.out}/verification.csv and verification_summary.png")
