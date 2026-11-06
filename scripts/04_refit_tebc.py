"""Refit all TEBC ephemerides from every TESS sector (needs network; resumable).

    python scripts/04_refit_tebc.py data/TEBC.csv data/tebc_refit.csv [--workers 4] [--sample 100 --seed 0]
        [--tess-cache cache/tess] [--oc-dir refit_oc/] [--keep-raw]

Run --sample 100 first to check timing and disk use. Re-running the same command
resumes: finished targets are skipped. Light curves are cached compactly in
--tess-cache (~0.5 MB for a typical target, up to ~5 MB in the CVZ; ~3-6 GB for all of TEBC); FITS downloads are deleted unless --keep-raw.
--oc-dir writes O-C diagrams for flagged targets. Ends with a summary and
<out stem>_summary.png.
"""
import argparse

import numpy as np

from ebatlas.catalogs import load_catalog
from ebatlas.refit_catalog import refit_catalog
from ebatlas.refit_summary import summarize_refit

ap = argparse.ArgumentParser()
ap.add_argument("catalog"); ap.add_argument("out")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--tess-cache", default="cache/tess")
ap.add_argument("--sample", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--oc-dir", default=None); ap.add_argument("--keep-raw", action="store_true")
a = ap.parse_args()

cat = load_catalog(a.catalog, "tebc")
cat = cat[cat["has_ephemeris"]]
if a.sample and a.sample < len(cat):
    cat = cat[np.sort(np.random.default_rng(a.seed).choice(len(cat), a.sample, replace=False))]
refit_catalog(cat, a.out, workers=a.workers, tess_cache=a.tess_cache, keep_raw=a.keep_raw, oc_dir=a.oc_dir)
summarize_refit(a.out)
