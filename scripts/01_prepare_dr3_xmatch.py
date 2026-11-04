"""Stage 1 (run now, needs network): TEBC -> Gaia DR3 IDs + DR3 context.

    python scripts/01_prepare_dr3_xmatch.py data/TEBC.csv data/tebc_dr3_xmatch.ecsv

On DR4 release day only the DR3 -> DR4 step remains (scripts/02_release_day.sh).
"""
import sys

import numpy as np

from ebatlas.catalogs import load_catalog
from ebatlas.crossmatch import prepare_dr3_crossmatch

src, dst = sys.argv[1], sys.argv[2]
cat = load_catalog(src, "tebc")
cat = cat[cat["has_ephemeris"]]
xm = prepare_dr3_crossmatch(cat)
xm.write(dst, overwrite=True)

ok = xm["dr3_source_id"] > 0
print(f"{len(xm)} EBs; {ok.sum()} with a DR3 match "
      f"({(xm['match_method'] == 'positional').sum()} positional, {(xm['n_dr3_candidates'] > 1).sum()} ambiguous)")
hep = np.asarray(xm["dr3_has_epoch_photometry"], bool)
eb = np.isfinite(np.asarray(xm["dr3_eb_frequency"], float))
print(f"DR3 already had epoch photometry for {hep.sum()}, and classified {eb.sum()} as EBs.")
print(f"=> {ok.sum() - hep.sum()} matched TEBC systems will get Gaia epoch photometry for the first time in DR4 "
      f"(if DR4 publishes it for them).")
print("wrote", dst)
