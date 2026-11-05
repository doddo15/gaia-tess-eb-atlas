"""TESS coverage audit (needs network for the MAST search; tess-point is offline).

    python scripts/audit_tess_coverage.py data/TEBC.csv verify_dr3_refit/verification.csv \\
        [--sector-status verify_dr3_refit/sector_status.csv] [--max-sector N]

For each target in the verification file: sectors observed (tess-point), products at
MAST per pipeline, and (with --sector-status) what the refit did with each sector.
--max-sector defaults to the latest sector with any product in this run (tess-point
also lists planned sectors). Writes <dir>/coverage_audit.csv and coverage_map.png.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

from ebatlas.catalogs import load_catalog
from ebatlas.coverage import observed_sectors, available_products, audit_target, coverage_figure

ap = argparse.ArgumentParser()
ap.add_argument("catalog"); ap.add_argument("verification")
ap.add_argument("--sector-status", default=None); ap.add_argument("--max-sector", type=int, default=None)
a = ap.parse_args()

cat = load_catalog(a.catalog, "tebc")
by_eb = {r["eb_id"]: r for r in cat}
ver = Table.read(a.verification)
status = Table.read(a.sector_status) if a.sector_status else None

prods = {}
for eb in ver["eb_id"]:
    prods[eb] = available_products(by_eb[eb]["tic_id"])
max_sector = a.max_sector or max((p["sector"] for v in prods.values() for p in v), default=0)
print(f"latest sector with any product among these targets: {max_sector}"
      + ("" if a.max_sector else " (used as the 'observed so far' cutoff; override with --max-sector)"))

rows = []
for eb in ver["eb_id"]:
    r = by_eb[eb]
    obs = observed_sectors(r["ra"], r["dec"], max_sector)
    st = status[np.asarray(status["eb_id"]) == eb] if status is not None else None
    rec = {"eb_id": eb, "tic_id": int(r["tic_id"]), "tmag": float(r["tmag"]),
           "tebc_sectors": str(r["sectors"]).replace(";", ","), **audit_target(obs, prods[eb], st)}
    rec["observed"] = ",".join(map(str, obs))
    rec["product_sectors"] = ",".join(map(str, sorted({p["sector"] for p in prods[eb]})))
    if st is not None and len(st):
        rec["timed_sectors"] = ",".join(str(int(s)) for s, v in zip(st["sector"], st["status"]) if v == "ok")
    rows.append(rec)

from ebatlas.atlas import _rows_to_table
aud = _rows_to_table(rows)
out = Path(a.verification).parent
aud.write(out / "coverage_audit.csv", overwrite=True)
coverage_figure(aud, out / "coverage_map.png", max_sector)

n = len(aud)
col = lambda c: np.asarray(aud[c], float)
print(f"{n} targets")
print(f"  observed sectors per target: median {np.median(col('n_observed')):.0f}  "
      f"(first {int(col('first_observed').min())}, last {int(col('last_observed').max())})")
print(f"  with a SPOC/TESS-SPOC/QLP product: median {np.median(col('n_with_product')):.0f}; "
      f"targets missing >=1 observed sector: {np.sum(col('n_missing') > 0)} "
      f"(median {np.median(col('n_missing')):.0f} missing)")
print(f"  latest product sector per target: median {np.median(col('last_product')):.0f}; "
      f"targets whose latest product is >10 sectors before the cutoff: {np.sum(col('last_product') < max_sector - 10)}")
if "n_timed" in aud.colnames:
    print(f"  timed in refit: median {np.nanmedian(col('n_timed')):.0f} sectors; "
          f"untimed reasons (totals): " + ", ".join(f"{k[2:]}={int(np.nansum(col(k)))}" for k in
          ("n_no_eclipse", "n_unconstrained", "n_too_few_points", "n_bad_noise", "n_negative_scale") if k in aud.colnames))
mag = col("tmag")
for lo, hi in ((0, 11), (11, 13), (13, 20)):
    m = (mag >= lo) & (mag < hi)
    if m.any():
        print(f"  Tmag {lo}-{hi}: n={m.sum():2d}, median fraction of observed sectors with a product "
              f"{np.median(col('n_with_product')[m] / np.maximum(col('n_observed')[m], 1)):.0%}")
print(f"wrote {out}/coverage_audit.csv and coverage_map.png")
