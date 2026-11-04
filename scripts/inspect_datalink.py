"""Download a few DataLink products and show their real layout (needs network).

    python scripts/inspect_datalink.py --xmatch data/tebc_dr3_xmatch.ecsv
    python scripts/inspect_datalink.py --ids 1234567890123 --release "Gaia DR3" --type EPOCH_PHOTOMETRY

For each data structure it saves the archive's raw zip under data/datalink_samples/,
prints every table's columns, and tries the ebatlas reader on it. Paste the output
(or send one zip) when a format changes.
"""
import argparse
import os
from pathlib import Path

import numpy as np
from astropy.table import Table

from ebatlas import datalink
from ebatlas.gaia_epochs import parse_dr3_result

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="*", type=int, default=None)
ap.add_argument("--xmatch", default=None, help="take 2 DR3 IDs with epoch photometry from the stage-1 file")
ap.add_argument("--release", default="Gaia DR3"); ap.add_argument("--type", default="EPOCH_PHOTOMETRY")
ap.add_argument("--out", default="data/datalink_samples")
a = ap.parse_args()

ids = a.ids
if not ids:
    xm = Table.read(a.xmatch)
    ok = np.array([str(v).lower() in ("true", "1") for v in xm["dr3_has_epoch_photometry"]]) & (xm["dr3_source_id"] > 0)
    ids = [int(x) for x in xm["dr3_source_id"][ok][:2]]
print("source ids:", ids)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
here = os.getcwd()
for structure in datalink.STRUCTURES:
    print(f"\n===== {a.release} {a.type} {structure} =====")
    os.chdir(out)
    try:
        from astroquery.gaia import Gaia
        res = Gaia.load_data(ids=ids, data_release=a.release, retrieval_type=a.type, data_structure=structure,
                             format="votable", valid_data=False, dump_to_file=True, overwrite_output_file=True)
    except Exception as exc:
        print("REFUSED:", str(exc).splitlines()[:2]); os.chdir(here); continue
    finally:
        os.chdir(here)
    for key, t in datalink.iter_tables(res):
        print(f"-- {key}: {len(t)} rows")
        for c in t.colnames:
            v = t[c][0] if len(t) else None
            shape = np.shape(v) if v is not None else ()
            print(f"   {c:32s} {str(t[c].dtype):10s} first={str(v)[:60]!r}{'  shape=' + str(shape) if shape else ''}")
    if a.type == "EPOCH_PHOTOMETRY" and "DR3" in a.release:
        try:
            parsed = parse_dr3_result(res)
            for sid, std in parsed.items():
                print(f"   ebatlas reader OK: {sid}: {len(std)} rows, bands {sorted(set(std['band']))}")
        except Exception as exc:
            print("   ebatlas reader FAILED:", exc)
print(f"\nraw zips saved in {out}/")
