#!/usr/bin/env bash
# Release day. Before running: confirm DR4_XMATCH (crossmatch.py) and DR4_DATALINK
# (dr4_photometry.py) against the published schema, then try --limit 10 first.
set -euo pipefail
ebatlas-build data/TEBC.csv --catalog tebc --release dr4 \
    --xmatch data/tebc_dr3_xmatch.ecsv --refit data/tebc_refit.csv --out atlas_dr4/ "$@"
