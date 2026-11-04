# Contributing

This project is built to be picked up mid-stream, which is the point of a relay.

1. Read the [relay log](docs/RELAY_LOG.md) to see where the last person stopped.
2. Pick an issue tagged `good first issue` or `relay-handoff`, or open one.
3. Keep functions returning the standard tables (see `gaia_epochs.STANDARD_COLUMNS`
   and `catalogs.UNIFIED_COLUMNS`) so modules stay swappable.
4. Add a test for anything that touches time, phase or cross-matching. Those are
   where silent bugs live.
5. Add yourself to `CONTRIBUTORS.md` and write a short entry in the relay log
   before you leave.

Good first issues:
- Readers + tests for TESS-10k and TESS M+Ms (TEBC done; see `read_tebc` for the pattern)
- Refit TEBC ephemerides using all TESS sectors (biggest single gain in fold quality)
- Static HTML gallery from `summary.csv` and the PNGs
- Scanning-law predictions (which targets should Gaia catch in eclipse?)
- Live-test `fetch_dr3_epoch_photometry` on a known DR3 EB and fix column names
- FFI light-curve fallback for targets without SPOC data

## Pre-release data
Pre-release Gaia files go in `tests/private/` and figures made from them in `private_figures/`.
Both are gitignored. Never commit them or post figures from them before the release.
