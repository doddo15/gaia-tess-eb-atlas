# Relay log

Short, dated entries: what you did, what's broken, what the next person should do.

## Pre-relay (setup)
- Package skeleton, ephemeris/time handling with tests, atlas plotting, synthetic demo.
- DR3 fetch and cross-match code written but not yet run against the live archive.
- **Next:** catalog mappings; live DR3 validation; scanning-law predictions.

## Pre-relay: TEBC
- Added `read_tebc` for the TEBC export (4584 rows; 4574 with usable ephemerides; 6 TICs with two entries, keyed by `eb_id`).
- Added `eclipses.measure_eclipses` (widths, depths, secondary phase from TESS; handles eccentric systems).
- Added `planning.phase_sigma_table` and `examples/tebc_ephemeris_precision.py`. Median sigma_phase at DR4 start is 0.013; only 34% < 0.005.
- **Next:** refit ephemerides with all TESS sectors; live-test DR3 fetch on a few TEBC targets; learn the DR4 epoch data model.

## Pre-relay: DR4 epoch astrometry
- Added `dr4_astrometry` (reader, transit_id decoding, barycentric times with NaN fill, AF-only AL observations,
  5-param fit, circular astrometric periodogram, Keplerian grid fit) and `dr4_tables` (relevant DR4 products).
- Validated on a 12-source DR4_RC3 sample (private, gitignored): UWE ≈ 1 for well-behaved sources; time-system findings in docs/DR4_NOTES.md.
- **Next:** epoch_photometry data model + sample; fill `fetch_dr4_epoch_photometry`.

## Pre-relay: DR4 epoch photometry + staged cross-match
- `dr4_photometry`: reader for the DR4 data model (wide format, barycentric times, g_n_obs, phot_ccd_proc_flags), cached/resumable fetch, PyGaia-based noise forecast.
- `synthetic.make_dr4_epoch_photometry` + `examples/release_day_rehearsal.py`: offline end-to-end rehearsal; recovered eclipse widths within ~5% and an eccentric secondary phase.
- `crossmatch`: stage 1 `prepare_dr3_crossmatch` (run now), stage 2 `dr3_to_dr4` (one uploaded-table query).
- **Next:** run stage 1 with network access; confirm DR4 schema names on release day.

## Pre-relay: published TIC/DR3 table
- `crossmatch.load_tic_dr3_table` reads the Hardegree-Ullman et al. (2026) TIC/Gaia DR3 table lazily (Polars); `prepare_dr3_crossmatch(tic_dr3=...)` uses it first, then queries, then position.
- New columns: `match_method`, `angular_distance_mas`, `g_mag_difference`, `passes_hu26_cuts`, `shared_dr3_id`.
- **Next:** download the table (or a TEBC-only extract) and run stage 1 with `--tic-dr3`.

## Pre-relay: DR3 verification
- `verify.py` + `scripts/03_verify_dr3.py`: period check vs. DR3 vari_eclipsing_binary, phase-offset scan of a TESS template
  against Gaia G (Fourier outside eclipses, running mean inside, bootstrap template errors), eclipse recovery.
- Error calibration on synthetics: pull widths 0.92–1.15 (2-min and 10-min TESS cadence, eclipse widths 0.02–0.06).
- DR3 reader change: `rejected_by_variability` no longer removes points; it is a separate column, ringed on pages and counted.
- DR3 fetch is now cached per source (cache/dr3_epoch_photometry/).
- **Next:** run stage 1, then `03_verify_dr3.py --limit 20`; inspect pages before the full run.
