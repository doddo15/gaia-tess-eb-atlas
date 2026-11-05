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

## 2026-11-04: first live runs
- Stage 1 on TEBC: 4564/4574 matched (24 positional, 291 with >1 DR3 candidate); 1966 have DR3 epoch photometry,
  1454 classified EB by DR3; 2598 matched systems have no DR3 epoch photometry (new in DR4, if published).
- `03_verify_dr3.py` failed: Gaia archive release 3.15 retired data_structure='INDIVIDUAL'.
  Fix: new `datalink.py` (DATAMODEL_GAIA then DATAMODEL_STANDARD, remembers the working one, no retries on
  permanent errors, parses multi-source/multi-table responses); DR3 reader accepts column-name aliases and saves
  unparseable tables to cache/.../_unparsed/; `scripts/inspect_datalink.py` dumps real responses.
- **Next:** run `inspect_datalink.py --xmatch data/tebc_dr3_xmatch.ecsv`, then rerun verification with --limit 20.

## 2026-11-04: first verification run (20 shortest-period TEBC systems)
- `inspect_datalink.py`: DATAMODEL_GAIA accepted and parsed (wide, per-source arrays); DATAMODEL_STANDARD is one
  file per source, one row per transit, with the source ID only in the file name.
- Result: 18/20 offsets constrained, |z|<3 for 72%; DR3 period 'double' for 6, not a DR3 EB for 14.
  These are P = 0.13-0.17 d, below the ~0.22 d contact-binary limit: likely TEBC half-period aliases.
- Fixes: (1) DATAMODEL_GAIA photometry_flag_*_reject / *_unavailable now set `rejected` (they were ignored, which
  is where the 10-35x BP/RP points came from); (2) DATAMODEL_STANDARD parsed, ID from file name; (3) normalised
  flux > 2 flagged as rejected (MAX_NORM_FLUX); (4) TESS normalised by its out-of-eclipse median like Gaia
  (it used the all-points median, so it floated above Gaia for deep/contact systems); (5) robust y-limits with
  off-scale triangles; zoom panels capped at ±0.5 phase; (6) TESS upward outliers clipped at 5σ (no lower clip);
  (7) verification tries 2P when DR3 says 'double' and adopts it if chi2_red improves by >10%;
  (8) `03_verify_dr3.py --sample/--seed/--min-period/--max-period`.
- **Next:** rerun with `--sample 50 --seed 1 --min-period 0.25` for a representative check; refine ephemerides
  from all TESS sectors (multiple tracks in folded TESS = period error across sectors).

## 2026-11-04: random-sample verification (50, P > 0.25 d) and ephemeris refits
- Sample result: 31/49 offsets constrained, but median chi2_red of the offset fits was 107 and the TESS-template
  bootstrap dominated every error: the multi-year TESS fold was smeared by the catalog (PM) period, so the
  |z|>=3 list from that run is not evidence of period changes yet. 10 period "mismatches" were mostly long-period
  detached systems with odd ratios (DR3 aliases, not our problem).
- `tess.fetch_tess_lc` now uses SPOC, TESS-SPOC and QLP, one product per sector (SPOC 120 s > SPOC 20 s >
  TESS-SPOC > QLP), returning a `TessLC` with sector, exposure time and pipeline per point.
- New `ephemeris_fit.refit_ephemeris`: per-sector timings (template smeared to each sector's exposure),
  progressive fitting outward from the catalog epoch, shift-and-stack templates, 5 s timing floor, chi2
  inflation, quadratic (period-change) test. On synthetics: constant P recovered to <0.5 s per sector;
  a 2e-9 d/cycle period change detected at ~190 sigma (stacking on a linear fold had biased timings by ~100 s).
- Verification refits by default (`--no-refit` to skip), records refit_* columns and catalog_drift_at_gaia,
  writes <eb_id>_oc.png (TESS sector O-C + the Gaia-era point), and flags offset_fit_ok
  (constrained, depth scale 0.2-5, chi2_red < 20); summary statistics use good fits only.
- New `templates.py` (shared template code, gridded for speed); layout-fallback warning names the page.
- **Next:** rerun the same sample (`--sample 50 --seed 1 --min-period 0.25`) and compare.
