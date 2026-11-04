# ebatlas: Repository Guide

Nov 4, 2026 · @Dom

## The big picture

The repo is one pipeline: each catalog EB is cross-matched to Gaia, its Gaia epochs are folded on the catalog ephemeris, and the result is one atlas page plus one row of summary.csv.

&#91;embedded content: ebatlas pipeline · 8 steps, 1 extension\]

The catalog's ephemeris also drives the eclipse-geometry step and the fold. DR4 epoch astrometry uses the same Gaia IDs but doesn't feed the pages yet.

Two standard tables let the pieces swap in and out:

- **Catalog table** (`catalogs.UNIFIED_COLUMNS`): eb\_id, tic\_id, period, t0 in BTJD, their errors, eclipse widths. Rows key on eb\_id (e.g. TIC441794509.02) because a few TICs host two signals.
- **Epoch table** (`gaia_epochs.STANDARD_COLUMNS`): one row per band per transit with band, time\_gaia, btjd, flux, flux\_err, mag, rejected. DR4 adds transit\_id, fov, n\_ccd, n\_ccd\_outlier and suspect.

Every reader converts into these, and everything downstream reads only these. A new catalog or data release needs one converter, not changes to the plotting.

## Package modules (src/ebatlas/)

The package has 13 modules in four groups. Core logic runs offline; data access needs the Gaia archive or MAST.

**Core logic (offline)**

| File | What it does | Main calls |
| --- | --- | --- |
| ephemeris.py | Gaia time (BJD\_TCB − 2455197.5) to and from TESS BTJD, handling the \~19 s TCB/TDB offset. The Ephemeris class gives phase, propagated phase uncertainty, and in-eclipse tests in three modes: nominal, secure (in eclipse even with 2σ drift), possible. | `Ephemeris(...)`, `.phase(t)`, `.phase_sigma(t)`, `.in_primary(t, mode)`, `gaia_time_to_btjd()` |
| catalogs.py | Loads EB catalogs into the catalog table. `read_tebc` handles every TEBC quirk: trailing commas, swapped Galactic columns, duplicate TICs, −1 placeholders, errors rounded to zero. Other catalogs use a column mapping (only an `example` template so far). | `load_catalog(path, "tebc")`, `row_to_ephemeris(row)` |
| eclipses.py | Measures eclipse width, depth and secondary phase from a folded TESS light curve. TEBC lacks these, and they decide which Gaia points are in eclipse. | `measure_eclipses(ephem, t, flux)` |
| planning.py | Phase uncertainty of each ephemeris at DR4 start, middle and end. | `phase_sigma_table(cat)` |
| plotting.py | `summarize` computes per-target statistics (epoch counts, in-eclipse counts, depth S/N, Gaia-flagged outliers in eclipse). `atlas_page` draws the three-row figure. | `summarize(ephem, epochs)`, `atlas_page(...)` |
| synthetic.py | Simulated EB light curves, Gaia-like sampling, and complete DR4 epoch\_photometry files with the exact data-model columns. Used by tests and the rehearsal. | `make_dr4_epoch_photometry(ephem, source_id)` |

**Data access (network)**

| File | What it does | Main calls |
| --- | --- | --- |
| crossmatch.py | Stage 1 maps TIC → DR3 (from the Hardegree-Ullman et al. 2026 table if you have it, else TIC → DR2 → DR3 queries, else position) and attaches DR3 context. Stage 2 maps DR3 → DR4 and checks `has_epoch_photometry` in one uploaded-table query. Also finds Gaia neighbours of a target. Unconfirmed DR4 names live in `DR4_XMATCH`. | `prepare_dr3_crossmatch(cat)`, `load_tic_dr3_table(path, tic_ids)`, `dr3_to_dr4(ids)`, `gaia_neighbours(ra, dec)` |
| gaia\_epochs.py | DR3 epoch photometry fetch and conversion; its DR4 function forwards to dr4\_photometry. | `fetch_dr3_epoch_photometry(ids)` |
| dr4\_photometry.py | DR4 epoch photometry: converts the data-model format and its flags, reads downloaded files, downloads with a per-source cache and retries, forecasts per-transit noise with PyGaia. Unconfirmed release string lives in `DR4_DATALINK`. | `read_epoch_photometry(file)`, `fetch_dr4_epoch_photometry(ids)`, `per_transit_g_noise_forecast(G)` |
| tess.py | Pulls and stitches TESS light curves through lightkurve, SPOC first. | `fetch_tess_lc(tic)` |
| dr4\_tables.py | Reference list of DR4 tables relevant to the project, with the assumed schema name. Nothing to run. | — |

**DR4 epoch astrometry (the tertiary-companion extension)**

| File | What it does | Main calls |
| --- | --- | --- |
| dr4\_astrometry.py | Reads EPOCH\_ASTROMETRY files, builds barycentric times (filling missing corrections), flattens to along-scan CCD measurements, and fits a single-star model, a circular-orbit periodogram and an eccentric orbit grid. Use `include_excess_noise=False` when hunting binarity. | `read_epoch_astrometry`, `transit_times`, `al_observations`, `fit_five_parameter_al`, `astrometric_periodogram`, `orbit_grid_fit` |

**Orchestration**

| File | What it does | Main calls |
| --- | --- | --- |
| atlas.py | `build_pages` does the per-target work: light curve, eclipse geometry if missing, fold, summary, figure, summary.csv. `build_atlas_dr3` and `build_atlas_dr4` are the two full flows; `main` is the `ebatlas-build` command. | `build_pages(...)`, `ebatlas-build` |

## Scripts, examples and tests

Scripts are the two release steps; examples are self-contained demos; tests check everything offline.

| File | Purpose | Network? |
| --- | --- | --- |
| scripts/01\_prepare\_dr3\_xmatch.py | Stage 1 for all of TEBC; writes data/tebc\_dr3\_xmatch.ecsv. Pass `--tic-dr3 <file>` to use the Hardegree-Ullman et al. table instead of per-target queries. Run before the relay. | Yes (DR3 context query) |
| scripts/02\_release\_day.sh | Stage 2, fetch and pages via `ebatlas-build --release dr4`. Extra flags pass through, e.g. `--limit 10`. | Yes |
| examples/synthetic\_demo.py | Quickest install check; makes docs/synthetic\_demo.png. | No |
| examples/tebc\_ephemeris\_precision.py | Phase precision of TEBC ephemerides at DR4 epochs; makes docs/tebc\_phase\_precision.png. | No |
| examples/release\_day\_rehearsal.py | Full DR4 dress rehearsal on synthetic data-model files; writes docs/release\_rehearsal/. | No |
| examples/dr4\_astrometry\_sample.py | Diagnostics on a DR4 epoch-astrometry file; writes to private\_figures/. | No |

The tests run with `pytest`:

| Test file | Covers |
| --- | --- |
| test\_ephemeris.py | Time conversion, phase wrapping, phase-uncertainty growth, eclipse masks |
| test\_catalogs.py, test\_tebc\_and\_eclipses.py | TEBC quirks (7-row fixture in tests/data/), eclipse measurement, ragged summaries |
| test\_gaia\_epochs.py | DR3 epoch-photometry conversion |
| test\_dr4\_photometry.py | DR4 reader and flags, file round trip, outlier counting, DR3→DR4 match resolution, TIC/DR3 table loading |
| test\_dr4\_astrometry.py | Synthetic fits always; one extra test runs only when your pre-release file is in tests/private/ |

## Docs, config and data folders

Everything public lives in docs/; anything from pre-release data or large downloads is gitignored.

| Path | What it is | In git? |
| --- | --- | --- |
| README.md | Public face: science case, status table, release-day plan, figures, pitfalls | Yes |
| docs/DR4\_NOTES.md | What we learned from the draft data model and the pre-release sample. It paraphrases the draft, so move it to a private location if the draft isn't public yet. | Yes |
| docs/RELAY\_LOG.md | Dated handoff notes for the next person in the relay | Yes |
| docs/release\_rehearsal/ | Synthetic rehearsal output (fake source IDs 9000…), safe to publish | Yes |
| CONTRIBUTING.md, CONTRIBUTORS.md | Good first issues, pre-release data rules, contributor list | Yes |
| pyproject.toml | Package definition and the `ebatlas-build` command; `pip install -e ".[dev,online]"` adds pytest, astroquery, lightkurve, PyGaia and Polars | Yes |
| LICENSE | MIT; put your name in | Yes |
| data/ | Catalog files and cross-match outputs (TEBC.csv, tebc\_dr3\_xmatch.ecsv, the TIC/DR3 table) | README only |
| tests/private/ | Pre-release Gaia files | No |
| private\_figures/ | Figures made from pre-release data | No |
| cache/, atlas\*/ | Per-source DR4 downloads and atlas output | No |

The rule for pre-release data: files go only in tests/private/ and figures only in private\_figures/. Never post either before the release.

## How to use it

There are four ways in: install and test, prepare now, run on release day, and work interactively.

1. Install and check: `pip install -e ".[dev,online]"`, then `pytest`. All tests should pass.
2. Rehearse offline anytime: `python examples/release_day_rehearsal.py data/TEBC.csv`.
3. Prepare before the relay: `python scripts/01_prepare_dr3_xmatch.py data/TEBC.csv data/tebc_dr3_xmatch.ecsv`, adding `--tic-dr3 <path>` if you downloaded the TIC/DR3 table.
4. On release day: confirm `DR4_XMATCH` and `DR4_DATALINK`, run `scripts/02_release_day.sh --limit 10`, check the pages, then run without `--limit`.

For notebook work the pieces combine directly:

```python
from ebatlas.catalogs import load_catalog, row_to_ephemeris
from ebatlas.dr4_photometry import read_epoch_photometry
from ebatlas.plotting import summarize, atlas_page

cat = load_catalog("data/TEBC.csv", "tebc")
eph = row_to_ephemeris(cat[0], primary_width=0.03)   # widths: catalog, measured, or yours
epochs = read_epoch_photometry("some_dr4_file.xml")  # {source_id: table}
sid, tab = next(iter(epochs.items()))
print(summarize(eph, tab))
atlas_page(eph, tab, out="page.png")
```

For the astrometry extension, `read_epoch_astrometry` → `al_observations(tab, include_excess_noise=False)` → `fit_five_parameter_al` per source; a UWE well above 1 is a cue to run `orbit_grid_fit`.

## Cross-matching TIC to Gaia: our route vs. the published table

The published TIC/Gaia DR3 table (Hardegree-Ullman et al. 2026, RNAAS 10, 77) follows the same route as our stage 1, so the repo now uses it first and keeps our queries as the fallback. Both start from TIC v8.2's Gaia DR2 IDs and go through `gaiadr3.dr2_neighbourhood`. They differ in what they keep.

|  | Our query route | Published table |
| --- | --- | --- |
| Route | TIC v8.2 → DR2 → dr2\_neighbourhood → DR3 | Same |
| Quality cuts | None; closest match kept, candidate count recorded | Distance < 100 mas and abs(ΔG) ≤ 0.2 mag |
| TICs without a DR2 ID | Position match within 2″ | Dropped |
| Two DR3 IDs for one TIC | Closest kept, `n_dr3_candidates` = 2 | Both kept (73 TICs total) |
| One DR3 ID for two TICs | Now flagged `shared_dr3_id` | Kept (\~10 million, a TIC v7→v8 artifact) |
| Cost | \~4,600 archive queries per run | One download; ExoFOP and the NASA Exoplanet Archive also list DR3 IDs as TIC aliases |
| Citable | No | Yes: doi:10.3847/2515-5172/ae5b91, table doi:10.26134/exofop6 |

How the repo combines them (`prepare_dr3_crossmatch(cat, tic_dr3=path)`):

1. Published table first. `load_tic_dr3_table` scans it lazily with Polars, so only TEBC rows load; the 33 GB file never sits in memory. Column names are auto-detected.
2. TICs the table lacks fall back to our query route, which applies no cuts.
3. TICs with no DR2 ID fall back to the position match.
4. Every match gets `passes_hu26_cuts`, so query matches face the same standard.

The cuts matter for this project in two ways:

- **Variable stars.** The ΔG ≤ 0.2 cut compares DR2 and DR3 mean magnitudes. Mean magnitudes of deep EBs, or systems split into components in DR3, can shift by more, so some TEBC systems may be missing from the table. Those are exactly worth recovering; the fallback does it, and the flag marks them for a look by eye.
- **Shared IDs.** Two TEBC entries pointing at one Gaia star usually means one TIC is a 2MASS-era duplicate. `shared_dr3_id` catches it before you fetch the same light curve twice.

What the table doesn't replace: the DR3 context query (RUWE, DR3 epoch photometry, DR3 EB classification) and stage 2. The paper says the authors plan the same table for DR4 if Gaia ships a DR4–DR3 neighbourhood table, which the DR4 table list includes (dr3\_neighbourhood). That table probably won't exist on release day, so our stage 2 stays the day-one route; swap to theirs once published.

To use it: download the table from ExoFOP (or extract the TEBC TICs from it once and keep that small file), put it in data/, and run stage 1 with `--tic-dr3`. Check the actual column names on first load; pass `columns=` if auto-detection misfires. Cite the paper and table DOI in anything that uses it.

## Release-day checklist

Three settings are educated guesses until DR4 is published; everything else is tested.

| Setting | Where | Current guess | Confirmed by |
| --- | --- | --- | --- |
| TAP schema name | `DR4_XMATCH["schema"]` in crossmatch.py, `DR4_SCHEMA` in dr4\_tables.py | gaiadr4 | Archive table browser |
| dr3\_neighbourhood column names | `DR4_XMATCH` in crossmatch.py | dr3\_source\_id, dr4\_source\_id, angular\_distance, magnitude\_difference | Archive table browser |
| astroquery data\_release string | `DR4_DATALINK` in dr4\_photometry.py | "Gaia DR4" | astroquery docs or one test fetch |

Confirmed by the draft data model: `all_source_flags.has_epoch_photometry`, the retrieval type EPOCH\_PHOTOMETRY, the epoch\_photometry columns and flags, and barycentric epoch times.

- [ ] Stage 1 run and data/tebc\_dr3\_xmatch.ecsv saved (before the relay)
- [ ] Three settings checked against the published schema
- [ ] `scripts/02_release_day.sh --limit 10` run; a few pages checked by eye
- [ ] Full run started; cache/ resumes it if interrupted
- [ ] docs/DR4\_NOTES.md moved to private if the draft data model is not yet public
