# DR4 notes (from the draft data model and a DR4_RC3 pre-release sample)

## Epoch astrometry (`epoch_astrometry`, DataLink `EPOCH_ASTROMETRY`)
- One row per (source_id, transit_id); per-CCD fields are length-10 arrays, SM=0, AF1..AF9=1..9.
- `obs_time_tcb`: int64 ns since J2010.0 TCB (= JD 2455197.5), **at Gaia**, per CCD. SM→AF9 spans ~46 s.
- `obs_time_bary_corr`: TCB(bary) − TCB(Gaia) in ns, at AF4. Verified against astropy's Rømer delay
  (agreement within 4.4 s, the Gaia–geocentre offset). ~10% NaN in RC3; we fill with the geocentric estimate and flag it.
- The VOTable `TIMESYS` says `refposition="BARYCENTER"`, contradicting the data model. We follow the data model.
  Worth reporting to the Gaia helpdesk.
- AC centroids are empty in DR4 (1-D astrometry). `centroid_pos_al` has rare gross outliers; use `used_by_agis_al`.
- Convention check: a 5-parameter fit with dw/dα* = sin ψ, dw/dδ = cos ψ, weights from
  `centroid_pos_error_al` ⊕ `agis_source_excess_noise` gives UWE ≈ 1.0 for well-behaved sources.
- Epoch astrometry has **no epoch fluxes** (`g_mag` is the mean value used for centroiding).
  The eclipse atlas needs `epoch_photometry` (DataLink, 7.2 TB).

## Epoch photometry (`epoch_photometry`, DataLink `EPOCH_PHOTOMETRY`)
- Wide format: one row per (source_id, transit_id) with `g_*`, `bp_*`, `rp_*` columns. Reader: `ebatlas.dr4_photometry`.
- `g_obs_time`, `bp_obs_time`, `rp_obs_time` are **already barycentric** (BJD_TCB − 2455197.5). Contrast with epoch astrometry.
- `g_n_obs` ∈ [0, 9] (SM not used); 0 = all AF fluxes discarded. BP/RP fluxes are NaN when rejected.
- `phot_ccd_proc_flags` (short[12]: SM, AF1–9, BP, RP), bit 7 = rejected from the *mean* photometry only. Transit-level
  fluxes use all calibrated CCDs, so **eclipses are not removed by outlier rejection**. We count in-eclipse transits with
  bit 7 set (`*_gaia_outlier_transits`): a measure of Gaia's pipeline noticing the eclipse.
- Variability-pipeline epoch flags are a separate product (`EPOCH_FLAGS_VARI`).
- `epoch_photometry_ccd` (`EPOCH_PHOTOMETRY_CCD`) has per-AF-CCD G fluxes over ~44 s. Not needed for folding.
- astroquery 0.4.11 already lists the DR4 retrieval types. Since archive release 3.15 (Nov 2026) the data structures
  are `DATAMODEL_GAIA` (the layout this data model documents; requested first) and `DATAMODEL_STANDARD` (IVOA-style
  fallback); `INDIVIDUAL` is retired. `ebatlas.datalink` handles the fallback and remembers what worked.

## Products that matter for this project
See `src/ebatlas/dr4_tables.py`. Highlights: `epoch_photometry`, `rvs_epoch_parameters_single/double`
(phase-fold epoch RVs on TESS ephemerides), `vari_epoch_flags` (were in-eclipse points rejected?),
`nss_acceleration_astro`, `nss_multiple_orbits`, `dr3_neighbourhood` (DR3→DR4), `all_source_flags`,
and `commanded_scan_law`.

## Still unconfirmed (check on release day)
- TAP schema name (`gaiadr4`?) and the column names of DR4 `dr3_neighbourhood` (`DR4_XMATCH` in `crossmatch.py`).
- The astroquery `data_release` string (`DR4_DATALINK` in `dr4_photometry.py`).
- The DR4 photometric zero points (only fluxes are used, so this does not affect the atlas).
- A real epoch_photometry sample would let us test the reader on actual files instead of data-model-faithful synthetics.
