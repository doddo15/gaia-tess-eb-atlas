"""DR4 archive products relevant to this project (from the ESA DR4 expected-content table).

Names are archive table names; the TAP schema prefix (e.g. "gaiadr4.") is
not confirmed yet -- set DR4_SCHEMA on release day.
"""
DR4_SCHEMA = "gaiadr4"   # TODO(relay): confirm

TABLES = {
    # Cross-matching (no DR2 link in DR4: TIC -> DR2 -> DR3 -> DR4)
    "dr3_neighbourhood": ("TAP", "DR3 -> DR4 cross-match"),
    "all_source_flags": ("TAP", "per-source flags incl. has_epoch_astrometry; covers all 2.8e9 sources"),
    "all_source_photometry": ("TAP", "photometry for all sources (EB hosts may be absent from gaia_source)"),
    "all_source_astrometry": ("TAP", "astrometry for all sources"),
    "gaia_source": ("TAP", "high-quality ~2e9 subset; best-suited model per source"),
    # Epoch data
    "epoch_photometry": ("DataLink", "per-transit G/BP/RP photometry -- the eclipse atlas core (7.2 TB)"),
    "epoch_photometry_ccd": ("DataLink", "per-CCD photometry within transits"),
    "epoch_astrometry": ("DataLink", "per-CCD AL astrometry (retrieval type EPOCH_ASTROMETRY)"),
    "rvs_epoch_parameters_single": ("DataLink", "epoch RVs: phase-fold on TESS ephemerides for SB1 curves"),
    "rvs_epoch_parameters_double": ("DataLink", "epoch RVs for double-lined systems"),
    "vari_epoch_flags": ("DataLink", "variability-pipeline epoch flags; retrieval type EPOCH_FLAGS_VARI"),
    "nss_epoch_flags": ("DataLink", "NSS-pipeline epoch flags"),
    # Source-level results to compare against TESS
    "vari_eclipsing_binary": ("TAP", "DR4 EB solutions (2.39e6 rows)"),
    "vari_summary": ("TAP", "variability summary"),
    "vari_classifier_result": ("TAP", "variability classes"),
    "nss_two_body_orbit": ("TAP", "orbital solutions (5.9e6)"),
    "nss_acceleration_astro": ("TAP", "astrometric acceleration solutions -- tertiary candidates"),
    "nss_multiplicity": ("TAP", "multiplicity information (2.29e8)"),
    "nss_multiple_orbits": ("TAP", "multiple-orbit solutions -- hierarchical triples"),
    "nss_masses": ("TAP", "masses"),
    "rvs_two_body_orbit": ("TAP", "RVS orbital solutions"),
    "commanded_scan_law": ("TAP", "scanning law: predict which epochs fall in eclipse"),
    "gaia_gap": ("TAP", "data gaps"),
}


def qualified(name):
    return f"{DR4_SCHEMA}.{name}"
