"""Load TESS EB catalogs into one unified schema.

Each catalog gets either a simple column mapping (CATALOG_MAPPINGS) or a
dedicated reader when its file needs special handling (e.g. TEBC).
Rows are keyed by `eb_id`, not `tic_id`: one TIC can host more than one
catalogued EB signal.
"""
from __future__ import annotations

import csv

import numpy as np
from astropy.table import Table

from .ephemeris import Ephemeris

UNIFIED_COLUMNS = [
    "eb_id", "tic_id", "ra", "dec", "tmag",
    "period", "period_err", "t0_btjd", "t0_err",
    "primary_width", "secondary_phase", "secondary_width",
    "morph_coeff", "source_catalog",
]

# Precision floors: catalogs round uncertainties, so a published 0 means "below the last digit".
PERIOD_ERR_FLOOR = 5e-8   # TEBC periods are given to 1e-7 d
T0_ERR_FLOOR = 5e-7       # TEBC bjd0 given to 1e-6 d

CATALOG_MAPPINGS = {
    "example": {
        "tic_id": "tic_id", "period": "period", "period_err": "period_err",
        "t0_btjd": "t0", "t0_err": "t0_err", "primary_width": "w1",
        "secondary_phase": "phi2", "secondary_width": "w2",
        "_t0_offset": 0.0,  # amount to SUBTRACT to get BTJD (2457000 if t0 is BJD)
    },
    "tess10k": None,    # Kostov et al. 2025 -- TODO
    "tess_mms": None,   # Oddo et al. 2026 -- TODO
}


def _num(values):
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def read_tebc(path):
    """Read the TESS Eclipsing Binary Catalog export (Prsa et al. 2022).

    Quirks handled (all checked against the 4584-row export):
    * every data row has a trailing comma (24 fields vs 23 header names);
    * fields have leading spaces; TIC IDs are zero-padded strings;
    * `bjd0` is already BTJD (matches each EB's first-sector start time);
    * `tic__glat` / `tic__glon` are swapped: `tic__glat` holds Galactic
      longitude l and `tic__glon` holds latitude b (verified with astropy);
    * `morph_coeff` uses -1 as a missing-value sentinel;
    * no eclipse widths or secondary phases -- measure them from TESS
      light curves with `ebatlas.eclipses.measure_eclipses`;
    * six TICs appear twice (two entries with similar periods). Both are
      kept, with eb_id suffixes "TIC<id>.01", ".02".
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh, skipinitialspace=True))
    header, data = rows[0], rows[1:]
    cols = {h: [r[i].strip() if i < len(r) else "" for r in data] for i, h in enumerate(header)}

    n = len(data)
    tic = np.array([int(x) for x in cols["tic__tess_id"]], dtype=np.int64)
    seen = {}
    eb_id = []
    for t in tic:
        seen[t] = seen.get(t, 0) + 1
        eb_id.append(f"TIC{t}.{seen[t]:02d}")

    morph = _num(cols["morph_coeff"])
    morph[morph < 0] = np.nan

    out = Table()
    out["eb_id"] = eb_id
    out["tic_id"] = tic
    out["ra"] = _num(cols["tic__ra"])
    out["dec"] = _num(cols["tic__dec"])
    out["tmag"] = _num(cols["tic__Tmag"])
    out["period"] = _num(cols["period"])
    out["period_err"] = np.fmax(_num(cols["period_uncert"]), PERIOD_ERR_FLOOR)
    out["t0_btjd"] = _num(cols["bjd0"])
    out["t0_err"] = np.fmax(_num(cols["bjd0_uncert"]), T0_ERR_FLOOR)
    out["primary_width"] = np.full(n, np.nan)
    out["secondary_phase"] = np.full(n, np.nan)   # unknown, not assumed 0.5
    out["secondary_width"] = np.full(n, np.nan)
    out["morph_coeff"] = morph
    out["source_catalog"] = ["tebc"] * n
    # Extras kept for later use
    out["glon"] = _num(cols["tic__glat"])          # sic: swapped in export
    out["glat"] = _num(cols["tic__glon"])
    out["teff"] = _num(cols["tic__teff"])
    out["logg"] = _num(cols["tic__logg"])
    out["sectors"] = cols["sectors"]
    out["tebc_flags"] = cols["flags"]
    # A missing t0 or period error means we cannot propagate the ephemeris.
    out["has_ephemeris"] = np.isfinite(out["t0_btjd"]) & np.isfinite(_num(cols["period_uncert"]))
    return out


READERS = {"tebc": read_tebc}


def load_catalog(path, name, mapping=None, fmt=None):
    """Return an astropy Table in the unified schema."""
    if name in READERS and mapping is None:
        return READERS[name](path)
    mapping = mapping or CATALOG_MAPPINGS.get(name)
    if mapping is None:
        raise NotImplementedError(
            f"No reader or column mapping for catalog {name!r} yet; add one to catalogs.py."
        )
    raw = Table.read(path, format=fmt)
    out = Table()
    n = len(raw)
    for col in UNIFIED_COLUMNS:
        if col in ("eb_id", "source_catalog"):
            continue
        src = mapping.get(col)
        if src is not None and src in raw.colnames:
            out[col] = np.asarray(raw[src], dtype=None if col == "tic_id" else float)
        else:
            out[col] = np.full(n, np.nan)
    out["t0_btjd"] = out["t0_btjd"] - mapping.get("_t0_offset", 0.0)
    out["tic_id"] = np.asarray(out["tic_id"]).astype(np.int64)
    seen = {}
    ids = []
    for t in out["tic_id"]:
        seen[t] = seen.get(t, 0) + 1
        ids.append(f"TIC{t}.{seen[t]:02d}")
    out["eb_id"] = ids
    out["source_catalog"] = [name] * n
    has = np.isfinite(out["t0_btjd"]) & np.isfinite(out["period"])
    out["has_ephemeris"] = has
    return out


def row_to_ephemeris(row, **overrides):
    """Build an Ephemeris from one unified-schema row (NaN -> unset).

    `overrides` (e.g. widths measured from the TESS light curve) take
    precedence over catalog values.
    """
    def val(k, default=None):
        if k in overrides and overrides[k] is not None:
            return float(overrides[k])
        if k not in row.colnames:
            return default
        v = row[k]
        return default if v is None or not np.isfinite(v) else float(v)

    return Ephemeris(
        period=float(row["period"]), t0=float(row["t0_btjd"]),
        period_err=val("period_err", 0.0), t0_err=val("t0_err", 0.0),
        primary_width=val("primary_width"),
        secondary_phase=val("secondary_phase", 0.5),
        secondary_width=val("secondary_width"),
    )


def apply_refit(cat, refit, require_status="ok"):
    """Copy of `cat` with ephemerides replaced by refit ones (matched on eb_id).

    `refit`: table or path from scripts/04_refit_tebc.py. Rows whose refit status is not
    `require_status` keep the catalog ephemeris. Adds `ephem_source` ('refit'|'catalog')
    and copies measured eclipse geometry when the catalog has none.
    """
    if not isinstance(refit, Table):
        refit = Table.read(refit)
    by = {str(r["eb_id"]): r for r in refit if str(r["status"]) == require_status}
    out = cat.copy()
    src = []
    for i, r in enumerate(out):
        x = by.get(str(r["eb_id"]))
        if x is None:
            src.append("catalog"); continue
        for col, rc in (("period", "refit_period"), ("period_err", "refit_period_err"),
                        ("t0_btjd", "refit_t0"), ("t0_err", "refit_t0_err")):
            out[col][i] = float(x[rc])
        for col in ("primary_width", "secondary_phase", "secondary_width"):
            if col in refit.colnames and not np.isfinite(out[col][i]) and np.isfinite(x[col]):
                out[col][i] = float(x[col])
        src.append("refit")
    out["ephem_source"] = src
    return out
