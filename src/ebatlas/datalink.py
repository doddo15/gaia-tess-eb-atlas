"""Shared Gaia DataLink access: structure fallback, sensible retries, multi-table parsing.

The archive's release 3.15 (Nov 2026) retired data_structure='INDIVIDUAL'. Products
are now served as DATAMODEL_GAIA (Gaia's own data model, documented column by column)
or DATAMODEL_STANDARD (an IVOA-style serialisation). We try them in the order given,
remember which one worked for each (release, retrieval type), and only retry errors
that can be transient.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from astropy.table import Table

STRUCTURES = ("DATAMODEL_GAIA", "DATAMODEL_STANDARD")
_WORKING = {}          # (data_release, retrieval_type) -> structure that worked

_PERMANENT = ("deprecated", "not supported", "invalid", "unknown", "not allowed", "not valid",
              "error 400", "error 401", "error 403", "error 404")
_STRUCTURE_WORDS = ("data structure", "data_structure", "datamodel")


def is_permanent(exc):
    """Errors no retry will fix (bad parameters, retired options)."""
    m = str(exc).lower()
    return any(k in m for k in _PERMANENT)


def is_structure_error(exc):
    m = str(exc).lower()
    return is_permanent(exc) and any(k in m for k in _STRUCTURE_WORDS)


def load(ids, retrieval_type, data_release, structures=STRUCTURES, retries=3, pause=5.0, **kw):
    """Gaia.load_data with structure fallback and retries. Returns (result_dict, structure_used)."""
    from astroquery.gaia import Gaia

    key = (data_release, retrieval_type)
    order = list(structures)
    if key in _WORKING and _WORKING[key] in order:
        order.remove(_WORKING[key]); order.insert(0, _WORKING[key])
    last = None
    for structure in order:
        for attempt in range(retries):
            try:
                res = Gaia.load_data(ids=ids, data_release=data_release, retrieval_type=retrieval_type,
                                     data_structure=structure, format="votable", valid_data=False, **kw)
                _WORKING[key] = structure
                return res, structure
            except Exception as exc:
                last = exc
                if is_structure_error(exc):
                    warnings.warn(f"{data_release} {retrieval_type}: structure {structure} refused; trying next")
                    break
                if is_permanent(exc) or attempt == retries - 1:
                    raise
                warnings.warn(f"DataLink request failed ({str(exc).splitlines()[0]}); retry {attempt + 1}/{retries - 1}")
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"No data structure accepted for {data_release} {retrieval_type}: {last}")


def iter_tables(res):
    """Yield (file_key, astropy Table) for every table in every returned file."""
    for k, tables in res.items():
        for t in (tables if isinstance(tables, (list, tuple)) else [tables]):
            yield k, (t.to_table() if hasattr(t, "to_table") else Table(t))


def split_by_source(tab):
    """{source_id: rows} for a table holding one or many sources."""
    cols = {c.lower(): c for c in tab.colnames}
    sc = cols.get("source_id")
    if sc is None:
        raise ValueError(f"No source_id column in DataLink table: {tab.colnames[:15]}")
    sid = np.asarray(tab[sc], dtype=np.int64)
    return {int(s): tab[sid == s] for s in np.unique(sid)}
