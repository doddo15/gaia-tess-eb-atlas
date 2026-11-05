"""TESS light curves from every available pipeline, one per sector.

`fetch_tess_lc` returns a `TessLC`: a tuple (t, flux, flux_err, sector, exptime, author)
so code that only needs (t, flux, ...) can keep indexing [0], [1]. Per sector, the
preferred product is chosen by AUTHOR_PRIORITY and then by cadence (see
`select_products`). Each sector is normalised by its own median; upward outliers are
clipped, never downward ones (eclipses).

Why several pipelines: SPOC 2-min data exist only for pre-selected targets, so many
EBs have them in a few sectors at most; TESS-SPOC and QLP light curves come from the
full-frame images and cover far more sectors, which is what an ephemeris refit needs.
FFI cadence is 1800 s (cycles 1-2), 600 s (3-4) and 200 s (5+); `exptime` lets later
steps account for the resulting smearing.
"""
from __future__ import annotations

import re
import warnings
from collections import namedtuple

import numpy as np

TessLC = namedtuple("TessLC", "t flux err sector exptime author")

AUTHOR_PRIORITY = ("SPOC", "TESS-SPOC", "QLP")
PREFERRED_CADENCE = {"SPOC": (120.0, 20.0)}     # within SPOC, 2-min before 20-s (same photometry, 6x fewer points)


def _sector_of(mission):
    m = re.search(r"(\d+)", str(mission))
    return int(m.group(1)) if m else -1


def select_products(rows, authors=AUTHOR_PRIORITY):
    """Choose one product per sector.

    rows: iterable of dicts with keys index, sector, author, exptime (s).
    Order: author priority, then PREFERRED_CADENCE for that author, then shortest exptime.
    Returns the chosen rows sorted by sector.
    """
    rank_a = {a: i for i, a in enumerate(authors)}
    best = {}
    for r in rows:
        a = str(r["author"])
        if a not in rank_a or r["sector"] < 0:
            continue
        pref = PREFERRED_CADENCE.get(a, ())
        cad_rank = pref.index(float(r["exptime"])) if float(r["exptime"]) in pref else len(pref)
        key = (rank_a[a], cad_rank, float(r["exptime"]))
        if r["sector"] not in best or key < best[r["sector"]][0]:
            best[r["sector"]] = (key, r)
    return [best[s][1] for s in sorted(best)]


def fetch_tess_lc(tic_id, authors=AUTHOR_PRIORITY, quality_bitmask="default", sigma_upper=5.0,
                  sectors=None):
    """All sectors of TESS photometry for one TIC, one product per sector, or None.

    `sectors` optionally restricts to a list of sector numbers.
    """
    import lightkurve as lk

    sr = lk.search_lightcurve(f"TIC {int(tic_id)}", mission="TESS")
    if len(sr) == 0:
        return None
    exptimes = np.atleast_1d(getattr(sr.exptime, "value", sr.exptime)).astype(float)
    rows = [{"index": i, "sector": _sector_of(sr.mission[i]), "author": str(sr.author[i]),
             "exptime": float(exptimes[i])} for i in range(len(sr))]
    chosen = select_products(rows, authors)
    if sectors is not None:
        chosen = [r for r in chosen if r["sector"] in set(sectors)]
    if not chosen:
        return None

    parts = []
    for r in chosen:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lc = sr[r["index"]].download(quality_bitmask=quality_bitmask)
        except Exception as exc:                      # one bad file should not lose the target
            warnings.warn(f"TIC {tic_id} sector {r['sector']} ({r['author']}): download failed ({exc})")
            continue
        if lc is None:
            continue
        lc = lc.remove_nans()
        if len(lc) < 10:
            continue
        lc = lc.normalize()
        if sigma_upper:
            lc = lc.remove_outliers(sigma_lower=float("inf"), sigma_upper=sigma_upper)
        n = len(lc)
        parts.append((np.asarray(lc.time.value, float), np.asarray(lc.flux.value, float),
                      np.asarray(lc.flux_err.value, float), np.full(n, r["sector"]),
                      np.full(n, r["exptime"]), np.full(n, r["author"], dtype=object)))
    if not parts:
        return None
    cat = [np.concatenate([p[k] for p in parts]) for k in range(6)]
    o = np.argsort(cat[0])
    return TessLC(*[c[o] for c in cat])


def as_tesslc(lc):
    """Accept old-style (t, f, e) tuples too: treat them as one sector with unknown exposure."""
    if lc is None or isinstance(lc, TessLC):
        return lc
    t = np.asarray(lc[0], float); f = np.asarray(lc[1], float)
    e = np.asarray(lc[2], float) if len(lc) > 2 and lc[2] is not None else np.full(t.size, np.nan)
    return TessLC(t, f, e, np.zeros(t.size, int), np.full(t.size, 120.0), np.full(t.size, "unknown", dtype=object))
