"""Release-day dress rehearsal, fully offline.

    python examples/release_day_rehearsal.py data/TEBC.csv

Real TEBC ephemerides; synthetic DR4 epoch photometry written to disk in the
DR4 data-model format; synthetic TESS light curves. Runs the same code path
as release day from `read_epoch_photometry` onwards. One system is given an
eccentric orbit to exercise the secondary-phase measurement.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io.votable import from_table, writeto

from ebatlas import Ephemeris
from ebatlas.atlas import build_pages
from ebatlas.catalogs import load_catalog
from ebatlas.dr4_photometry import read_epoch_photometry
from ebatlas.planning import phase_sigma_table
from ebatlas.synthetic import make_dr4_epoch_photometry, eb_model

cat = load_catalog(sys.argv[1] if len(sys.argv) > 1 else "data/TEBC.csv", "tebc")
cat = cat[cat["has_ephemeris"]]
sig = phase_sigma_table(cat)
good = ((cat["morph_coeff"] < 0.3) & (sig["sigma_phase_start"] < 0.002)
        & (cat["period"] > 1.5) & (cat["period"] < 12) & (cat["tmag"] < 11))
sel = cat[good]
sel.sort("tmag")
sel = sel[[0, len(sel) // 2, len(sel) - 1]]

root = Path(__file__).resolve().parents[1] / "docs" / "release_rehearsal"
(root / "fake_datalink").mkdir(parents=True, exist_ok=True)
truth, ids = {}, {}
for k, r in enumerate(sel):
    P = float(r["period"])
    w1 = float(np.clip(0.035 * (2.0 / P) ** (2 / 3), 0.008, 0.08))
    phi2 = 0.42 if k == 1 else 0.5
    eph = Ephemeris(P, float(r["t0_btjd"]), float(r["period_err"]), float(r["t0_err"]),
                    primary_width=w1, secondary_phase=phi2, secondary_width=w1 * 0.9)
    sid = 9_000_000_000_000_000 + k
    raw = make_dr4_epoch_photometry(eph, sid, g_mag=float(r["tmag"]) + 0.4, seed=10 + k,
                                    depths={"g": (0.35, 0.12), "bp": (0.38, 0.10), "rp": (0.32, 0.14)},
                                    n_scans=38)
    path = root / "fake_datalink" / f"EPOCH_PHOTOMETRY-{sid}.xml"
    writeto(from_table(raw), str(path))
    truth[int(r["tic_id"])] = eph
    ids[r["eb_id"]] = sid

epochs = {}
for p in sorted((root / "fake_datalink").glob("*.xml")):
    epochs.update(read_epoch_photometry(p))


def fake_tess(tic, rng=np.random.default_rng(1)):
    eph = truth[tic]
    t = np.r_[np.arange(eph.t0 - 13, eph.t0 + 13, 2 / 1440)]
    f = eb_model(eph, t, 0.33, 0.11) * (1 + rng.normal(0, 0.0015, t.size))
    return t, f, np.full(t.size, 0.0015)


summary = build_pages(sel, ids, epochs, root, lc_getter=fake_tess, release="DR4")
cols = ["eb_id", "meas_primary_width", "meas_secondary_phase", "n_primary_secure",
        "n_secondary_secure", "primary_gaia_outlier_transits", "primary_depth_snr"]
summary[cols].pprint(max_width=200)
for r in sel:
    e = truth[int(r["tic_id"])]
    print(f"truth {r['eb_id']}: w1={e.primary_width:.4f} phi2={e.secondary_phase}")
