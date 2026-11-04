"""Offline demo: synthetic EB, Gaia-like sampling, TESS-like light curve.

    python examples/synthetic_demo.py

Produces docs/synthetic_demo.png and prints the atlas summary row.
No network needed, so this is the quickest way to check an install.
"""
from pathlib import Path

import numpy as np
from astropy.table import Table

from ebatlas import Ephemeris, btjd_to_gaia_time, gaia_time_to_btjd
from ebatlas.ephemeris import DR4_START_GAIA, DR4_END_GAIA
from ebatlas.plotting import atlas_page, summarize

rng = np.random.default_rng(4)
eph = Ephemeris(period=2.718282, t0=1500.0, period_err=3e-6, t0_err=2e-4,
                primary_width=0.045, secondary_phase=0.5, secondary_width=0.045)


def model(btjd, d1, d2):
    ph = eph.phase(btjd)
    f = 1.0 - 0.008 * np.cos(4 * np.pi * ph)            # ellipsoidal-ish
    for c, d, w in ((0.0, d1, eph.primary_width), (0.5, d2, eph.secondary_width)):
        x = np.abs(((ph - c + 0.5) % 1.0) - 0.5) / (w / 2)
        f -= d * np.clip(1 - x**2, 0, None) ** 0.6
    return f


# Gaia-like: ~36 scans over the DR4 window, two FoV transits 106.5 min apart.
# Real sampling should come from the scanning law (e.g. GOST) per sky position.
scans = np.sort(rng.uniform(DR4_START_GAIA, DR4_END_GAIA, 36))
t_gaia = np.sort(np.r_[scans, scans + 106.5 / 1440.0])
bt = gaia_time_to_btjd(t_gaia)

rows = []
for band, d1, d2, sig in (("G", 0.35, 0.12, 0.003), ("BP", 0.38, 0.10, 0.008), ("RP", 0.32, 0.14, 0.008)):
    f = model(bt, d1, d2) * (1 + rng.normal(0, sig, bt.size))
    for ti, tb, fi in zip(t_gaia, bt, f):
        rows.append((band, ti, tb, fi * 1e5, sig * 1e5, np.nan, False))
gaia = Table(rows=rows, names=("band", "time_gaia", "btjd", "flux", "flux_err", "mag", "rejected"))

# Mimic the pitfall: Gaia's outlier rejection discarding the deepest in-eclipse G point.
g = np.where(gaia["band"] == "G")[0]
gaia["rejected"][g[np.argmin(gaia["flux"][g])]] = True

# TESS-like: two sectors of 2-min cadence around BTJD 1500 and 2230.
tt = np.r_[np.arange(1490, 1517, 2 / 1440), np.arange(2220, 2247, 2 / 1440)]
tt = tt[(np.mod(tt - 1490, 27) < 12.5) | (np.mod(tt - 1490, 27) > 14.0)]   # orbit gap
tf = model(tt, 0.30, 0.11) * (1 + rng.normal(0, 0.002, tt.size))

out = Path(__file__).resolve().parents[1] / "docs" / "synthetic_demo.png"
atlas_page(eph, gaia, tess=(tt, tf), title="Synthetic EB (demo)", out=out)
print(summarize(eph, gaia))
print("wrote", out)
