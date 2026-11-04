"""Diagnostics on DR4 epoch astrometry (pre-release sample or release data).

    python examples/dr4_astrometry_sample.py tests/private/dr4_rc3_epoch_astrometry_12src.xml

Writes private_figures/dr4_epoch_astrometry_sample.png (gitignored: do not
publish figures made from pre-release data).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ebatlas.dr4_astrometry import (read_epoch_astrometry, al_observations, fit_five_parameter_al,
                                    astrometric_periodogram, orbit_grid_fit)

tab = read_epoch_astrometry(sys.argv[1])
obs = al_observations(tab, include_excess_noise=False)
sids = np.unique(obs["source_id"])
gmag = {s: np.nanmedian(np.asarray(tab["g_mag"][tab["source_id"] == s], float)) for s in sids}
uwe = {s: fit_five_parameter_al(obs[obs["source_id"] == s])["uwe"] for s in sids}
worst = max(uwe, key=uwe.get)
o = obs[obs["source_id"] == worst]

periods = np.exp(np.linspace(np.log(5), np.log(4000), 4000))
dchi, c5 = astrometric_periodogram(o, periods)
best = orbit_grid_fit(o, np.exp(np.linspace(np.log(200), np.log(4000), 120)))
f5 = fit_five_parameter_al(o, clip=1e9)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
g = np.array([gmag[s] for s in sids]); u = np.array([uwe[s] for s in sids])
ax[0].scatter(g, u, c=["#d1495b" if s == worst else "#1f5fbf" for s in sids], edgecolor="k", s=45)
ax[0].axhline(1.4, ls="--", color="0.5", lw=0.8); ax[0].text(g.min(), 1.55, "RUWE-style 1.4 cut", fontsize=8, color="0.4")
ax[0].set_yscale("log"); ax[0].set_xlabel("G [mag]"); ax[0].set_ylabel("UWE, 5-param fit (no excess noise)")
ax[0].set_title(f"{len(sids)} sources, DR4_RC3 epoch astrometry")

ax[1].plot(periods, dchi / c5, color="k", lw=0.8)
ax[1].set_xscale("log"); ax[1].set_xlabel("Trial period [d]"); ax[1].set_ylabel(r"$\Delta\chi^2/\chi^2_{5p}$ (circular orbit)")
ax[1].axvspan(0.5 * 365.25 * 5.5, 4000, color="0.9", zorder=0); ax[1].text(1100, 0.05, "P > ~half\nbaseline", fontsize=8, color="0.4")
ax[1].set_title(f"Astrometric periodogram, {worst}", fontsize=10)

tid = np.asarray(o["transit_id"]); un = np.unique(tid)
tm = np.array([o["t_years"][tid == k].mean() for k in un]) + 2016.0
# Data and orbit term, both relative to the 5-param part of the JOINT fit.
A5 = np.column_stack([np.sin(o["psi"]), np.cos(o["psi"]), o["f_al"],
                      np.sin(o["psi"]) * o["t_years"], np.cos(o["psi"]) * o["t_years"]])
base5 = A5 @ best["params"][:5]
data_orb = np.asarray(o["w"]) - base5
orb = best["model"] - base5
rm = np.array([data_orb[tid == k].mean() for k in un]); mm = np.array([orb[tid == k].mean() for k in un])
ax[2].plot(tm, rm, "o", ms=4, color="#1f5fbf", label="AL data minus single-star part")
ax[2].plot(tm, mm, "x", ms=5, color="#d1495b", label=f"orbit term: P≈{best['period']:.0f} d, e≈{best['e']:.1f}")
ax[2].set_xlabel("Year"); ax[2].set_ylabel("Per-transit mean AL [mas]"); ax[2].legend(fontsize=8)
ax[2].set_title(f"UWE 5p = {uwe[worst]:.0f} → orbit grid = {best['uwe']:.1f}", fontsize=10)

out = Path("private_figures"); out.mkdir(exist_ok=True)
fig.savefig(out / "dr4_epoch_astrometry_sample.png", dpi=130)
print("wrote", out / "dr4_epoch_astrometry_sample.png")
