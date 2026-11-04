"""How precisely do TEBC ephemerides land on Gaia DR4 epochs?

    python examples/tebc_ephemeris_precision.py path/to/TEBC.csv

Writes docs/tebc_phase_precision.png and prints summary fractions.
Needs only the catalog file; no network access.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ebatlas.catalogs import load_catalog
from ebatlas.planning import phase_sigma_table

path = sys.argv[1] if len(sys.argv) > 1 else "TEBC.csv"
cat = load_catalog(path, "tebc")
cat = cat[cat["has_ephemeris"]]
tab = phase_sigma_table(cat)

print(f"{len(cat)} TEBC EBs with a usable ephemeris")
for name in ("start", "mid", "end"):
    s = np.asarray(tab[f"sigma_phase_{name}"])
    print(f"DR4 {name:5s}: median sigma_phase {np.median(s):.4f} | "
          f"<0.005 {np.mean(s < 0.005):4.0%} | <0.02 {np.mean(s < 0.02):4.0%} | >0.25 (lost) {np.mean(s > 0.25):4.0%}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
s0 = np.asarray(tab["sigma_phase_start"])
sc = a1.scatter(cat["period"], s0, c=cat["morph_coeff"], s=5, cmap="viridis", vmin=0, vmax=1)
fig.colorbar(sc, ax=a1, label="TEBC morphology coefficient")
a1.set_xscale("log"); a1.set_yscale("log")
for y, lab in ((0.005, "fine"), (0.02, "smeared"), (0.25, "lost")):
    a1.axhline(y, color="0.4", lw=0.8, ls="--")
    a1.text(0.05, y * 1.15, lab, fontsize=8, color="0.3")
a1.set_xlabel("Period [d]"); a1.set_ylabel(r"$\sigma_\phi$ at DR4 start (2014.6)")
a1.set_title("Phase uncertainty after propagating TEBC ephemerides back to 2014")

bins = np.logspace(-6, 1, 400)
for name, col in (("start", "#d1495b"), ("mid", "#edae49"), ("end", "#1f5fbf")):
    s = np.sort(np.asarray(tab[f"sigma_phase_{name}"]))
    a2.plot(s, np.arange(1, len(s) + 1) / len(s), color=col, label=f"DR4 {name}")
for y in (0.005, 0.02, 0.25):
    a2.axvline(y, color="0.4", lw=0.8, ls="--")
a2.set_xscale("log"); a2.set_xlim(1e-5, 10)
a2.set_xlabel(r"$\sigma_\phi$"); a2.set_ylabel("Cumulative fraction of TEBC")
a2.legend(); a2.set_title(f"{len(cat)} EBs (Prša et al. 2022)")

out = Path(__file__).resolve().parents[1] / "docs" / "tebc_phase_precision.png"
fig.savefig(out, dpi=130)
print("wrote", out)
