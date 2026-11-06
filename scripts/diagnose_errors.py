"""Where does the Gaia-vs-TESS misfit come from? (reads a verification.csv)

    python scripts/diagnose_errors.py verify_dr3_refit/verification.csv [--noise-curve]

--noise-curve also queries ~50k DR3 constant stars once (cached in
data/dr3_noise_curve.ecsv) to compare Gaia's formal per-transit errors with the
empirical noise at each source's magnitude. Writes <dir>/error_diagnostics.png.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("verification"); ap.add_argument("--noise-curve", action="store_true")
ap.add_argument("--curve-file", default="data/dr3_noise_curve.ecsv")
a = ap.parse_args()

res = Table.read(a.verification)
truthy = lambda c: np.array([str(v).strip().lower() in ("true", "1") for v in c])   # CSV stores "True"/"False" text
if "diag_chi2_red_in" not in res.colnames:
    raise SystemExit("No diag_* columns: rerun scripts/03_verify_dr3.py with the current code first.")
con = truthy(res["offset_constrained"])
r = res[con]
col = lambda c: np.asarray(r[c], float)
cin, coot = col("diag_chi2_red_in"), col("diag_chi2_red_oot")
cin_t, coot_t = col("diag_chi2_red_in_tnoise"), col("diag_chi2_red_oot_tnoise")
ratio, g, rel = col("diag_oot_scatter_ratio"), col("diag_g_mag"), col("diag_rel_err_formal")
contact = truthy(r["meas_contact_like"])

print(f"{len(r)} constrained fits")
print(f"  median chi2_red  in eclipse {np.nanmedian(cin):7.1f}   out of eclipse {np.nanmedian(coot):7.1f}")
print(f"  ... with TESS template noise added: in {np.nanmedian(cin_t):7.1f}   out {np.nanmedian(coot_t):7.1f}")
print(f"  median template-noise / Gaia-error in eclipse: {np.nanmedian(col('diag_tnoise_over_err_in')):.2f}")
print(f"  out-of-eclipse scatter / formal error: median {np.nanmedian(ratio):.2f} "
      f"(contact {np.nanmedian(ratio[contact]):.2f}, detached {np.nanmedian(ratio[~contact]):.2f})")
share_in = np.nansum(cin * col("diag_n_in")) / np.nansum(cin * col("diag_n_in") + coot * col("diag_n_oot"))
print(f"  share of total chi2 from in-eclipse points: {share_in:.0%} "
      f"(they are {np.nansum(col('diag_n_in')) / np.nansum(col('diag_n_in') + col('diag_n_oot')):.0%} of the points)")

curve = None
if a.noise_curve:
    from ebatlas.noise import fetch_constant_stars, noise_curve, interp_curve
    cf = Path(a.curve_file)
    if cf.exists():
        curve = Table.read(cf)
    else:
        curve = noise_curve(fetch_constant_stars()); cf.parent.mkdir(parents=True, exist_ok=True); curve.write(cf)
    print(f"  noise curve covers G = {curve['g'].min():.2f}-{curve['g'].max():.2f}")
    inside = (g >= curve["g"].min()) & (g <= curve["g"].max())
    print(f"  targets inside the curve's magnitude range: {inside.sum()} of {len(g)}")
    lo, up = interp_curve(curve, g)
    lo, up, rel_c = lo[inside], up[inside], rel[inside]
    print(f"  formal per-transit error vs constant-star noise at same G: "
          f"below lower bound for {np.mean(rel_c < lo):.0%}, below upper bound for {np.mean(rel_c < up):.0%}")
    print(f"  empirical/formal from curve (median): {np.nanmedian(lo / rel_c):.2f} - {np.nanmedian(up / rel_c):.2f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
for m, c, lab in ((contact, "#d1495b", "contact-like"), (~contact, "#1f5fbf", "detached")):
    ax[0].loglog(coot[m], cin[m], "o", color=c, ms=5, label=lab)
lim = [0.3, max(np.nanmax(cin), np.nanmax(coot), 10) * 2]
ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel(r"$\chi^2_{\rm red}$ out of eclipse"); ax[0].set_ylabel(r"$\chi^2_{\rm red}$ in eclipse")
ax[0].legend(fontsize=8); ax[0].set_title("Where the misfit is", fontsize=10)
ax[1].semilogy(g[contact], ratio[contact], "o", color="#d1495b", ms=5)
ax[1].semilogy(g[~contact], ratio[~contact], "o", color="#1f5fbf", ms=5)
ax[1].axhline(1, color="k", ls="--", lw=0.8)
ax[1].set_xlabel("G [mag]"); ax[1].set_ylabel("out-of-eclipse scatter / formal error")
ax[1].set_title("Are Gaia's per-transit errors realistic?", fontsize=10)
ax[2].semilogy(g, rel, "o", color="k", ms=4, label="formal (median per source)")
if curve is not None:
    ax[2].fill_between(curve["g"], curve["lower"], curve["upper"], color="#9ec5e8", alpha=0.6,
                       label="constant stars (DR3 bounds)")
ax[2].set_xlabel("G [mag]"); ax[2].set_ylabel("relative per-transit G error"); ax[2].legend(fontsize=8)
ax[2].set_title("Formal errors vs. constant-star noise", fontsize=10)
out = Path(a.verification).parent / "error_diagnostics.png"
fig.savefig(out, dpi=130)
print("wrote", out)
