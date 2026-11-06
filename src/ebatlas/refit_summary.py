"""Summary of a catalog refit (reads the CSV written by refit_catalog)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table


def _b(col):
    return np.array([str(v).strip().lower() in ("true", "1") for v in col])


def summarize_refit(path, figure=True):
    t = Table.read(path, format="ascii.csv")
    n = len(t)
    st = {str(k): int(v) for k, v in zip(*np.unique(np.asarray(t["status"]).astype(str), return_counts=True))}
    print(f"\n{n} EBs: {st}")
    ok = np.asarray(t["status"]).astype(str) == "ok"
    if not ok.any():
        return t
    f = lambda c: np.asarray(t[c][ok], float)
    print(f"  refit ok: median {np.median(f('n_sectors')):.0f} sectors timed, baseline "
          f"{np.median(f('baseline_days')) / 365.25:.1f} yr (90th pct {np.percentile(f('baseline_days'), 90) / 365.25:.1f})")
    print(f"  timing jitter: median {np.nanmedian(f('jitter_s')):.0f} s; zero jitter for {np.mean(f('jitter_s') == 0):.0%}")
    for flag in ("flag_catalog_superseded", "flag_period_change", "flag_variable_depth", "flag_catalog_far_off"):
        print(f"  {flag[5:]:20s} {np.sum(_b(t[flag][ok])):5d}  ({np.mean(_b(t[flag][ok])):.0%})")
    d = np.abs(f("drift_dr4_mid")); dl = np.abs(f("drift_last_sector"))
    print(f"  catalog vs refit, |phase| at DR4 mid-epoch: median {np.nanmedian(d):.4f}; >0.01 for {np.mean(d > 0.01):.0%}, "
          f">0.05 for {np.mean(d > 0.05):.0%}")
    print(f"  catalog vs refit, |phase| at the last timed sector: median {np.nanmedian(dl):.4f}; >0.01 for {np.mean(dl > 0.01):.0%}")
    err = np.asarray(t["status"]).astype(str) == "error"
    if err.any():
        msgs = {}
        for m in np.asarray(t["error"][err]).astype(str):
            msgs[m.split("|")[0][:80]] = msgs.get(m.split("|")[0][:80], 0) + 1
        print("  errors:", dict(sorted(msgs.items(), key=lambda kv: -kv[1])[:5]))
    if figure:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
        ss = f("period_shift_sigma")
        ax[0].hist(np.clip(ss[np.isfinite(ss)], -30, 30), bins=60, color="#1f5fbf")
        ax[0].set_xlabel("(P_refit − P_catalog) / σ, clipped at ±30"); ax[0].set_title("Catalog periods vs refit", fontsize=10)
        ax[1].hist(np.log10(np.clip(d[np.isfinite(d)], 1e-5, 0.5)), bins=50, color="#d1495b")
        ax[1].axvline(-2, color="0.4", ls=":"); ax[1].set_xlabel("log10 |Δphase| at DR4 mid-epoch")
        ax[1].set_title("How far catalog ephemerides drift by DR4", fontsize=10)
        ax[2].scatter(f("baseline_days") / 365.25, np.abs(f("quad_snr")), s=6, color="k", alpha=0.6)
        ax[2].axhline(5, color="#d1495b", ls="--", lw=0.8); ax[2].set_yscale("log")
        ax[2].set_xlabel("timed baseline [yr]"); ax[2].set_ylabel("|quadratic term| / σ")
        ax[2].set_title("Period-change significance", fontsize=10)
        out = Path(path).with_name(Path(path).stem + "_summary.png")
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"  wrote {out}")
    return t
