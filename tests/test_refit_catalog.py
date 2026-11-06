import numpy as np
from astropy.table import Table

from ebatlas.catalogs import apply_refit
from ebatlas.refit_catalog import refit_catalog, done_ids
from ebatlas.refit_summary import summarize_refit
from ebatlas.tess import save_compact, load_compact, TessLC


def _lc_factory():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("th", pathlib.Path(__file__).parent / "test_refit_hardening.py")
    th = importlib.util.module_from_spec(spec); spec.loader.exec_module(th)
    return th


def _cat(th):
    return Table({"eb_id": ["TIC1.01", "TIC2.01", "TIC3.01"], "tic_id": [1, 2, 3],
                  "period": [th.P + 3e-6, th.P, th.P], "period_err": [1e-6] * 3, "t0_btjd": [th.T0] * 3,
                  "t0_err": [1e-4] * 3, "primary_width": [np.nan] * 3, "secondary_phase": [np.nan] * 3,
                  "secondary_width": [np.nan] * 3, "has_ephemeris": [True] * 3})


def test_refit_catalog_resume_and_errors(tmp_path):
    th = _lc_factory()
    lcs = {1: th._lc([th._sector(s) for s in (1, 27, 54, 80)]), 2: None}

    def getter(tic):
        if tic == 3:
            raise RuntimeError("simulated MAST failure")
        return lcs[tic]

    out = tmp_path / "refit.csv"
    cat = _cat(th)
    refit_catalog(cat[:2], out, lc_getter=getter)
    assert done_ids(out) == {"TIC1.01", "TIC2.01"}
    refit_catalog(cat, out, lc_getter=getter)                      # resumes: only TIC3 is new
    t = Table.read(out, format="ascii.csv")
    assert list(t["eb_id"]) == ["TIC1.01", "TIC2.01", "TIC3.01"]
    assert list(t["status"]) == ["ok", "no_tess", "error"]
    assert "simulated MAST failure" in str(t["error"][2])
    assert abs(t["refit_period"][0] - th.P) < 5e-7 and str(t["flag_catalog_superseded"][0]) == "True"
    assert (tmp_path / "refit_timings.csv").exists() and (tmp_path / "refit_sectors.csv").exists()
    summarize_refit(out, figure=True)
    assert (tmp_path / "refit_summary.png").exists()

    new = apply_refit(cat, t)
    assert list(new["ephem_source"]) == ["refit", "catalog", "catalog"]
    assert abs(new["period"][0] - th.P) < 5e-7 and new["period"][1] == th.P
    assert np.isfinite(new["primary_width"][0])


def test_compact_cache_roundtrip(tmp_path):
    n = 1000
    lc = TessLC(np.linspace(1500, 1527, n), np.ones(n), np.full(n, 1e-3), np.full(n, 14),
                np.full(n, 120.0), np.array(["SPOC"] * 500 + ["QLP"] * 500, dtype=object))
    save_compact(lc, tmp_path / "TIC1.npz")
    back = load_compact(tmp_path / "TIC1.npz")
    assert np.allclose(back.t, lc.t, rtol=0, atol=1e-9) and list(back.author[[0, -1]]) == ["SPOC", "QLP"]
