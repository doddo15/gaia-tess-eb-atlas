import numpy as np
import pytest
from astropy.table import Table

from ebatlas import datalink
from ebatlas.gaia_epochs import parse_dr3_result

# The exact message the archive returned on 2026-11-04 (release 3.15).
DEPRECATED = ("Error 500:\nThe 'INDIVIDUAL' data structure has been deprecated - use 'DATAMODEL_STANDARD' instead\n\n"
              "Cannot process request: 'https://gea.esac.esa.int/data-server/data'")


def test_error_classification():
    assert datalink.is_permanent(Exception(DEPRECATED)) and datalink.is_structure_error(Exception(DEPRECATED))
    assert not datalink.is_permanent(Exception("Error 503: Service Unavailable"))
    assert not datalink.is_permanent(Exception("HTTPSConnectionPool: Read timed out."))


def test_structure_fallback_and_memory(monkeypatch):
    calls = []

    class FakeGaia:
        @staticmethod
        def load_data(ids, data_structure, **kw):
            calls.append(data_structure)
            if data_structure == "DATAMODEL_GAIA":
                raise Exception("Error 500: The 'DATAMODEL_GAIA' data structure is not supported for this product")
            return {"f.xml": [Table({"source_id": [1]})]}

    import astroquery.gaia
    monkeypatch.setattr(astroquery.gaia, "Gaia", FakeGaia)
    datalink._WORKING.clear()
    with pytest.warns(UserWarning):
        _, used = datalink.load([1], "EPOCH_PHOTOMETRY", "Gaia DR3", pause=0)
    assert used == "DATAMODEL_STANDARD" and calls == ["DATAMODEL_GAIA", "DATAMODEL_STANDARD"]
    datalink.load([1], "EPOCH_PHOTOMETRY", "Gaia DR3", pause=0)
    assert calls[-1] == "DATAMODEL_STANDARD"                     # remembered: no second refusal


def test_no_retry_on_permanent(monkeypatch):
    n = {"calls": 0}

    class FakeGaia:
        @staticmethod
        def load_data(**kw):
            n["calls"] += 1
            raise Exception("Error 400: invalid ID list")

    import astroquery.gaia
    monkeypatch.setattr(astroquery.gaia, "Gaia", FakeGaia)
    datalink._WORKING.clear()
    with pytest.raises(Exception):
        datalink.load([1], "EPOCH_PHOTOMETRY", "Gaia DR3", structures=("DATAMODEL_GAIA",), pause=0)
    assert n["calls"] == 1


def test_parse_long_with_aliases_multi_source():
    t = Table({"SOURCE_ID": [1, 1, 2, 2], "passband": ["G", "BP", "G", "RP"], "obs_time": [1700.0, 1700.0, 1800.0, 1800.0],
               "flux": [1e5, 5e4, 2e5, 9e4], "flux_err": [10.0] * 4, "magnitude": [15.0] * 4,
               "rejected_by_photometry": ["false", "true", "false", "false"],
               "rejected_by_variability": [True, False, False, False]})
    out = parse_dr3_result({"x.xml": [t]})
    assert sorted(out) == [1, 2]
    assert list(out[1]["band"]) == ["G", "BP"] and list(out[1]["rejected"]) == [False, True]
    assert bool(out[1]["rejected_by_variability"][0])


def test_parse_wide_multi_source():
    rows = []
    for sid in (10, 20):
        rows.append({"source_id": sid, "g_transit_time": np.array([1700.0, 1700.1]), "g_transit_flux": np.array([1e5, 9e4]),
                     "g_transit_flux_error": np.array([10.0, 10.0]), "g_transit_mag": np.array([15.0, 15.1]),
                     "bp_obs_time": np.array([1700.0, 1700.1]), "bp_flux": np.array([5e4, np.nan]),
                     "variability_flag_g_reject": np.array([False, True])})
    t = Table(rows=rows)
    out = parse_dr3_result({"y.xml": [t]})
    assert sorted(out) == [10, 20]
    g = out[10][out[10]["band"] == "G"]
    assert len(g) == 2 and list(g["rejected_by_variability"]) == [False, True]
    bp = out[10][out[10]["band"] == "BP"]
    assert list(bp["rejected"]) == [False, True]                   # NaN flux -> rejected


def test_unparsed_table_is_saved(tmp_path):
    with pytest.raises(RuntimeError, match="inspect_datalink"):
        parse_dr3_result({"weird.xml": [Table({"source_id": [1], "foo": [2.0]})]}, unparsed_dir=tmp_path)
    assert (tmp_path / "weird.ecsv").exists()
