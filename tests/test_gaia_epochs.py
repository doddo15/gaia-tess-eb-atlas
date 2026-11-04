import numpy as np
from astropy.table import Table

from ebatlas.gaia_epochs import standardize_dr3, STANDARD_COLUMNS


def test_long_format():
    tab = Table({"source_id": [1] * 4, "band": ["G", "G", "BP", "RP"],
                 "time": [1700.0, 1700.07, 1700.0, np.nan], "flux": [1e5, 0.9e5, 5e4, 6e4],
                 "flux_error": [10.0] * 4, "mag": [15.0] * 4,
                 "rejected_by_photometry": [False, False, True, False],
                 "rejected_by_variability": [False, True, False, False]})
    out = standardize_dr3(tab)
    assert out.colnames == STANDARD_COLUMNS + ["rejected_by_variability"]
    assert len(out) == 3                       # NaN time dropped
    assert list(out["rejected"]) == [False, False, True]               # photometry rejection only
    assert list(out["rejected_by_variability"]) == [False, True, False]  # kept, flagged separately
