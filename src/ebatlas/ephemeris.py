"""Time-system conversion and ephemeris propagation.

Conventions
-----------
* Gaia epoch times ("Gaia time") are BJD in TCB minus 2455197.5 days.
* TESS times (BTJD) are BJD in TDB minus 2457000 days.
* TCB - TDB is ~19 s during the Gaia mission. That is negligible for
  phase-folding most EBs but NOT for eclipse-timing work, so we convert
  properly rather than just subtracting offsets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.time import Time

GAIA_TIME_OFFSET = 2455197.5   # Gaia time = BJD_TCB - 2455197.5
BTJD_OFFSET = 2457000.0        # BTJD      = BJD_TDB - 2457000

# Approximate DR4 (66-month) observing window, in Gaia time.
DR4_START_GAIA = 2456863.5 - GAIA_TIME_OFFSET   # 2014-07-25
DR4_END_GAIA = 2458868.5 - GAIA_TIME_OFFSET     # 2020-01-20


def gaia_time_to_btjd(t_gaia):
    """Convert Gaia time (BJD_TCB - 2455197.5) to BTJD (BJD_TDB - 2457000)."""
    t = np.atleast_1d(np.asarray(t_gaia, dtype=float))
    tdb = Time(np.full_like(t, GAIA_TIME_OFFSET), t, format="jd", scale="tcb").tdb
    return (tdb.jd1 - BTJD_OFFSET) + tdb.jd2


def btjd_to_gaia_time(btjd):
    """Inverse of :func:`gaia_time_to_btjd`."""
    t = np.atleast_1d(np.asarray(btjd, dtype=float))
    tcb = Time(np.full_like(t, BTJD_OFFSET), t, format="jd", scale="tdb").tcb
    return (tcb.jd1 - GAIA_TIME_OFFSET) + tcb.jd2


def _wrap(x):
    """Wrap phase-like quantity into [-0.5, 0.5)."""
    return np.mod(np.asarray(x) + 0.5, 1.0) - 0.5


@dataclass
class Ephemeris:
    """Linear ephemeris T(E) = t0 + E * period, with t0 in BTJD.

    Eclipse widths and the secondary phase are in phase units (0-1).
    Widths are optional; without them, in-eclipse flags are unavailable.
    """
    period: float
    t0: float
    period_err: float = 0.0
    t0_err: float = 0.0
    primary_width: Optional[float] = None
    secondary_phase: float = 0.5
    secondary_width: Optional[float] = None

    def cycle(self, btjd):
        return (np.asarray(btjd, dtype=float) - self.t0) / self.period

    def phase(self, btjd):
        """Orbital phase in [-0.5, 0.5), primary eclipse at 0."""
        return _wrap(self.cycle(btjd))

    def phase_sigma(self, btjd):
        """1-sigma phase uncertainty from propagating t0 and period errors.

        Ignores the t0-period covariance (most catalogs don't publish it);
        choosing t0 near the middle of the TESS baseline minimises it.
        """
        E = self.cycle(btjd)
        return np.sqrt(self.t0_err**2 + (E * self.period_err) ** 2) / self.period

    def _eclipse_mask(self, btjd, center, width, mode):
        if width is None:
            raise ValueError("Eclipse width not set for this ephemeris.")
        d = np.abs(_wrap(self.phase(btjd) - center))
        sig = self.phase_sigma(btjd)
        half = width / 2.0
        if mode == "nominal":
            return d < half
        if mode == "secure":       # in eclipse even allowing for 2-sigma drift
            return d < half - 2 * sig
        if mode == "possible":     # could be in eclipse within 2-sigma
            return d < half + 2 * sig
        raise ValueError(f"Unknown mode {mode!r}")

    def in_primary(self, btjd, mode="nominal"):
        return self._eclipse_mask(btjd, 0.0, self.primary_width, mode)

    def in_secondary(self, btjd, mode="nominal"):
        return self._eclipse_mask(btjd, self.secondary_phase, self.secondary_width, mode)

    def is_reliable_at(self, btjd, max_sigma_frac=0.25):
        """True where phase uncertainty < max_sigma_frac * primary width.

        Short-period EBs propagated back thousands of cycles from TESS to
        2014-2019 can lose phase coherence; flag those before trusting a fold.
        """
        ref = self.primary_width if self.primary_width else 0.05
        return self.phase_sigma(btjd) < max_sigma_frac * ref
