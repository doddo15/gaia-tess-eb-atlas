"""ebatlas: a Gaia DR4 x TESS eclipsing-binary atlas toolkit."""
from .ephemeris import Ephemeris, gaia_time_to_btjd, btjd_to_gaia_time

__all__ = ["Ephemeris", "gaia_time_to_btjd", "btjd_to_gaia_time"]
__version__ = "0.1.0.dev0"
