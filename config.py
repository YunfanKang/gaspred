"""Configuration: regions, data-source series IDs, and release lags.

All EIA series are weekly and fetched through the v2 /seriesid/ path, which accepts
the full APIv1 id (e.g. ``PET.WGTSTUS1.W``). Series ids were verified against
eia.gov/dnav. FRED ids and yfinance tickers are standard.
"""

REGIONS = ["US", "PADD1", "PADD2", "PADD3", "PADD4", "PADD5"]

# EIA area codes embedded in series ids (NUS = U.S. total, R10..R50 = PADD 1..5)
_AREA = {"US": "NUS", "PADD1": "R10", "PADD2": "R20",
         "PADD3": "R30", "PADD4": "R40", "PADD5": "R50"}


def _pet(core: str) -> str:
    """Wrap an APIv1 core id as the full weekly id used by /v2/seriesid/."""
    return f"PET.{core}.W"


# --- Target: retail regular all-formulations price, weekly, $/gal (per region) ---
EIA_RETAIL = {r: _pet(f"EMM_EPMR_PTE_{_AREA[r]}_DPG") for r in REGIONS}

# --- Region feature: ending stocks of total gasoline, weekly, thousand barrels ---
_STOCKS_CORE = {"US": "WGTSTUS1", "PADD1": "WGTSTP11", "PADD2": "WGTSTP21",
                "PADD3": "WGTSTP31", "PADD4": "WGTSTP41", "PADD5": "WGTSTP51"}
EIA_GAS_STOCKS = {r: _pet(c) for r, c in _STOCKS_CORE.items()}

# --- Region feature: refinery % utilization of operable capacity, weekly, percent ---
_UTIL_CORE = {"US": "WPULEUS3",
              "PADD1": "W_NA_YUP_R10_PER", "PADD2": "W_NA_YUP_R20_PER",
              "PADD3": "W_NA_YUP_R30_PER", "PADD4": "W_NA_YUP_R40_PER",
              "PADD5": "W_NA_YUP_R50_PER"}
EIA_REFINERY_UTIL = {r: _pet(c) for r, c in _UTIL_CORE.items()}

# --- National demand proxy: product supplied of finished motor gasoline, weekly, kbbl/d ---
# Published U.S.-only weekly; broadcast to all regions as a national demand signal.
EIA_PRODUCT_SUPPLIED = _pet("WGFUPUS2")

# --- FRED series (daily) ---
FRED_WTI = "DCOILWTICO"        # WTI spot, $/bbl
FRED_BRENT = "DCOILBRENTEU"    # Brent spot, $/bbl
FRED_USD_INDEX = "DTWEXBGS"    # Nominal broad U.S. dollar index

# --- Futures front-month, continuous (yfinance) ---
YF_TICKERS = {"wti_front": "CL=F", "brent_front": "BZ=F", "rbob_front": "RB=F"}

# --- Release lags: days from an observation's period-end to public availability ---
# These drive the look-ahead-safe alignment (see align.to_weekly_asof).
LAG_WPSR_DAYS = 5     # WPSR data (stocks, refinery util, product supplied): Fri week-end -> Wed release
LAG_RETAIL_DAYS = 1   # retail price: Monday survey -> Mon/Tue release
LAG_DAILY_DAYS = 1    # daily market data (crude, USD, futures)

WEEK_ANCHOR = "W-FRI"        # weekly decision/label day for the panel
DEFAULT_START = "2015-01-01"
