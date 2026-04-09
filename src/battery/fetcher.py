"""
Market data fetcher for CAISO and ERCOT day-ahead prices.

Uses the gridstatus library, which provides a clean unified interface
to US ISO market data. This removes the pain of dealing with each ISO's
bespoke API format.

Install: pip install gridstatus

CAISO (NP15 hub):   Northern California pricing node. Most liquid.
ERCOT (HB_HOUSTON): Houston Hub. Representative of ERCOT West/South.

Real data is non-negotiable for this project. Every LP you run should
be on real prices — not synthetic data. The distribution of real prices
(fat tails, negative prices, price spikes) is the whole point.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime
from typing import Optional


# ── Cache paths ──────────────────────────────────────────────────────────────
# Raw data lives in data/raw/, processed lives in data/processed/.
# Cache to avoid re-downloading the same data repeatedly.
DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def fetch_caiso_prices(
    start: str,
    end: str,
    node: str = "TH_NP15_GEN-APND", #NoCal
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch CAISO day-ahead LMP prices for a given node and date range.

    Args:
        start: Start date as "YYYY-MM-DD"
        end:   End date as "YYYY-MM-DD"
        node:  CAISO pricing node. Default is NP15 (North California).
               Other options: TH_SP15_GEN-APND (South CA), TH_ZP26_GEN-APND (Central)
        use_cache: If True, load from local parquet file if it exists.

    Returns:
        pd.Series of hourly LMP prices ($/MWh), indexed by datetime.

    Notes:
        LMP = Locational Marginal Price. It has three components:
        LMP = Energy Component + Congestion Component + Loss Component
        The energy component is the systemwide marginal cost.
        Congestion component can be positive or negative depending on
        whether the node is import-constrained or export-constrained.
    """
    cache_path = PROCESSED_DIR / f"caiso_{node}_{start}_{end}.parquet"

    if use_cache and cache_path.exists():
        print(f"Loading CAISO data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    try:
        import gridstatus
        iso = gridstatus.CAISO()

        # gridstatus returns a DataFrame; we want just the LMP column
        # as a Series indexed by interval_start
        df = iso.get_lmp(
            start=start,
            end=end,
            market="DAY_AHEAD_HOURLY",
            locations=[node],
        )

        df.set_index("Time", inplace=True)
        prices = df["LMP"].resample('h').mean()
        prices.to_pickle(cache_path)
        return prices

    except ImportError:
        raise ImportError(
            "Install gridstatus: pip install gridstatus\n"
            "Then re-run this function."
        )


def fetch_ercot_prices(
    start: str,
    end: str,
    hub: str = "HB_HOUSTON",
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch ERCOT day-ahead settlement point prices.

    Args:
        start: Start date as "YYYY-MM-DD"
        end:   End date as "YYYY-MM-DD"
        hub:   ERCOT hub. Options: HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST
        use_cache: Cache locally as parquet.

    Returns:
        pd.Series of hourly prices ($/MWh), indexed by datetime.

    Notes:
        ERCOT uses "Settlement Point Prices" not LMP (ERCOT has no congestion
        pricing at the hub level — hubs are synthetic aggregates).
        The real ERCOT action is at "Load Zones" and "Resource Nodes".
        For this project, HB_HOUSTON is a good representative.
    """
    cache_path = PROCESSED_DIR / f"ercot_{hub}_{start}_{end}.parquet"

    if use_cache and cache_path.exists():
        print(f"Loading ERCOT data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    try:
        import gridstatus
        iso = gridstatus.Ercot()

        df = iso.get_lmp(
            start=start,
            end=end,
            market="DAY_AHEAD_HOURLY",
            locations=[hub],
        )

        df.set_index("Time", inplace=True)
        prices = df["LMP"].resample("h").mean()
        prices.to_pickle(cache_path)
        return prices

    except ImportError:
        raise ImportError("Install gridstatus: pip install gridstatus")


def load_sample_prices(market: str = "ercot") -> pd.Series:
    """
    Load a small sample of synthetic prices for testing before real data is set up.

    This lets you test the LP formulation immediately without waiting
    for API access. The synthetic prices have a realistic daily shape:
    low overnight, high morning and evening peaks, very high midday 
    (for testing negative price behavior).

    Args:
        market: "ercot" or "caiso" — affects price scale/volatility.

    Returns:
        pd.Series of 24 hourly prices ($/MWh).
    """
    # A stylized "duck curve" day in Texas (summer)
    # Note: this includes a brief negative price period at midday (solar glut)
    if market == "ercot":
        prices = np.array([
            25, 22, 20, 18, 17, 18,   # Hours 0-5 (overnight, low load)
            35, 55, 70, 65, 45, -5,   # Hours 6-11 (morning ramp, solar starts)
            -10, -8, 5, 20, 45, 85,   # Hours 12-17 (midday glut, evening ramp)
            120, 150, 95, 65, 45, 30, # Hours 18-23 (peak, then decline)
        ], dtype=float)
    else:  # caiso
        prices = np.array([
            30, 27, 25, 23, 22, 24,
            40, 60, 75, 70, 50, 10,
            -15, -12, 5, 25, 50, 90,
            130, 160, 100, 70, 50, 35,
        ], dtype=float)

    index = pd.date_range("2025-01-15 00:00", periods=24, freq="h")
    return pd.Series(prices, index=index, name=f"{market}_prices")


def compute_price_statistics(prices: pd.Series) -> dict:
    """
    Compute summary statistics relevant to battery optimization.

    The statistics that matter most for a battery operator:
    - Daily spread (max - min price per day): determines arbitrage opportunity
    - Spike frequency (hours above $200/MWh): ancillary service / peak discharge value
    - Negative price frequency: cheap charging opportunity
    - Volatility: relevant for stochastic optimization
    """
    daily = prices.resample('D')

    return {
        'mean_price': prices.mean(),
        'std_price': prices.std(),
        'min_price': prices.min(),
        'max_price': prices.max(),
        'pct_negative': (prices < 0).mean() * 100,
        'pct_above_100': (prices > 100).mean() * 100,
        'pct_above_200': (prices > 200).mean() * 100,
        'daily_spread_mean': (daily.max() - daily.min()).mean(),
        'daily_spread_p90': (daily.max() - daily.min()).quantile(0.9),
    }
