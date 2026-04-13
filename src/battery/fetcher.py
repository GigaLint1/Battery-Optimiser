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

        # Fetch in monthly chunks to avoid API gaps on large ranges
        chunks = []
        chunk_start = pd.Timestamp(start)
        chunk_end = pd.Timestamp(end)

        while chunk_start < chunk_end:
            next_month = (chunk_start + pd.offsets.MonthBegin(1)).normalize()
            this_end = min(next_month, chunk_end)
            print(f"  Fetching CAISO {chunk_start.date()} to {this_end.date()}...")

            df = iso.get_lmp(
                start=str(chunk_start.date()),
                end=str(this_end.date()),
                market="DAY_AHEAD_HOURLY",
                locations=[node],
            )
            chunks.append(df)
            chunk_start = this_end

        df = pd.concat(chunks, ignore_index=True)
        df.set_index("Time", inplace=True)
        prices = df["LMP"].resample('h').mean()

        # Strip timezone so reindex matches
        prices.index = prices.index.tz_localize(None)

        # Drop duplicate timestamps (DST transitions cause repeated hours)
        prices = prices[~prices.index.duplicated(keep='first')]

        # Reindex to a complete hourly grid to expose any gaps
        full_index = pd.date_range(start, end, freq='h', inclusive='left')
        prices = prices.reindex(full_index)

        n_missing = prices.isna().sum()
        if n_missing > 0:
            print(f"  Warning: {n_missing} hours missing out of {len(full_index)}")

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
            date=start,
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

import glob
import os

def process_ercot_data(folder_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load all CSVs in a folder into a single combined DataFrame.

    Args:
        folder_path: Path to folder containing CSVs
        **read_csv_kwargs: Any keyword args to pass to pd.read_csv
                           (e.g. parse_dates=['date'], dtype={'col': str})
    
    Returns:
        Combined DataFrame, sorted by any date column if present
    """

    # Find CSV files
    pattern = os.path.join(folder_path, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    # Load each into a list (memory-efficient accumulation)
    frames = []
    for file in files:
        df = pd.read_csv(file, **read_csv_kwargs)
        df["_source_file"] = os.path.basename(file)  # optional: track origin
        frames.append(df)

    # Concatenate once
    combined = pd.concat(frames, ignore_index=True)

    combined = combined[combined['BusName'] == 'ADICKS_345B']

    combined["Date"] = pd.to_datetime(combined["DeliveryDate"]) + pd.to_timedelta(combined["HourEnding"].str.split(":").str[0].astype(int) - 1, unit='h')
    combined.drop(combined.columns[[0,1,2,4,5]],inplace=True, axis=1)
    combined.set_index("Date", inplace=True)
    
    return combined