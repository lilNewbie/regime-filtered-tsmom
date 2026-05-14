from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf


DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM",
                    "TLT", "IEF", "GLD", "DBC", "VNQ"]


def load_prices(tickers: list[str] = DEFAULT_UNIVERSE,
                start: str = "2007-01-01",
                end: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for the given universe.

    Returns a DataFrame with dates as index, tickers as columns.
    Drops rows where any ticker has missing data (inner-join behavior).
    """
    data = yf.download(tickers, start=start, end=end,
                       auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex; 'Close' is adjusted because auto_adjust=True
    prices = data["Close"].copy()
    prices = prices.dropna(how="any")
    if prices.empty:
        raise RuntimeError("No price data returned. Check tickers / network.")
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns. Log returns are more natural for spectral analysis
    because they are additive over time."""
    return np.log(prices / prices.shift(1)).dropna()


if __name__ == "__main__":
    px = load_prices()
    rets = compute_log_returns(px)
    print(f"Prices shape: {px.shape}")
    print(f"Returns shape: {rets.shape}")
    print(f"Date range: {rets.index.min()} to {rets.index.max()}")
    print(rets.tail())