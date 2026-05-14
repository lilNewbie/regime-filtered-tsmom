from __future__ import annotations

import numpy as np
import pandas as pd

from spectral import (
    rolling_psd_feature,
    rolling_realized_vol,
    rolling_zscore,
)


# ---- Signals ---------------------------------------------------------------

def tsmom_signal(returns: pd.DataFrame,
                 long_lookback: int = 252,
                 short_lookback: int = 21) -> pd.DataFrame:
    """Standard TSMOM signal: sign of (12m return - 1m return).

    The long-minus-short construction is the Moskowitz-Ooi-Pedersen variant
    that removes the most recent month to avoid short-term reversal effects
    that contaminate pure 12m momentum.
    """
    log_prices = np.log((1 + returns).cumprod())  # equivalent up to a constant
    long_ret = log_prices - log_prices.shift(long_lookback)
    short_ret = log_prices - log_prices.shift(short_lookback)
    momentum = long_ret - short_ret
    # Sign gives {-1, 0, +1}; we use np.sign which returns 0 at exactly 0.
    return np.sign(momentum).fillna(0)


# ---- Position sizing -------------------------------------------------------

def vol_scaled_positions(signal: pd.DataFrame,
                         returns: pd.DataFrame,
                         vol_lookback: int = 60,
                         target_vol: float = 0.10) -> pd.DataFrame:
    """Scale each asset's position so ex-ante contribution is ~target_vol.

    Position size = signal * (target_vol / realized_vol). This is the
    standard TSMOM normalization — it prevents high-vol assets from
    dominating and gives each asset an equal risk slot.
    """
    vol = returns.rolling(window=vol_lookback).std() * np.sqrt(252)
    # Avoid division by zero in early/degenerate rows
    vol = vol.replace(0, np.nan)
    positions = signal * (target_vol / vol)
    return positions.fillna(0)


# ---- Regime indicators -----------------------------------------------------

def psd_regime_mask(returns: pd.DataFrame,
                    psd_window: int = 60,
                    cutoff_period: float = 20.0,
                    zscore_lookback: int = 252) -> pd.DataFrame:
    """Per-asset regime mask based on PSD log-ratio z-score.

    Returns 1.0 when asset is in 'trending' regime (z > 0), else 0.0.
    Mask is applied multiplicatively to the position.
    """
    masks = {}
    for col in returns.columns:
        feat = rolling_psd_feature(returns[col],
                                   window=psd_window,
                                   cutoff_period=cutoff_period,
                                   nperseg=psd_window,
                                   noverlap=psd_window // 2)
        z = rolling_zscore(feat, lookback=zscore_lookback)
        # NaN z-scores (warm-up period) -> 0 mask (no trading)
        masks[col] = (z > 0).astype(float).where(z.notna(), 0.0)
    return pd.DataFrame(masks)


def vol_regime_mask(returns: pd.DataFrame,
                    vol_window: int = 20,
                    zscore_lookback: int = 252) -> pd.DataFrame:
    """Per-asset regime mask based on realized vol z-score.

    Returns 1.0 when asset is in 'low-vol' regime (z <= 0), else 0.0.
    This is the baseline filter we compare the PSD filter against.

    Note the inversion vs. PSD: for the vol filter we trade in LOW-vol
    periods, so the mask is 1 when z is LOW.
    """
    masks = {}
    for col in returns.columns:
        rv = rolling_realized_vol(returns[col], window=vol_window)
        z = rolling_zscore(rv, lookback=zscore_lookback)
        masks[col] = (z <= 0).astype(float).where(z.notna(), 0.0)
    return pd.DataFrame(masks)


# ---- Rebalancing ----------------------------------------------------------

def to_monthly_positions(daily_positions: pd.DataFrame) -> pd.DataFrame:
    """Hold position constant within a month. Rebalance on the last trading
    day of each month using that day's signal.

    We take the last value of each month, then forward-fill over the next
    month's trading days. This is a standard monthly-rebalance TSMOM setup.
    """
    # Month-end positions
    month_ends = daily_positions.resample("ME").last()
    # Shift by 1 day so the position taken at month-end t is held starting
    # the next trading day — prevents using t+1 information at t.
    monthly = month_ends.reindex(daily_positions.index, method="ffill")
    monthly = monthly.shift(1)  # lag so today's position was set yesterday
    return monthly.fillna(0)


# ---- Full strategy construction -------------------------------------------

def build_strategy(returns: pd.DataFrame,
                   regime_mask: pd.DataFrame | None = None,
                   target_vol: float = 0.10) -> pd.DataFrame:
    """Construct monthly-rebalanced TSMOM positions, optionally regime-filtered.

    Parameters
    ----------
    returns      : daily log returns, one column per asset
    regime_mask  : daily 0/1 mask, same shape as returns. If None, no filter.

    Returns
    -------
    DataFrame of daily positions (size per asset, in units of $ notional
    per $1 of capital allocated to that asset slot).
    """
    sig = tsmom_signal(returns)
    pos = vol_scaled_positions(sig, returns, target_vol=target_vol)

    if regime_mask is not None:
        # Align on index/columns
        mask = regime_mask.reindex(index=pos.index, columns=pos.columns).fillna(0)
        pos = pos * mask

    monthly_pos = to_monthly_positions(pos)
    return monthly_pos


if __name__ == "__main__":
    # Quick integration test
    from data import load_prices, compute_log_returns
    px = load_prices(start="2015-01-01")
    rets = compute_log_returns(px)

    pos_uncond = build_strategy(rets)
    psd_mask = psd_regime_mask(rets)
    pos_psd = build_strategy(rets, regime_mask=psd_mask)
    vol_mask = vol_regime_mask(rets)
    pos_vol = build_strategy(rets, regime_mask=vol_mask)

    print("Unconditional final positions:")
    print(pos_uncond.iloc[-1])
    print(f"\nPSD-filtered fraction of time in market: "
          f"{(psd_mask.sum(axis=1) > 0).mean():.2%}")
    print(f"Vol-filtered fraction of time in market: "
          f"{(vol_mask.sum(axis=1) > 0).mean():.2%}")