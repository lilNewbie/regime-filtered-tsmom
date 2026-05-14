from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def welch_psd_ratio(returns_window: np.ndarray,
                    cutoff_period: float = 20.0,
                    nperseg: int = 60,
                    noverlap: int = 30,
                    fs: float = 1.0) -> float:
    """Compute log(low-band power / high-band power) for one window of returns.

    Parameters
    ----------
    returns_window : 1-D array of daily returns (length ~= nperseg or longer)
    cutoff_period : boundary between "low" and "high" freq bands, in days.
                    Frequencies below 1/cutoff_period are "low"; above are "high".
    nperseg       : length of each Welch segment (in days).
    noverlap      : overlap between segments (in days).
    fs            : sampling frequency. For daily data we use 1.0 (1 sample/day),
                    so frequencies are in cycles/day.

    Returns
    -------
    log(low_power / high_power). Returns np.nan if window has insufficient
    variance or the estimate is degenerate.
    """
    x = np.asarray(returns_window, dtype=float)
    if len(x) < nperseg or np.std(x) == 0 or np.any(~np.isfinite(x)):
        return np.nan

    # Detrending: removes linear drift within the window. Important because
    # a nonzero mean return would otherwise show up as zero-frequency power
    # and bias the low-band total.
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg,
                              noverlap=noverlap, detrend="linear",
                              scaling="density")

    # Drop the DC (zero-frequency) component entirely. After detrending it
    # should be ~0, but we exclude it to be safe.
    mask_positive = freqs > 0
    freqs = freqs[mask_positive]
    psd = psd[mask_positive]

    cutoff_freq = 1.0 / cutoff_period  # cycles/day
    low_mask = freqs < cutoff_freq
    high_mask = freqs >= cutoff_freq

    # Integrate power in each band. Using np.trapz would be more accurate
    # but a simple sum is fine given the coarse frequency grid from Welch.
    low_power = psd[low_mask].sum()
    high_power = psd[high_mask].sum()

    if low_power <= 0 or high_power <= 0:
        return np.nan

    return float(np.log(low_power / high_power))


def rolling_psd_feature(returns: pd.Series,
                        window: int = 60,
                        cutoff_period: float = 20.0,
                        nperseg: int = 60,
                        noverlap: int = 30) -> pd.Series:
    """Rolling PSD log-ratio feature for a single asset's return series.

    The feature at date t uses returns from [t-window+1, t] inclusive,
    so it is a backward-looking indicator — no lookahead bias.
    """
    def _compute(arr):
        return welch_psd_ratio(arr, cutoff_period=cutoff_period,
                               nperseg=nperseg, noverlap=noverlap)

    # pandas rolling.apply with raw=True passes numpy arrays (faster).
    feature = returns.rolling(window=window, min_periods=window).apply(
        _compute, raw=True
    )
    return feature.rename("psd_log_ratio")


def rolling_realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility (annualized). Used both for position sizing
    and as the baseline regime indicator."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def rolling_zscore(series: pd.Series, lookback: int = 252) -> pd.Series:
    """Rolling z-score using only past data (no lookahead).

    At date t, we use the mean and std of values from [t-lookback+1, t].
    This means the classifier is 'self-normalizing' over a year of history.
    """
    mean = series.rolling(window=lookback, min_periods=lookback // 2).mean()
    std = series.rolling(window=lookback, min_periods=lookback // 2).std()
    return (series - mean) / std


if __name__ == "__main__":
    # Quick sanity check against synthetic data with known properties.
    rng = np.random.default_rng(42)
    n = 2000

    # Case 1: white noise — PSD ratio should be near 0 (equal power).
    white = pd.Series(rng.standard_normal(n))
    feat_white = rolling_psd_feature(white).dropna()
    print(f"White noise PSD log-ratio: mean={feat_white.mean():.3f}, "
          f"std={feat_white.std():.3f}  (expected near 0)")

    # Case 2: AR(1) with positive autocorr — power shifts to low freq,
    # log-ratio should be positive.
    ar_pos = np.zeros(n)
    for i in range(1, n):
        ar_pos[i] = 0.3 * ar_pos[i-1] + rng.standard_normal()
    feat_ar_pos = rolling_psd_feature(pd.Series(ar_pos)).dropna()
    print(f"AR(1) +0.3   PSD log-ratio: mean={feat_ar_pos.mean():.3f}, "
          f"std={feat_ar_pos.std():.3f}  (expected > 0)")

    # Case 3: AR(1) with negative autocorr — power shifts to high freq.
    ar_neg = np.zeros(n)
    for i in range(1, n):
        ar_neg[i] = -0.3 * ar_neg[i-1] + rng.standard_normal()
    feat_ar_neg = rolling_psd_feature(pd.Series(ar_neg)).dropna()
    print(f"AR(1) -0.3   PSD log-ratio: mean={feat_ar_neg.mean():.3f}, "
          f"std={feat_ar_neg.std():.3f}  (expected < 0)")