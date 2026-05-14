import numpy as np
import pandas as pd

import strategy
import backtest
import spectral


def make_synthetic_prices(n_days=4000, n_assets=10, seed=7):
    """Generate synthetic prices with time-varying autocorrelation.

    Every ~250 days we flip the AR coefficient between +0.1 (trending) and
    -0.1 (mean-reverting). This gives our regime filter something real to
    detect, so we can sanity-check whether the filter helps performance.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2007-01-01", periods=n_days)
    tickers = [f"AST{i:02d}" for i in range(n_assets)]

    rets = np.zeros((n_days, n_assets))
    # Per-asset regime phase offset so they're not all synchronized
    phase = rng.integers(0, 250, size=n_assets)

    for a in range(n_assets):
        ar = 0.0
        prev = 0.0
        for t in range(n_days):
            # Flip AR every 250 days, offset by phase
            phase_in_cycle = (t + phase[a]) % 500
            ar = 0.15 if phase_in_cycle < 250 else -0.10
            shock = rng.standard_normal() * 0.012  # ~19% ann vol
            prev = ar * prev + shock
            rets[t, a] = prev

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


if __name__ == "__main__":
    print("Generating synthetic price data with regime switching...")
    prices = make_synthetic_prices()
    rets = np.log(prices / prices.shift(1)).dropna()
    print(f"  {rets.shape[0]} days, {rets.shape[1]} assets")

    print("\nBuilding regime masks and strategies...")
    psd_mask = strategy.psd_regime_mask(rets)
    vol_mask = strategy.vol_regime_mask(rets)

    pos_uncond = strategy.build_strategy(rets)
    pos_psd = strategy.build_strategy(rets, regime_mask=psd_mask)
    pos_vol = strategy.build_strategy(rets, regime_mask=vol_mask)

    print("Running backtests...")
    r_uncond = backtest.run_backtest(pos_uncond, rets, cost_bps=5.0,
                                     start_date="2010-01-01")
    r_psd = backtest.run_backtest(pos_psd, rets, cost_bps=5.0,
                                  start_date="2010-01-01")
    r_vol = backtest.run_backtest(pos_vol, rets, cost_bps=5.0,
                                  start_date="2010-01-01")

    strategies = {
        "Unconditional": r_uncond,
        "PSD-filtered": r_psd,
        "Vol-filtered": r_vol,
    }

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA RESULTS (regime-switching simulation)")
    print("=" * 60)
    print(backtest.compare_strategies(strategies).to_string())

    print("\nSharpe diff tests:")
    print(" PSD vs Unconditional:",
          backtest.sharpe_diff_test(r_psd, r_uncond))
    print(" PSD vs Vol-filtered :",
          backtest.sharpe_diff_test(r_psd, r_vol))