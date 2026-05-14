from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from spectral import rolling_psd_feature, rolling_zscore, rolling_realized_vol
from strategy import build_strategy, vol_regime_mask
from backtest import run_backtest, performance_metrics


def psd_regime_mask_custom(returns: pd.DataFrame,
                           psd_window: int,
                           cutoff_period: float,
                           zscore_lookback: int = 252) -> pd.DataFrame:
    """Same as strategy.psd_regime_mask, but with overridable parameters."""
    masks = {}
    for col in returns.columns:
        feat = rolling_psd_feature(returns[col],
                                   window=psd_window,
                                   cutoff_period=cutoff_period,
                                   nperseg=psd_window,
                                   noverlap=psd_window // 2)
        z = rolling_zscore(feat, lookback=zscore_lookback)
        masks[col] = (z > 0).astype(float).where(z.notna(), 0.0)
    return pd.DataFrame(masks)


def run_one_config(returns: pd.DataFrame,
                   vol_mask: pd.DataFrame,
                   psd_window: int,
                   cutoff_period: float,
                   cost_bps: float = 5.0,
                   start_date: str = "2010-01-01") -> dict:
    """Run one parameter configuration. Returns metrics for intersection
    strategy plus baselines."""
    psd_mask = psd_regime_mask_custom(returns, psd_window, cutoff_period)

    # Intersection
    int_mask = ((psd_mask == 1) & (vol_mask == 1)).astype(float)
    pos_int = build_strategy(returns, regime_mask=int_mask)
    r_int = run_backtest(pos_int, returns, cost_bps=cost_bps,
                         start_date=start_date)

    # PSD-only
    pos_psd = build_strategy(returns, regime_mask=psd_mask)
    r_psd = run_backtest(pos_psd, returns, cost_bps=cost_bps,
                         start_date=start_date)

    m_int = performance_metrics(r_int)
    m_psd = performance_metrics(r_psd)

    return {
        "intersection_sharpe": m_int["sharpe"],
        "intersection_calmar": m_int["calmar"],
        "intersection_max_dd": m_int["max_drawdown"],
        "psd_only_sharpe":    m_psd["sharpe"],
    }


def main():
    print("Loading data...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    print("Building (fixed) vol mask...")
    vol_mask = vol_regime_mask(rets)

    # Establish baselines (don't depend on PSD params)
    pos_uncond = build_strategy(rets)
    r_uncond = run_backtest(pos_uncond, rets, cost_bps=5.0)
    pos_vol = build_strategy(rets, regime_mask=vol_mask)
    r_vol = run_backtest(pos_vol, rets, cost_bps=5.0)
    uncond_sharpe = performance_metrics(r_uncond)["sharpe"]
    vol_sharpe = performance_metrics(r_vol)["sharpe"]

    print(f"\nBaselines (independent of PSD params):")
    print(f"  Unconditional Sharpe: {uncond_sharpe:.3f}")
    print(f"  Vol-filtered Sharpe:  {vol_sharpe:.3f}")

    # Grid
    psd_windows = [40, 60, 90, 120]
    cutoff_periods = [15, 20, 25, 30]

    print(f"\nRunning sensitivity grid: "
          f"{len(psd_windows)} x {len(cutoff_periods)} = "
          f"{len(psd_windows) * len(cutoff_periods)} configs...")

    rows = []
    for w in psd_windows:
        for c in cutoff_periods:
            print(f"  window={w}, cutoff={c}...", end=" ", flush=True)
            r = run_one_config(rets, vol_mask, psd_window=w,
                               cutoff_period=c)
            r["psd_window"] = w
            r["cutoff_period"] = c
            rows.append(r)
            print(f"intersection Sharpe = {r['intersection_sharpe']:.3f}")

    df = pd.DataFrame(rows)

    # Pivot tables for readability
    print("\n" + "=" * 70)
    print("INTERSECTION STRATEGY SHARPE — sensitivity grid")
    print("=" * 70)
    pivot_int = df.pivot(index="psd_window", columns="cutoff_period",
                         values="intersection_sharpe")
    print(pivot_int.round(3).to_string())

    print("\n" + "=" * 70)
    print("PSD-ONLY STRATEGY SHARPE — sensitivity grid (for context)")
    print("=" * 70)
    pivot_psd = df.pivot(index="psd_window", columns="cutoff_period",
                        values="psd_only_sharpe")
    print(pivot_psd.round(3).to_string())

    print("\n" + "=" * 70)
    print("INTERSECTION MAX DRAWDOWN — sensitivity grid")
    print("=" * 70)
    pivot_dd = df.pivot(index="psd_window", columns="cutoff_period",
                       values="intersection_max_dd")
    print(pivot_dd.round(4).to_string())

    # Summary stats
    int_sharpes = df["intersection_sharpe"]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Intersection Sharpe across grid:")
    print(f"  Min:    {int_sharpes.min():.3f}")
    print(f"  Max:    {int_sharpes.max():.3f}")
    print(f"  Median: {int_sharpes.median():.3f}")
    print(f"  All above vol-only baseline ({vol_sharpe:.3f})? "
          f"{(int_sharpes > vol_sharpe).all()}")
    print(f"  All above unconditional baseline ({uncond_sharpe:.3f})? "
          f"{(int_sharpes > uncond_sharpe).all()}")

    # Save to CSV for the appendix
    df.round(4).to_csv("output/parameter_sensitivity.csv", index=False)
    print("\nSaved output/parameter_sensitivity.csv")


if __name__ == "__main__":
    main()