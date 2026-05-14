from __future__ import annotations


import numpy as np
import pandas as pd

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from strategy import (
    build_strategy,
    psd_regime_mask,
    vol_regime_mask,
)
from backtest import run_backtest, performance_metrics, sharpe_diff_test


def cell_conditional_analysis(returns: pd.DataFrame,
                              psd_mask: pd.DataFrame,
                              vol_mask: pd.DataFrame,
                              cost_bps: float = 5.0,
                              start_date: str = "2010-01-01") -> pd.DataFrame:
    """Compute TSMOM performance restricted to each of the four regime cells.

    For each cell, we build a mask that is 1 only on (date, asset) pairs
    where BOTH the PSD condition and the vol condition are met, then apply
    that mask to the unconditional TSMOM position.

    Returns a DataFrame with one row per cell and metric columns.
    """
    # Cell definitions: (name, psd condition, vol condition)
    # psd_mask == 1 means PSD trending; vol_mask == 1 means vol low
    cells = {
        "A: trending & low-vol":     (psd_mask == 1) & (vol_mask == 1),
        "B: choppy  & low-vol":      (psd_mask == 0) & (vol_mask == 1),
        "C: trending & high-vol":    (psd_mask == 1) & (vol_mask == 0),
        "D: choppy  & high-vol":     (psd_mask == 0) & (vol_mask == 0),
    }

    rows = []
    for cell_name, cell_mask in cells.items():
        cell_mask_float = cell_mask.astype(float)
        positions = build_strategy(returns, regime_mask=cell_mask_float)
        r = run_backtest(positions, returns, cost_bps=cost_bps,
                         start_date=start_date)
        m = performance_metrics(r)
        m["cell"] = cell_name
        # Occupancy: fraction of (date, asset) cells falling in this bucket
        m["occupancy"] = cell_mask.mean().mean()
        rows.append(m)

    df = pd.DataFrame(rows).set_index("cell")
    cols = ["occupancy", "ann_return", "ann_vol", "sharpe",
            "max_drawdown", "calmar", "hit_rate_monthly"]
    return df[cols].round(4)


def combined_filter_strategy(returns: pd.DataFrame,
                             psd_mask: pd.DataFrame,
                             vol_mask: pd.DataFrame,
                             mode: str = "intersection",
                             cost_bps: float = 5.0,
                             start_date: str = "2010-01-01") -> pd.Series:
    """Build a strategy using both filters combined.

    mode='intersection' : trade only when PSD trending AND vol low (cell A)
    mode='union'        : trade when PSD trending OR vol low
    """
    if mode == "intersection":
        combined = ((psd_mask == 1) & (vol_mask == 1)).astype(float)
    elif mode == "union":
        combined = ((psd_mask == 1) | (vol_mask == 1)).astype(float)
    else:
        raise ValueError(f"unknown mode: {mode}")

    positions = build_strategy(returns, regime_mask=combined)
    return run_backtest(positions, returns, cost_bps=cost_bps,
                        start_date=start_date)


if __name__ == "__main__":
    print("Loading data...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    print("Building regime masks...")
    psd_mask = psd_regime_mask(rets)
    vol_mask = vol_regime_mask(rets)

    print("\n" + "=" * 70)
    print("CELL-CONDITIONAL ANALYSIS: where does momentum's edge live?")
    print("=" * 70)
    cell_df = cell_conditional_analysis(rets, psd_mask, vol_mask)
    print(cell_df.to_string())

    print("\n" + "=" * 70)
    print("COMBINED FILTER STRATEGIES")
    print("=" * 70)

    r_intersect = combined_filter_strategy(rets, psd_mask, vol_mask,
                                           mode="intersection")
    r_union = combined_filter_strategy(rets, psd_mask, vol_mask,
                                       mode="union")

    # For comparison, also rebuild the single-filter and unconditional strats
    r_uncond = run_backtest(build_strategy(rets), rets)
    r_vol_only = run_backtest(build_strategy(rets, regime_mask=vol_mask), rets)

    comparison = pd.DataFrame({
        "Unconditional":     performance_metrics(r_uncond),
        "Vol-filtered only": performance_metrics(r_vol_only),
        "Intersection (PSD & Vol)": performance_metrics(r_intersect),
        "Union (PSD | Vol)": performance_metrics(r_union),
    }).T.round(4)
    print(comparison.to_string())

    print("\nStatistical tests (Memmel):")
    print(" Intersection vs Vol-only:",
          sharpe_diff_test(r_intersect, r_vol_only))
    print(" Intersection vs Unconditional:",
          sharpe_diff_test(r_intersect, r_uncond))
    print(" Vol-only vs Unconditional:",
          sharpe_diff_test(r_vol_only, r_uncond))