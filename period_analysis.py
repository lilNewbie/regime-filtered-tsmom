from __future__ import annotations

import numpy as np
import pandas as pd

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from strategy import build_strategy, psd_regime_mask, vol_regime_mask
from backtest import run_backtest, performance_metrics
from combined import combined_filter_strategy


# Define the periods we'll analyze
PERIODS = [
    ("Full sample",            "2010-01-01", "2025-12-31"),
    ("In-sample 2010-2017",    "2010-01-01", "2017-12-31"),
    ("OOS 2018-2025",          "2018-01-01", "2025-12-31"),
    ("2010-2014",              "2010-01-01", "2014-12-31"),
    ("2015-2019",              "2015-01-01", "2019-12-31"),
    ("2020-2024",              "2020-01-01", "2024-12-31"),
]


def slice_returns(r: pd.Series, start: str, end: str) -> pd.Series:
    """Slice a return series to a date range, dropping NaNs."""
    return r.loc[start:end].dropna()


def slice_excluding_period(r: pd.Series,
                           exclude_start: str,
                           exclude_end: str) -> pd.Series:
    """Return a series with a specific period removed."""
    mask = (r.index < exclude_start) | (r.index > exclude_end)
    return r.loc[mask].dropna()


def metrics_for_series(r: pd.Series) -> dict:
    """Wrapper that handles short/empty series gracefully."""
    if len(r) < 30 or r.std() == 0:
        return {k: np.nan for k in
                ["ann_return", "ann_vol", "sharpe",
                 "max_drawdown", "calmar", "hit_rate_monthly"]}
    return performance_metrics(r)


def cell_sharpes_for_period(returns: pd.DataFrame,
                            psd_mask: pd.DataFrame,
                            vol_mask: pd.DataFrame,
                            start: str,
                            end: str,
                            cost_bps: float = 5.0) -> dict:
    """Compute the 4 cell Sharpes restricted to a date range.

    For each cell we build the cell mask, restrict returns to the period,
    then build and backtest the strategy.
    """
    cells = {
        "A: trend+lowvol":  (psd_mask == 1) & (vol_mask == 1),
        "B: chop+lowvol":   (psd_mask == 0) & (vol_mask == 1),
        "C: trend+highvol": (psd_mask == 1) & (vol_mask == 0),
        "D: chop+highvol":  (psd_mask == 0) & (vol_mask == 0),
    }
    out = {}
    for name, mask in cells.items():
        cell_mask = mask.astype(float)
        positions = build_strategy(returns, regime_mask=cell_mask)
        r = run_backtest(positions, returns, cost_bps=cost_bps,
                         start_date="2010-01-01")
        rs = slice_returns(r, start, end)
        m = metrics_for_series(rs)
        out[name] = m["sharpe"]
    return out


def main():
    print("Loading data and building regime masks...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    psd_mask = psd_regime_mask(rets)
    vol_mask = vol_regime_mask(rets)

    # Build all four strategies once (they're path-independent across periods)
    pos_uncond = build_strategy(rets)
    pos_psd = build_strategy(rets, regime_mask=psd_mask)
    pos_vol = build_strategy(rets, regime_mask=vol_mask)

    r_uncond = run_backtest(pos_uncond, rets, cost_bps=5.0)
    r_psd = run_backtest(pos_psd, rets, cost_bps=5.0)
    r_vol = run_backtest(pos_vol, rets, cost_bps=5.0)
    r_intersect = combined_filter_strategy(rets, psd_mask, vol_mask,
                                           mode="intersection")

    strategies = {
        "Unconditional":  r_uncond,
        "PSD-filtered":   r_psd,
        "Vol-filtered":   r_vol,
        "Intersection":   r_intersect,
    }

    # ---- Strategy Sharpes by period ---------------------------------------
    print("\n" + "=" * 78)
    print("STRATEGY SHARPE BY PERIOD")
    print("=" * 78)
    rows = []
    for label, start, end in PERIODS:
        row = {"period": label, "start": start, "end": end}
        for name, r in strategies.items():
            rs = slice_returns(r, start, end)
            m = metrics_for_series(rs)
            row[name] = m["sharpe"]
        row["n_days"] = len(slice_returns(r_uncond, start, end))
        rows.append(row)

    # Also add an excl-2020-to-2022 row
    row = {"period": "Excl. 2020-2022", "start": "—", "end": "—"}
    for name, r in strategies.items():
        rs = slice_excluding_period(r, "2020-01-01", "2022-12-31")
        rs = rs.loc["2010-01-01":]  # standard backtest start
        m = metrics_for_series(rs)
        row[name] = m["sharpe"]
    row["n_days"] = len(
        slice_excluding_period(r_uncond, "2020-01-01", "2022-12-31")
        .loc["2010-01-01":]
    )
    rows.append(row)

    sharpe_df = pd.DataFrame(rows).set_index("period")
    sharpe_df = sharpe_df[["start", "end", "Unconditional", "PSD-filtered",
                           "Vol-filtered", "Intersection", "n_days"]]
    print(sharpe_df.round(3).to_string())

    # ---- Full metrics for the intersection strategy by period -------------
    print("\n" + "=" * 78)
    print("INTERSECTION STRATEGY — FULL METRICS BY PERIOD")
    print("=" * 78)
    rows = []
    for label, start, end in PERIODS:
        rs = slice_returns(r_intersect, start, end)
        m = metrics_for_series(rs)
        m["period"] = label
        rows.append(m)
    # Excl 2020-2022
    rs = slice_excluding_period(r_intersect, "2020-01-01", "2022-12-31")
    rs = rs.loc["2010-01-01":]
    m = metrics_for_series(rs)
    m["period"] = "Excl. 2020-2022"
    rows.append(m)
    intersect_df = pd.DataFrame(rows).set_index("period")
    print(intersect_df.round(4).to_string())

    # ---- 2x2 cell Sharpes by period --------------------------------------
    print("\n" + "=" * 78)
    print("2x2 CELL SHARPES BY PERIOD")
    print("=" * 78)
    rows = []
    for label, start, end in PERIODS:
        cell = cell_sharpes_for_period(rets, psd_mask, vol_mask, start, end)
        cell["period"] = label
        rows.append(cell)
    cell_df = pd.DataFrame(rows).set_index("period")
    print(cell_df.round(3).to_string())

    # ---- Mask agreement by period -----------------------------------------
    print("\n" + "=" * 78)
    print("PSD vs VOL MASK AGREEMENT BY PERIOD")
    print("=" * 78)
    rows = []
    for label, start, end in PERIODS:
        p = psd_mask.loc[start:end]
        v = vol_mask.loc[start:end]
        if len(p) < 30:
            continue
        agreement = (p == v).mean().mean()
        rows.append({
            "period": label,
            "agreement": agreement,
            "psd_in_market_pct": p.mean().mean(),
            "vol_in_market_pct": v.mean().mean(),
            "n_days": len(p),
        })
    print(pd.DataFrame(rows).set_index("period").round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()