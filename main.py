from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from strategy import (
    build_strategy,
    psd_regime_mask,
    vol_regime_mask,
)
from backtest import (
    run_backtest,
    compare_strategies,
    sharpe_diff_test,
)
from spectral import rolling_psd_feature, rolling_zscore

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # ---- Load data ---------------------------------------------------------
    print("Loading price data...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)
    print(f"  {rets.shape[0]} days, {rets.shape[1]} assets")
    print(f"  Date range: {rets.index.min().date()} to {rets.index.max().date()}")

    # ---- Build regime masks ------------------------------------------------
    print("\nComputing PSD regime masks...")
    psd_mask = psd_regime_mask(rets)
    print("Computing vol regime masks...")
    vol_mask = vol_regime_mask(rets)

    # ---- Build strategies --------------------------------------------------
    print("\nBuilding strategies...")
    pos_uncond = build_strategy(rets)
    pos_psd = build_strategy(rets, regime_mask=psd_mask)
    pos_vol = build_strategy(rets, regime_mask=vol_mask)

    # ---- Backtest ----------------------------------------------------------
    print("Running backtests (start 2010-01-01, 5 bps costs)...")
    r_uncond = run_backtest(pos_uncond, rets, cost_bps=5.0)
    r_psd = run_backtest(pos_psd, rets, cost_bps=5.0)
    r_vol = run_backtest(pos_vol, rets, cost_bps=5.0)

    strategies = {
        "Unconditional TSMOM": r_uncond,
        "PSD-filtered TSMOM": r_psd,
        "Vol-filtered TSMOM": r_vol,
    }

    # ---- Metrics -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    metrics = compare_strategies(strategies)
    print(metrics.to_string())

    # ---- Sharpe difference tests ------------------------------------------
    print("\n" + "=" * 60)
    print("SHARPE DIFFERENCE TESTS (Memmel 2003)")
    print("=" * 60)
    print("\nPSD-filtered vs Unconditional:")
    print(sharpe_diff_test(r_psd, r_uncond))
    print("\nPSD-filtered vs Vol-filtered:")
    print(sharpe_diff_test(r_psd, r_vol))
    print("\nVol-filtered vs Unconditional:")
    print(sharpe_diff_test(r_vol, r_uncond))
    

    # ---- Sub-period breakdown ---------------------------------------------
    print("\n" + "=" * 60)
    print("SUB-PERIOD BREAKDOWN (Sharpe)")
    print("=" * 60)
    periods = [
        ("2010-2014", "2010-01-01", "2014-12-31"),
        ("2015-2019", "2015-01-01", "2019-12-31"),
        ("2020-2024", "2020-01-01", "2024-12-31"),
        ("Excl. 2022", None, None),  # special handling
    ]
    rows = []
    for label, start, end in periods:
        row = {"period": label}
        for name, r in strategies.items():
            if label == "Excl. 2022":
                rs = r[~((r.index >= "2022-01-01") & (r.index < "2023-01-01"))]
            else:
                rs = r.loc[start:end]
            if len(rs) > 30 and rs.std() > 0:
                row[name] = (rs.mean() * 252) / (rs.std() * np.sqrt(252))
            else:
                row[name] = np.nan
        rows.append(row)
    print(pd.DataFrame(rows).set_index("period").round(3).to_string())

    # ---- Diagnostic plots -------------------------------------------------
    print("\nGenerating plots...")

    # Equity curves
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in strategies.items():
        equity = (1 + r).cumprod()
        ax.plot(equity.index, equity.values, label=name, linewidth=1.3)
    ax.set_title("Equity curves (walk-forward, 5 bps transaction costs)")
    ax.set_ylabel("Cumulative return (×)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curves2.png", dpi=140)
    plt.close(fig)

    # PSD feature + regime overlay for SPY as illustration
    spy_feat = rolling_psd_feature(rets["SPY"])
    spy_z = rolling_zscore(spy_feat)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot((1 + rets["SPY"]).cumprod(), color="k", linewidth=1)
    # Shade choppy regime periods
    choppy = spy_z <= 0
    in_block = False
    start = None
    for t, is_choppy in choppy.items():
        if is_choppy and not in_block:
            start = t
            in_block = True
        elif not is_choppy and in_block:
            axes[0].axvspan(start, t, color="red", alpha=0.12)
            in_block = False
    if in_block:
        axes[0].axvspan(start, choppy.index[-1], color="red", alpha=0.12)
    axes[0].set_title("SPY price with shaded 'choppy' regimes (PSD z-score ≤ 0)")
    axes[0].set_ylabel("Price (×)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(spy_z.index, spy_z.values, color="steelblue", linewidth=1)
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_title("PSD log-ratio z-score (SPY)")
    axes[1].set_ylabel("z-score")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "spy_regime_overlay.png", dpi=140)
    plt.close(fig)

    # Agreement between PSD and vol regime masks (key diagnostic!)
    agreement = (psd_mask == vol_mask).mean().mean()
    # Per-asset correlation of the two mask time series
    per_asset_corr = {}
    for col in psd_mask.columns:
        a = psd_mask[col].reindex(vol_mask.index).dropna()
        b = vol_mask[col].reindex(a.index).dropna()
        a = a.loc[b.index]
        if a.std() > 0 and b.std() > 0:
            per_asset_corr[col] = a.corr(b)
    print("\n" + "=" * 60)
    print("REGIME FILTER OVERLAP DIAGNOSTIC")
    print("=" * 60)
    print(f"Fraction of (date, asset) cells where PSD mask == Vol mask: "
          f"{agreement:.2%}")
    print("Per-asset correlation of PSD mask and Vol mask:")
    for k, v in per_asset_corr.items():
        print(f"  {k}: {v:+.3f}")

    # Save metrics to CSV
    metrics.to_csv(OUTPUT_DIR / "metrics.csv")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    return strategies, metrics


if __name__ == "__main__":
    main()