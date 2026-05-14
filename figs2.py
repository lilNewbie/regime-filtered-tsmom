from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from spectral import rolling_psd_feature, rolling_zscore, rolling_realized_vol
from strategy import psd_regime_mask, vol_regime_mask, build_strategy
from backtest import run_backtest, performance_metrics


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def make_regime_scatter():
    """Scatter of PSD z-score vs vol z-score across all (date, asset) cells."""
    print("Loading data for regime scatter...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    # Recompute z-scores per asset (not just the binary masks)
    psd_z_all = []
    vol_z_all = []
    for col in rets.columns:
        feat = rolling_psd_feature(rets[col])
        zp = rolling_zscore(feat)
        rv = rolling_realized_vol(rets[col], window=20)
        zv = rolling_zscore(rv)
        # Trim warm-up and align
        df = pd.DataFrame({"zp": zp, "zv": zv}).dropna().loc["2010-01-01":]
        psd_z_all.extend(df["zp"].values)
        vol_z_all.extend(df["zv"].values)

    psd_z_all = np.array(psd_z_all)
    vol_z_all = np.array(vol_z_all)

    # Downsample for plotting speed if huge
    if len(psd_z_all) > 30000:
        idx = np.random.RandomState(0).choice(len(psd_z_all), 30000, replace=False)
        psd_z_plot = psd_z_all[idx]
        vol_z_plot = vol_z_all[idx]
    else:
        psd_z_plot, vol_z_plot = psd_z_all, vol_z_all

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(psd_z_plot, vol_z_plot, alpha=0.06, s=4, color="steelblue",
               rasterized=True)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel(r"$z^{\mathrm{psd}}_{i,t}$  (PSD regime)")
    ax.set_ylabel(r"$z^{\mathrm{vol}}_{i,t}$  (Vol regime)")
    ax.set_title("Joint distribution of regime indicators")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    # Quadrant labels (note: low-vol = z_vol <= 0, so cell A is top-left
    # of "favorable" but bottom-right when we plot z_vol on y-axis)
    ax.text(2.5, -2.5, "Cell A\n(trend $\\cap$ low-vol)",
            ha="center", va="center", fontsize=9, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="grey", alpha=0.8))
    ax.text(-2.5, -2.5, "Cell B\n(chop $\\cap$ low-vol)",
            ha="center", va="center", fontsize=9, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.6))
    ax.text(2.5, 2.5, "Cell C\n(trend $\\cap$ high-vol)",
            ha="center", va="center", fontsize=9, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.6))
    ax.text(-2.5, 2.5, "Cell D\n(chop $\\cap$ high-vol)",
            ha="center", va="center", fontsize=9, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.6))

    # Correlation in the corner
    corr = np.corrcoef(psd_z_all, vol_z_all)[0, 1]
    ax.text(0.02, 0.98, f"Pearson $\\rho$ = {corr:+.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    fig.tight_layout()
    out = FIG_DIR / "regime_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_cell_bar_chart():
    """Bar chart of cell Sharpes."""
    print("Loading data for cell bar chart...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    psd_mask = psd_regime_mask(rets)
    vol_mask = vol_regime_mask(rets)

    cells = {
        "A:\ntrend $\\cap$ low-vol":  (psd_mask == 1) & (vol_mask == 1),
        "B:\nchop $\\cap$ low-vol":   (psd_mask == 0) & (vol_mask == 1),
        "C:\ntrend $\\cap$ high-vol": (psd_mask == 1) & (vol_mask == 0),
        "D:\nchop $\\cap$ high-vol":  (psd_mask == 0) & (vol_mask == 0),
    }

    cell_sharpes = {}
    for name, mask in cells.items():
        positions = build_strategy(rets, regime_mask=mask.astype(float))
        r = run_backtest(positions, rets, cost_bps=5.0,
                         start_date="2010-01-01")
        m = performance_metrics(r)
        cell_sharpes[name] = m["sharpe"]

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(cell_sharpes.keys())
    sharpes = list(cell_sharpes.values())
    colors = ["#1f7a4d" if s == max(sharpes) else "#888"
              if s >= 0 else "#a8423a" for s in sharpes]
    bars = ax.bar(names, sharpes, color=colors, edgecolor="black",
                  linewidth=0.7)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Sharpe by regime cell (unconditional TSMOM restricted to cell)")
    ax.grid(axis="y", alpha=0.3)

    for bar, sharpe in zip(bars, sharpes):
        height = bar.get_height()
        offset = 0.02 if height >= 0 else -0.04
        ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                f"{sharpe:+.2f}", ha="center", va="bottom" if height >= 0 else "top",
                fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = FIG_DIR / "cell_sharpes_bars.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import pandas as pd  # local import to keep top of file clean
    globals()["pd"] = pd  # used inside make_regime_scatter
    make_regime_scatter()
    make_cell_bar_chart()