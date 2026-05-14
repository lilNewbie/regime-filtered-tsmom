from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from strategy import build_strategy, psd_regime_mask, vol_regime_mask
from backtest import run_backtest
from combined import combined_filter_strategy


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def rolling_sharpe(returns, window=252):
    """Annualized rolling Sharpe."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean * 252) / (std * np.sqrt(252))


def main():
    print("Loading data and building strategies...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    psd_mask = psd_regime_mask(rets)
    vol_mask = vol_regime_mask(rets)

    r_uncond = run_backtest(build_strategy(rets), rets, cost_bps=5.0)
    r_psd = run_backtest(build_strategy(rets, regime_mask=psd_mask),
                         rets, cost_bps=5.0)
    r_vol = run_backtest(build_strategy(rets, regime_mask=vol_mask),
                         rets, cost_bps=5.0)
    r_int = combined_filter_strategy(rets, psd_mask, vol_mask,
                                     mode="intersection", cost_bps=5.0)

    print("Computing rolling Sharpes...")
    strategies = {
        "Unconditional":   (rolling_sharpe(r_uncond), "#888888", 1.2),
        "PSD-filtered":    (rolling_sharpe(r_psd),    "#d97a3a", 1.2),
        "Vol-filtered":    (rolling_sharpe(r_vol),    "#3a7ad9", 1.2),
        "Intersection":    (rolling_sharpe(r_int),    "#1f7a4d", 1.8),
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (rs, color, lw) in strategies.items():
        rs_clean = rs.dropna()
        ax.plot(rs_clean.index, rs_clean.values,
                label=name, color=color, linewidth=lw, alpha=0.85)

    ax.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax.set_title("Rolling 252-day annualized Sharpe ratio")
    ax.set_ylabel("Sharpe (annualized)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "rolling_sharpe.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()