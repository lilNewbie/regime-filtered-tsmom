from pathlib import Path
import matplotlib.pyplot as plt

from data import load_prices, compute_log_returns, DEFAULT_UNIVERSE
from strategy import build_strategy, psd_regime_mask, vol_regime_mask
from backtest import run_backtest
from combined import combined_filter_strategy


def main():
    print("Loading data...")
    px = load_prices(tickers=DEFAULT_UNIVERSE, start="2007-01-01")
    rets = compute_log_returns(px)

    print("Building regime masks...")
    psd_mask = psd_regime_mask(rets)
    vol_mask = vol_regime_mask(rets)

    print("Building strategies...")
    r_uncond = run_backtest(build_strategy(rets), rets, cost_bps=5.0)
    r_psd = run_backtest(build_strategy(rets, regime_mask=psd_mask),
                         rets, cost_bps=5.0)
    r_vol = run_backtest(build_strategy(rets, regime_mask=vol_mask),
                         rets, cost_bps=5.0)
    r_int = combined_filter_strategy(rets, psd_mask, vol_mask,
                                     mode="intersection", cost_bps=5.0)

    strategies = {
        "Unconditional TSMOM":   (r_uncond, "#888888",  "-",  1.3),
        "PSD-filtered TSMOM":    (r_psd,    "#d97a3a",  "-",  1.3),
        "Vol-filtered TSMOM":    (r_vol,    "#3a7ad9",  "-",  1.3),
        "Intersection-filtered": (r_int,    "#1f7a4d",  "-",  1.8),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, (r, color, ls, lw) in strategies.items():
        equity = (1 + r).cumprod()
        ax.plot(equity.index, equity.values,
                label=name, color=color, linestyle=ls, linewidth=lw)

    ax.set_title("Cumulative returns (net of 5 bps transaction costs)")
    ax.set_ylabel(r"Growth of \$1")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    figures_dir = Path(__file__).parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    out_report = figures_dir / "equity_curves.png"
    out_console = output_dir / "equity_curves.png"
    fig.savefig(out_report, dpi=300, bbox_inches="tight")
    fig.savefig(out_console, dpi=140)
    plt.close(fig)
    print(f"Saved {out_report}")
    print(f"Saved {out_console}")


if __name__ == "__main__":
    main()