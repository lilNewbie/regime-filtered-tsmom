from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data import load_prices, compute_log_returns
from spectral import rolling_psd_feature, rolling_zscore


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    print("Loading SPY data...")
    px = load_prices(tickers=["SPY"], start="2007-01-01")
    rets = compute_log_returns(px)["SPY"]

    print("Computing PSD feature and z-score...")
    feat = rolling_psd_feature(rets)
    z = rolling_zscore(feat)

    # Trim warm-up
    feat = feat.loc["2010-01-01":]
    z = z.loc["2010-01-01":]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Top panel: raw feature
    axes[0].plot(feat.index, feat.values, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel(r"$\phi_t$  (PSD log-ratio)")
    axes[0].set_title("SPY: PSD log-ratio feature")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(feat.mean(), color="grey", linestyle="--", alpha=0.5,
                    linewidth=0.8, label=f"sample mean = {feat.mean():.2f}")
    axes[0].legend(loc="upper right", fontsize=9)

    # Bottom panel: z-score
    axes[1].plot(z.index, z.values, color="darkorange", linewidth=0.8)
    axes[1].axhline(0, color="k", linestyle="--", alpha=0.7, linewidth=0.8)
    axes[1].fill_between(z.index, 0, z.values, where=(z > 0),
                         color="steelblue", alpha=0.2, label="Trending (z > 0)")
    axes[1].fill_between(z.index, 0, z.values, where=(z <= 0),
                         color="firebrick", alpha=0.2, label="Choppy (z ≤ 0)")
    axes[1].set_ylabel(r"$z_t^{\mathrm{psd}}$")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Rolling 252-day z-score")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=9)

    # Tidy date formatting
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out = FIG_DIR / "psd_feature_spy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()