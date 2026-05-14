# Regime-Filtered Time-Series Momentum

Time-series momentum with frequency-domain and volatility regime filters, evaluated on a 10-asset ETF universe (2010–2026).

## What this is

A systematic trading strategy that filters a standard time-series momentum (TSMOM) signal using two independent regime classifiers: one based on the spectral structure of returns, one based on realized volatility. The two filters are empirically independent, but momentum's risk-adjusted edge is concentrated in the cell where both filters classify the asset as favorable. The intersection-filtered strategy delivers a Sharpe of 0.60 (approx.) and is the only strategy in the comparison set whose performance is stable across in-sample, out-of-sample, and sub-period evaluations.

Full methodology, results, and discussion are in [`report.pdf`](report.pdf). This repo contains the code that produces every table and figure in the report.

## Quick start

```bash
git clone https://github.com/lilNewbie/regime-filtered-tsmom.git
cd regime-filtered-tsmom
pip install -r requirements.txt
python main.py
```

`main.py` runs the headline backtest comparison and produces console output plus a plots in `output/`. Data is pulled from Yahoo Finance via `yfinance`; an internet connection is required on first run.

## What each script does

| Script | Purpose |
|---|---|
| `main.py` | Headline strategy comparison: unconditional, PSD-filtered, and vol-filtered TSMOM with full metrics and equity curves. |
| `combined.py` | Cell-conditional analysis and the intersection-filtered strategy. The central methodological contribution. |
| `period_analysis.py` | Period-by-period robustness: in-sample/out-of-sample split, sub-periods, excluded-2020-2022. |
| `parameter_sensitivity.py` | Sensitivity grid across PSD window length and frequency cutoff. |
| `figs2.py`, `rolling_sharpe_figure.py`, `equity_curves.py` | Generate the figures used in the report. |

The core modules (`spectral.py`, `strategy.py`, `backtest.py`, `data.py`) implement the building blocks: the rolling Welch PSD feature, the TSMOM signal and regime masks, the backtest engine with transaction costs, and the data loader.

## Strategy summary

Four strategies are evaluated, sharing the same TSMOM base (12m–1m signal, vol-scaled to 10% per asset, monthly rebalanced):

1. **Unconditional TSMOM** — trades every month, no regime filter.
2. **PSD-filtered TSMOM** — trades only when the rolling PSD log-ratio z-score is positive (trending regime).
3. **Vol-filtered TSMOM** — trades only when the rolling realized vol z-score is non-positive (low-vol regime).
4. **Intersection-filtered TSMOM** — trades only when both filters classify the asset as favorable.

All parameters are pre-registered. No in-sample tuning was performed.

## Universe

10 liquid ETFs spanning major asset classes: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, VNQ.

## Context

Developed as the course project for *Robo Advisors and Systematic Trading* at NYU Stern School of Business.