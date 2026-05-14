from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(positions: pd.DataFrame,
                 returns: pd.DataFrame,
                 cost_bps: float = 5.0,
                 start_date: str = "2010-01-01") -> pd.Series:
    """Compute the portfolio equity curve.

    Parameters
    ----------
    positions : daily positions per asset (set at close of t-1, held through t)
    returns   : daily log returns per asset
    cost_bps  : transaction cost in basis points per unit of turnover per asset
    start_date: drop results before this date (warm-up exclusion)

    Returns
    -------
    Series of daily portfolio returns (arithmetic), indexed by date.
    """
    # Align
    pos = positions.reindex(returns.index).fillna(0)
    rets = returns.reindex(pos.index).fillna(0)

    # Equal-weight across N asset slots — each slot gets 1/N of capital
    n_assets = pos.shape[1]
    weights = pos / n_assets  # dollar weight per asset per $1 of capital

    # Portfolio arithmetic return (convert log returns to arith for summing)
    arith_rets = np.exp(rets) - 1
    gross_ret = (weights * arith_rets).sum(axis=1)

    # Transaction costs: proportional to change in absolute weight
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    costs = turnover * (cost_bps / 10000.0)

    net_ret = gross_ret - costs
    net_ret = net_ret.loc[start_date:]
    return net_ret.rename("portfolio_return")


def performance_metrics(returns: pd.Series) -> dict:
    """Summary metrics for a strategy return series.

    All annualizations use 252 trading days. Sharpe uses a zero risk-free
    rate (a common simplification — justifiable because we're comparing
    strategies that share the same risk-free benchmark).
    """
    r = returns.dropna()
    if len(r) < 2:
        return {}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    monthly = (1 + r).resample("ME").prod() - 1
    hit_rate = (monthly > 0).mean()

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "hit_rate_monthly": hit_rate,
        "n_days": len(r),
    }


def compare_strategies(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """Side-by-side metric comparison across named strategies."""
    rows = {name: performance_metrics(r) for name, r in strategy_returns.items()}
    df = pd.DataFrame(rows).T
    return df.round(4)


# ---- Statistical comparison -----------------------------------------------

def sharpe_diff_test(r1: pd.Series, r2: pd.Series) -> dict:
    """Approximate test of Sharpe ratio difference (Jobson-Korkie / Memmel).

    This is the right test for 'is strategy 1's Sharpe significantly
    different from strategy 2's Sharpe' when the return series are
    contemporaneous (same dates). The exact Memmel (2003) correction
    adjusts for the fact that the two Sharpes are estimated from
    overlapping data.

    Returns dict with z-statistic and two-sided p-value.
    """
    from scipy import stats as sp_stats

    # Align on common dates
    common = r1.index.intersection(r2.index)
    x = r1.loc[common].dropna()
    y = r2.loc[common].dropna()
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]
    n = len(x)

    if n < 30:
        return {"z": np.nan, "pvalue": np.nan, "note": "insufficient data"}

    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sh_x = mx / sx if sx > 0 else 0
    sh_y = my / sy if sy > 0 else 0

    corr = x.corr(y)
    # Memmel (2003) standard error for difference of Sharpe ratios
    variance = (1.0 / n) * (
        2 - 2 * corr
        + 0.5 * (sh_x ** 2 + sh_y ** 2 - 2 * sh_x * sh_y * corr ** 2)
    )
    se = np.sqrt(max(variance, 1e-12))
    z = (sh_x - sh_y) / se
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))

    return {"sharpe_1": sh_x * np.sqrt(252),
            "sharpe_2": sh_y * np.sqrt(252),
            "z": z, "pvalue": p, "n": n}