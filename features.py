"""
stress_test.py
--------------
Monte Carlo stress testing on the strategy's realized return stream.

Two resampling modes:
  - "iid"   : bootstrap individual daily returns (breaks autocorrelation,
              good for tail-risk / worst-case sizing)
  - "block" : block bootstrap (preserves short-run autocorrelation /
              volatility clustering, more realistic for drawdown paths)
"""

import numpy as np
import pandas as pd


def monte_carlo_stress(returns: pd.Series, n_paths: int = 10_000,
                        horizon: int = 252, mode: str = "block",
                        block_size: int = 10, seed: int = 7) -> pd.DataFrame:
    rets = returns.dropna().values
    if len(rets) < 20:
        raise ValueError("Not enough returns for a meaningful stress test")

    rng = np.random.default_rng(seed)
    terminal = np.empty(n_paths)
    max_dd = np.empty(n_paths)
    sharpe = np.empty(n_paths)

    for i in range(n_paths):
        if mode == "iid":
            sample = rng.choice(rets, size=horizon, replace=True)
        elif mode == "block":
            sample = _block_bootstrap(rets, horizon, block_size, rng)
        else:
            raise ValueError("mode must be 'iid' or 'block'")

        equity = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1
        terminal[i] = equity[-1] - 1
        max_dd[i] = dd.min()
        vol = sample.std()
        sharpe[i] = (sample.mean() / vol * np.sqrt(252)) if vol > 0 else 0.0

    return pd.DataFrame({
        "terminal_return": terminal,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    })


def _block_bootstrap(rets: np.ndarray, horizon: int, block_size: int,
                      rng: np.random.Generator) -> np.ndarray:
    n = len(rets)
    out = np.empty(horizon)
    filled = 0
    while filled < horizon:
        start = rng.integers(0, max(1, n - block_size))
        block = rets[start:start + block_size]
        take = min(len(block), horizon - filled)
        out[filled:filled + take] = block[:take]
        filled += take
    return out


def summarize_stress(stress_df: pd.DataFrame) -> pd.DataFrame:
    percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    summary = stress_df.describe(percentiles=percentiles).T
    summary["prob_loss"] = (stress_df["terminal_return"] < 0).mean() if "terminal_return" in stress_df else np.nan
    return summary
