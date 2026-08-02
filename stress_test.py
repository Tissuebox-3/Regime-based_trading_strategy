"""
strategy.py
-----------
Regime-dependent entry/exit rules and position sizing.

Design principle: the rules key off whether a regime label *contains*
"low_vol" / "high_vol" / "trend" / "meanrev", so the same strategy code
works whether regimes came from the percentile, k-means, or HMM
classifier in regime.py (all three name their buckets this way).

Logic:
  - low-vol regimes            -> mean reversion (buy oversold z-score dips)
                                   OR trend-following if the regime label
                                   itself flags a trend (percentile method
                                   distinguishes low_vol_trend vs low_vol_meanrev)
  - high-vol regimes           -> smaller size, faster exits, tighter/no-trade filter
  - everything else (mid-vol)  -> reduced-size trend following
"""

import numpy as np
import pandas as pd

# Position size multipliers by regime bucket (used to scale order value,
# not just a binary entries/exits signal)
SIZE_BY_REGIME = {
    "low_vol": 1.0,
    "mid_vol": 0.6,
    "high_vol": 0.3,
}


def _vol_bucket(regime_label: str) -> str:
    if "high_vol" in regime_label:
        return "high_vol"
    if "mid_vol" in regime_label or "neutral" in regime_label:
        return "mid_vol"
    return "low_vol"  # default / low_vol_*


def build_signals(features: pd.DataFrame, regime: pd.Series):
    """
    Returns
    -------
    entries : pd.Series[bool]
    exits   : pd.Series[bool]
    size_pct: pd.Series[float]  -- fraction of standard position size to use,
                                    driven by regime volatility bucket
    """
    z20 = features["zscore_20"]
    ma20, ma100 = features["ma_20"], features["ma_100"]
    trending_up = ma20 > ma100
    trending_down = ma20 < ma100

    is_low_vol = regime.str.contains("low_vol", na=False)
    is_mid_vol = regime.str.contains("mid_vol|neutral", na=False, regex=True)
    is_high_vol = regime.str.contains("high_vol", na=False)
    label_says_trend = regime.str.contains("trend", na=False)
    label_says_meanrev = regime.str.contains("meanrev", na=False)

    entries = pd.Series(False, index=features.index)
    exits = pd.Series(False, index=features.index)

    # --- Low-vol regime ---
    # If the classifier explicitly separated trend vs mean-reversion
    # (percentile method), respect that split. Otherwise (kmeans/hmm,
    # which only signal vol level) fall back to price-action: trend if
    # MA20>MA100, mean reversion otherwise.
    low_vol_trend_mask = is_low_vol & (label_says_trend | (~label_says_meanrev & trending_up))
    low_vol_meanrev_mask = is_low_vol & (label_says_meanrev | (~label_says_trend & ~trending_up))

    entries |= low_vol_trend_mask & trending_up
    exits |= low_vol_trend_mask & trending_down

    entries |= low_vol_meanrev_mask & (z20 < -1.0)
    exits |= low_vol_meanrev_mask & (z20 > 0.0)

    # --- Mid-vol regime: reduced trend-following only, no mean reversion ---
    entries |= is_mid_vol & trending_up & (features["mom_20"] > 0)
    exits |= is_mid_vol & trending_down

    # --- High-vol regime: no-trade filter unless a very stretched dip,
    #     and exit fast on any mean reversion at all (fast exits) ---
    entries |= is_high_vol & (z20 < -2.0)
    exits |= is_high_vol & (z20 > -0.5)

    entries = entries.fillna(False)
    exits = exits.fillna(False)

    size_pct = regime.map(_vol_bucket).map(SIZE_BY_REGIME).fillna(1.0)

    return entries, exits, size_pct


def buy_and_hold_signals(index: pd.Index):
    """Benchmark 1: buy on day 1, hold forever."""
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    entries.iloc[0] = True
    return entries, exits


def always_flat_signals(index: pd.Index):
    """Benchmark 2: never trade (cash)."""
    entries = pd.Series(False, index=index)
    exits = pd.Series(False, index=index)
    return entries, exits


def sma_crossover_signals(price: pd.Series, fast: int = 20, slow: int = 100):
    """Benchmark 3: simple moving average crossover, regime-agnostic."""
    ma_fast = price.rolling(fast).mean()
    ma_slow = price.rolling(slow).mean()
    cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_down = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    return cross_up.fillna(False), cross_down.fillna(False)
