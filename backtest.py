"""
make_sample_data.py
--------------------
Generates a synthetic but realistic multi-regime price series so the
pipeline can be run and tested end-to-end without needing a live data feed.

Replace data/prices.csv with real data (e.g. exported from your broker,
Stooq, or any data vendor you have access to) when you're ready -- the
rest of the pipeline doesn't care where the CSV came from as long as it
has the columns: date, TRADE_ASSET, BENCHMARK.

Simulation design:
  - Alternates between low-vol/trending, low-vol/choppy, and high-vol/
    stressed regimes with a Markov-like regime switch, so the downstream
    regime classifier has real structure to detect (not just noise).
  - BENCHMARK is a simple buy-and-hold-able series (e.g. a proxy for
    SPY) that is correlated with, but not identical to, TRADE_ASSET.
"""

import numpy as np
import pandas as pd


def generate_synthetic_prices(
    n_days: int = 2000,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Regime definitions: (mean daily drift, daily vol, avg duration in days)
    regimes = {
        "calm_bull": (0.0006, 0.007, 60),
        "choppy": (0.0000, 0.010, 40),
        "stressed": (-0.0010, 0.028, 25),
    }
    regime_names = list(regimes.keys())

    dates = pd.bdate_range("2016-01-04", periods=n_days)

    asset_rets = np.empty(n_days)
    regime_path = []

    i = 0
    current = rng.choice(regime_names)
    while i < n_days:
        mu, sigma, avg_dur = regimes[current]
        dur = max(5, int(rng.exponential(avg_dur)))
        dur = min(dur, n_days - i)
        shock = rng.standard_t(df=5, size=dur) * sigma + mu
        asset_rets[i:i + dur] = shock
        regime_path.extend([current] * dur)
        i += dur
        # Transition: stressed periods are more likely to follow choppy ones
        if current == "calm_bull":
            current = rng.choice(regime_names, p=[0.7, 0.25, 0.05])
        elif current == "choppy":
            current = rng.choice(regime_names, p=[0.35, 0.45, 0.20])
        else:
            current = rng.choice(regime_names, p=[0.30, 0.45, 0.25])

    asset_price = start_price * np.cumprod(1 + asset_rets)

    # Benchmark: correlated but distinct process (lower vol, steadier drift)
    bench_rets = 0.55 * asset_rets + rng.normal(0.0002, 0.006, size=n_days)
    bench_price = start_price * np.cumprod(1 + bench_rets)

    df = pd.DataFrame(
        {
            "date": dates,
            "TRADE_ASSET": asset_price,
            "BENCHMARK": bench_price,
            "true_regime": regime_path[:n_days],
        }
    )
    return df


if __name__ == "__main__":
    df = generate_synthetic_prices()
    out_path = "/home/claude/quant_project/data/prices.csv"
    df.drop(columns=["true_regime"]).to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())
    print("\nRegime distribution (ground truth, for reference only):")
    print(df["true_regime"].value_counts())
