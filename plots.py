"""
main.py
-------
End-to-end pipeline:
  1. Load data
  2. Build features
  3. Classify regime (percentile / kmeans / hmm -- swap via REGIME_METHOD)
  4. Build regime-dependent signals
  5. Full-sample backtest
  6. Walk-forward out-of-sample backtest
  7. Transaction-cost sensitivity sweep
  8. Benchmark comparison (buy&hold, always-flat, SMA crossover)
  9. Monte Carlo stress test
  10. Save all plots + CSV outputs

Run with:  python3 src/main.py
"""

import os
import numpy as np
import pandas as pd

from features import build_features
from regime import classify_regime
from strategy import (
    build_signals, buy_and_hold_signals, always_flat_signals,
    sma_crossover_signals,
)
from backtest import run_backtest, walk_forward_backtest, cost_sensitivity_sweep
from stress_test import monte_carlo_stress, summarize_stress
from plots import (
    plot_equity_curve, plot_drawdown, plot_regime_timeline,
    plot_trade_return_hist, plot_mc_distribution,
)

# -----------------------
# CONFIG
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "prices.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
ASSET_COL = "TRADE_ASSET"
BENCH_COL = "BENCHMARK"

REGIME_METHOD = "percentile"  # "percentile" | "kmeans" | "hmm"
FEE = 0.0005                  # 5 bps
SLIPPAGE = 0.0005              # 5 bps
N_WF_SPLITS = 5
COST_SWEEP_BPS = (0, 5, 10, 25)
MC_PATHS = 10_000
MC_HORIZON = 252


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    for col in (ASSET_COL, BENCH_COL):
        if col not in df.columns:
            raise ValueError(f"CSV must contain a '{col}' column")
    return df


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) DATA
    prices = load_prices(CSV_PATH)
    asset = prices[ASSET_COL].dropna()
    bench = prices[BENCH_COL].dropna()

    # 2) FEATURES
    features = build_features(asset)

    # 3) REGIME
    regime = classify_regime(features, method=REGIME_METHOD)

    print_header(f"REGIME DISTRIBUTION ({REGIME_METHOD})")
    print(regime.value_counts())

    # 4) SIGNALS
    entries, exits, size_pct = build_signals(features, regime)

    # 5) FULL-SAMPLE BACKTEST
    pf = run_backtest(asset, entries, exits, size_pct=size_pct, fees=FEE, slippage=SLIPPAGE)
    print_header("FULL-SAMPLE BACKTEST STATS (regime strategy)")
    print(pf.stats())

    # 6) WALK-FORWARD OUT-OF-SAMPLE BACKTEST
    # Regime classifier is refit on each window's TRAIN slice and applied
    # (frozen) to that window's TEST slice -- no lookahead. See
    # backtest.walk_forward_backtest / regime.py's fit/predict classes.
    wf_results = walk_forward_backtest(
        asset, features, regime_method=REGIME_METHOD,
        n_splits=N_WF_SPLITS, fees=FEE, slippage=SLIPPAGE,
    )
    print_header("WALK-FORWARD OUT-OF-SAMPLE RESULTS (regime refit per window)")
    print(wf_results.to_string(index=False))
    wf_results.to_csv(os.path.join(OUT_DIR, "walk_forward_results.csv"), index=False)

    # 7) TRANSACTION-COST SENSITIVITY
    cost_sweep = cost_sensitivity_sweep(asset, entries, exits, size_pct=size_pct,
                                         cost_bps_list=COST_SWEEP_BPS)
    print_header("TRANSACTION-COST SENSITIVITY")
    print(cost_sweep.to_string(index=False))
    cost_sweep.to_csv(os.path.join(OUT_DIR, "cost_sensitivity.csv"), index=False)

    # 8) BENCHMARK COMPARISON
    bh_entries, bh_exits = buy_and_hold_signals(asset.index)
    pf_bh = run_backtest(asset, bh_entries, bh_exits, fees=FEE, slippage=SLIPPAGE)

    flat_entries, flat_exits = always_flat_signals(asset.index)
    pf_flat = run_backtest(asset, flat_entries, flat_exits, fees=FEE, slippage=SLIPPAGE)

    sma_entries, sma_exits = sma_crossover_signals(asset)
    pf_sma = run_backtest(asset, sma_entries, sma_exits, fees=FEE, slippage=SLIPPAGE)

    def summarize(pf_, name):
        s = pf_.stats()
        return {
            "strategy": name,
            "total_return_%": s.get("Total Return [%]", np.nan),
            "sharpe": s.get("Sharpe Ratio", np.nan),
            "max_drawdown_%": s.get("Max Drawdown [%]", np.nan),
            "n_trades": s.get("Total Trades", np.nan),
        }

    comparison = pd.DataFrame([
        summarize(pf, "regime_strategy"),
        summarize(pf_bh, "buy_and_hold"),
        summarize(pf_flat, "always_flat"),
        summarize(pf_sma, "sma_crossover_20_100"),
    ])
    print_header("BENCHMARK COMPARISON")
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUT_DIR, "benchmark_comparison.csv"), index=False)

    # 9) MONTE CARLO STRESS TEST
    strat_returns = pf.returns()
    stress = monte_carlo_stress(strat_returns, n_paths=MC_PATHS, horizon=MC_HORIZON, mode="block")
    stress_summary = summarize_stress(stress)
    print_header("MONTE CARLO STRESS TEST SUMMARY")
    print(stress_summary)
    stress.to_csv(os.path.join(OUT_DIR, "monte_carlo_paths.csv"), index=False)
    stress_summary.to_csv(os.path.join(OUT_DIR, "monte_carlo_summary.csv"))

    # 10) PLOTS
    plot_equity_curve(pf, bench, os.path.join(OUT_DIR, "equity_curve.png"))
    plot_drawdown(pf, os.path.join(OUT_DIR, "drawdown.png"))
    plot_regime_timeline(asset, regime, os.path.join(OUT_DIR, "regime_timeline.png"))
    plot_trade_return_hist(pf, os.path.join(OUT_DIR, "trade_return_hist.png"))
    plot_mc_distribution(stress, os.path.join(OUT_DIR, "monte_carlo_distribution.png"))

    # Save signals/features for inspection
    features.assign(regime=regime, entries=entries, exits=exits,
                     size_pct=size_pct).to_csv(
        os.path.join(OUT_DIR, "strategy_signals.csv"))

    print_header("DONE")
    print(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
