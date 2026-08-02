# Regime-Based Systematic Trading Strategy with Walk-Forward Validation and Monte Carlo Stress Testing

A regime-aware systematic trading pipeline: it classifies the market into
volatility/trend regimes, applies different rules (mean reversion, trend
following, defensive sizing) depending on the regime, and evaluates the
result out-of-sample with walk-forward testing, transaction-cost
sensitivity, benchmark comparisons, and Monte Carlo stress testing.

## Folder structure

```
quant_project/
├── data/
│   └── prices.csv          # date, TRADE_ASSET, BENCHMARK (synthetic sample data included)
├── src/
│   ├── make_sample_data.py # generates the synthetic sample data
│   ├── features.py         # returns, vol, momentum, MAs, z-scores, drawdown
│   ├── regime.py           # 3 swappable regime classifiers
│   ├── strategy.py         # regime-dependent entries/exits/sizing + benchmarks
│   ├── backtest.py         # vectorbt engine, walk-forward, cost sensitivity
│   ├── stress_test.py      # Monte Carlo bootstrap stress test
│   ├── plots.py            # all evaluation charts
│   └── main.py             # runs the full pipeline end to end
└── outputs/                # generated CSVs + PNGs land here
```

## Quick start

```bash
pip install vectorbt scikit-learn hmmlearn matplotlib pandas numpy

# 1) generate sample data (skip this once you have real data in data/prices.csv)
python3 src/make_sample_data.py

# 2) run the full pipeline
python3 src/main.py
```

Everything in `outputs/` is regenerated on each run.

## Using real data

Replace `data/prices.csv` with your own file containing at minimum:

```
date,TRADE_ASSET,BENCHMARK
2016-01-04,100.48,101.07
...
```

`TRADE_ASSET` is the instrument you trade (e.g. SPY, QQQ, TLT, GLD, or a
basket you've combined into one series). `BENCHMARK` is what you compare
against (often the same as `TRADE_ASSET` for a pure buy-and-hold
comparison, or a different index). No other code changes are needed —
`main.py` reads whatever is in `data/prices.csv`.

## Pipeline stages

**1. Features** (`features.py`) — daily/5-day returns, log returns,
10/20/60-day momentum, 20/50/100-day moving averages, 10/20/60-day
annualized rolling vol, vol-of-vol, 20/60-day price z-scores, drawdown
from running peak, and volume-based features if a volume column exists.

**2. Regime classification** (`regime.py`) — three interchangeable
methods, selected via `REGIME_METHOD` in `main.py`:
- `"percentile"` (default) — vol tertiles × MA20/MA100 trend sign, giving
  5 named buckets (`low_vol_trend`, `low_vol_meanrev`, `mid_vol_neutral`,
  `high_vol_trend`, `high_vol_defensive`).
- `"kmeans"` — unsupervised clustering on vol / trend-strength / momentum,
  clusters ranked and labeled by average volatility.
- `"hmm"` — Gaussian Hidden Markov Model fit on returns + rolling vol,
  states ranked and labeled by average volatility.

**3. Strategy logic** (`strategy.py`) — rules key off whether a regime
label contains `low_vol` / `mid_vol` / `high_vol` and `trend` /
`meanrev`, so the same strategy code works under any of the three
classifiers:
- Low-vol trending → trend following (long while MA20 > MA100)
- Low-vol non-trending → mean reversion (buy z-score < −1, exit at 0)
- Mid-vol → reduced-size trend following only
- High-vol → no-trade filter unless deeply oversold (z < −2), fast exits
  (z > −0.5), and position size cut via `SIZE_BY_REGIME`

Three benchmarks are also built for comparison: buy-and-hold, always-flat
(cash), and a plain 20/100 SMA crossover.

**4. Evaluation** (`backtest.py`, `stress_test.py`, `plots.py`):
- Full-sample backtest with fees + slippage (`vectorbt.Portfolio.from_signals`)
- **Walk-forward testing (leakage-free)**: the sample is split into N
  rolling blocks (default 5); within each block the first 70% is
  "train" and the last 30% is "test". The regime classifier's
  parameters — percentile vol thresholds, k-means centroids, or HMM
  transition/emission parameters — are **fit only on that block's train
  slice**, then applied via `.predict()` (frozen, no refitting) to
  classify the test slice. Signals and the backtest are then built and
  run on the test slice only, so every reported number is genuinely
  out-of-sample. See `regime.py`'s classifier classes and
  `backtest.walk_forward_backtest`. Results: `outputs/walk_forward_results.csv`
  (also reports each window's regime mix, so you can see e.g. a window
  that was mostly `high_vol_defensive` vs mostly `low_vol_trend`).
- **Transaction-cost sensitivity**: the same signals re-run at 0/5/10/25
  bps round-trip cost. See `outputs/cost_sensitivity.csv`.
- **Benchmark comparison** across all four strategies. See
  `outputs/benchmark_comparison.csv`.
- **Monte Carlo stress test**: block-bootstraps the strategy's realized
  daily returns (preserves short-run autocorrelation/vol clustering) to
  build a distribution of 1-year terminal returns and max drawdowns. See
  `outputs/monte_carlo_summary.csv` and `monte_carlo_distribution.png`.

## Outputs

| File | Contents |
|---|---|
| `equity_curve.png` | Strategy vs benchmark growth of $1 |
| `drawdown.png` | Strategy drawdown over time |
| `regime_timeline.png` | Price with regime periods shaded |
| `trade_return_hist.png` | Distribution of individual closed-trade returns |
| `monte_carlo_distribution.png` | MC terminal-return and max-drawdown histograms |
| `walk_forward_results.csv` | Per-window out-of-sample stats |
| `cost_sensitivity.csv` | Performance at 0/5/10/25 bps |
| `benchmark_comparison.csv` | Strategy vs buy&hold vs flat vs SMA crossover |
| `monte_carlo_paths.csv` / `monte_carlo_summary.csv` | Raw MC paths + percentile summary |
| `strategy_signals.csv` | Full feature/regime/signal table for inspection |

## Known limitations / next steps

- The synthetic data in `data/prices.csv` is for pipeline testing only —
  swap in real market data before drawing any research conclusions.
- `hmmlearn`'s EM fit is randomly initialized and can print "Model is
  not converging" on short training windows; this is a benign warning
  (it just means the log-likelihood improvement went below tolerance),
  not a crash. Increase `min_train_size` in `walk_forward_backtest` or
  `n_iter` in `HMMRegimeClassifier` if you want tighter convergence.
- Strategy parameters themselves (z-score thresholds, MA windows) are
  still fixed constants rather than refit per window — only the regime
  classifier is refit. Extending `build_signals` to accept fitted
  thresholds is the natural next step if you want to walk-forward those too.
