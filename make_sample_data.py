"""
plots.py
--------
Saves the standard evaluation charts:
  - equity curve (strategy vs benchmark)
  - drawdown curve
  - regime timeline
  - histogram of trade returns
  - Monte Carlo terminal-return distribution
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def plot_equity_curve(pf, benchmark_price: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    strat_equity = pf.value()
    strat_equity_norm = strat_equity / strat_equity.iloc[0]
    bench_norm = benchmark_price / benchmark_price.iloc[0]

    ax.plot(strat_equity_norm.index, strat_equity_norm.values, label="Strategy", linewidth=1.6)
    ax.plot(bench_norm.index, bench_norm.reindex(strat_equity_norm.index).values,
            label="Benchmark (buy & hold)", linewidth=1.2, alpha=0.75)
    ax.set_title("Equity Curve: Strategy vs Benchmark")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_drawdown(pf, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 4))
    dd = pf.drawdown() * 100
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.5)
    ax.plot(dd.index, dd.values, color="firebrick", linewidth=0.8)
    ax.set_title("Strategy Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_regime_timeline(price: pd.Series, regime: pd.Series, out_path: str):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(price.index, price.values, color="black", linewidth=1.0, zorder=3)

    labels = regime.dropna().unique().tolist()
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(sorted(labels))}

    # Shade background by contiguous regime blocks
    r = regime.reindex(price.index).ffill()
    start = r.index[0]
    current = r.iloc[0]
    for i in range(1, len(r)):
        if r.iloc[i] != current or i == len(r) - 1:
            end = r.index[i]
            ax.axvspan(start, end, color=color_map.get(current, "grey"), alpha=0.18, zorder=1)
            start = r.index[i]
            current = r.iloc[i]

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in color_map.values()]
    ax.legend(handles, list(color_map.keys()), loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Price with Regime Timeline")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trade_return_hist(pf, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    trades = pf.trades.records_readable
    if len(trades) == 0:
        ax.text(0.5, 0.5, "No closed trades", ha="center", va="center")
    else:
        rets = trades["Return"] * 100
        ax.hist(rets, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Trade Return (%)")
    ax.set_title("Distribution of Individual Trade Returns")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mc_distribution(stress_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(stress_df["terminal_return"] * 100, bins=50, color="seagreen",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="black", linewidth=1)
    p5 = np.percentile(stress_df["terminal_return"], 5) * 100
    axes[0].axvline(p5, color="firebrick", linestyle="--", linewidth=1.2,
                     label=f"5th pct: {p5:.1f}%")
    axes[0].set_title("Monte Carlo: Terminal Return Distribution")
    axes[0].set_xlabel("Terminal Return (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(stress_df["max_drawdown"] * 100, bins=50, color="indianred",
                 edgecolor="white", alpha=0.85)
    dd5 = np.percentile(stress_df["max_drawdown"], 5) * 100
    axes[1].axvline(dd5, color="black", linestyle="--", linewidth=1.2,
                     label=f"5th pct: {dd5:.1f}%")
    axes[1].set_title("Monte Carlo: Max Drawdown Distribution")
    axes[1].set_xlabel("Max Drawdown (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
