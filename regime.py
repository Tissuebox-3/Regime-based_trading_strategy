"""
backtest.py
-----------
Wraps vectorbt.Portfolio.from_signals with:
  - regime-scaled position sizing
  - transaction cost + slippage
  - walk-forward train/test splitting (rolling windows)
  - transaction-cost sensitivity sweep
"""

import numpy as np
import pandas as pd
import vectorbt as vbt


def run_backtest(price: pd.Series, entries: pd.Series, exits: pd.Series,
                  size_pct: pd.Series = None, fees: float = 0.0005,
                  slippage: float = 0.0005, init_cash: float = 100_000.0):
    """
    size_pct: optional fraction-of-standard-size per bar (from regime
    sizing). vectorbt's `size` + size_type='percent' lets us scale the
    order to a fraction of available cash at entry.
    """
    kwargs = dict(
        close=price,
        entries=entries,
        exits=exits,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        freq="1D",
    )
    if size_pct is not None:
        kwargs["size"] = size_pct.clip(0.05, 1.0)
        kwargs["size_type"] = "percent"

    pf = vbt.Portfolio.from_signals(**kwargs)
    return pf


def walk_forward_windows(index: pd.DatetimeIndex, n_splits: int = 5,
                          train_frac: float = 0.7):
    """
    Rolling walk-forward windows. Splits the index into `n_splits`
    contiguous blocks; within each block, the first `train_frac` is the
    "train" period (used to fit regime thresholds/params) and the
    remainder is out-of-sample "test".

    Returns a list of dicts: {"train": DatetimeIndex, "test": DatetimeIndex}
    """
    n = len(index)
    block_size = n // n_splits
    windows = []
    for i in range(n_splits):
        start = i * block_size
        end = n if i == n_splits - 1 else (i + 1) * block_size
        block = index[start:end]
        if len(block) < 20:
            continue
        cut = int(len(block) * train_frac)
        train_idx = block[:cut]
        test_idx = block[cut:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        windows.append({"train": train_idx, "test": test_idx})
    return windows


def walk_forward_backtest(price: pd.Series, features: pd.DataFrame,
                           regime_method: str = "percentile",
                           regime_kwargs: dict = None,
                           n_splits: int = 5, fees: float = 0.0005,
                           slippage: float = 0.0005,
                           min_train_size: int = 100) -> pd.DataFrame:
    """
    Leakage-free walk-forward backtest: for each rolling window, the
    regime classifier's parameters (vol thresholds / k-means centroids /
    HMM transition & emission params) are FIT ONLY on that window's TRAIN
    slice, then applied via `.predict()` to classify the TEST slice.
    Signals are built from the test-slice regime labels and the
    backtest is run on the test slice only, so every number reported
    here is genuinely out-of-sample.

    Parameters
    ----------
    price, features : full-sample series/frame (features already computed
        causally over the whole history, which is fine -- only the
        regime classifier's *parameters* need to avoid seeing the future)
    regime_method : "percentile" | "kmeans" | "hmm"
    regime_kwargs : extra kwargs passed to the classifier constructor
    """
    # Local imports to avoid a circular import at module load time
    from regime import get_classifier
    from strategy import build_signals

    regime_kwargs = regime_kwargs or {}
    windows = walk_forward_windows(price.index, n_splits=n_splits)
    rows = []

    for i, w in enumerate(windows):
        train_idx, test_idx = w["train"], w["test"]
        if len(train_idx) < min_train_size:
            continue

        train_features = features.loc[train_idx]
        test_features = features.loc[test_idx]

        # --- FIT regime classifier on TRAIN only ---
        clf = get_classifier(regime_method, **regime_kwargs)
        try:
            clf.fit(train_features)
        except Exception as exc:
            rows.append({"window": i + 1, "test_start": test_idx[0],
                          "test_end": test_idx[-1], "n_days": len(test_idx),
                          "error": str(exc)})
            continue

        # --- PREDICT regime on TEST only (frozen train-fit params) ---
        test_regime = clf.predict(test_features)

        # --- Build signals purely from test-window regime/features ---
        entries, exits, size_pct = build_signals(test_features, test_regime)

        p = price.loc[test_idx]
        if p.isna().all() or len(p) < 10:
            continue

        pf = run_backtest(p, entries, exits, size_pct=size_pct, fees=fees, slippage=slippage)
        stats = pf.stats()
        rows.append({
            "window": i + 1,
            "train_start": train_idx[0],
            "train_end": train_idx[-1],
            "test_start": test_idx[0],
            "test_end": test_idx[-1],
            "n_train_days": len(train_idx),
            "n_test_days": len(test_idx),
            "regime_mix_test": test_regime.value_counts(normalize=True).round(2).to_dict(),
            "total_return_%": stats.get("Total Return [%]", np.nan),
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "max_drawdown_%": stats.get("Max Drawdown [%]", np.nan),
            "n_trades": stats.get("Total Trades", np.nan),
            "win_rate_%": stats.get("Win Rate [%]", np.nan),
        })
    return pd.DataFrame(rows)


def cost_sensitivity_sweep(price: pd.Series, entries: pd.Series, exits: pd.Series,
                            size_pct: pd.Series = None,
                            cost_bps_list=(0, 5, 10, 25)) -> pd.DataFrame:
    """
    Re-runs the same signal set at different flat transaction-cost
    assumptions (fees = slippage = bps each side) to show whether
    performance survives realistic costs.
    """
    rows = []
    for bps in cost_bps_list:
        c = bps / 10_000.0
        pf = run_backtest(price, entries, exits, size_pct=size_pct, fees=c, slippage=c)
        stats = pf.stats()
        rows.append({
            "cost_bps_each_side": bps,
            "total_return_%": stats.get("Total Return [%]", np.nan),
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "max_drawdown_%": stats.get("Max Drawdown [%]", np.nan),
            "n_trades": stats.get("Total Trades", np.nan),
        })
    return pd.DataFrame(rows)
