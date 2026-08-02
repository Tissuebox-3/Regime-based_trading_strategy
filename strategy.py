"""
regime.py
---------
Three interchangeable regime classifiers, from simplest to most
sophisticated. Each is implemented as a class with `.fit(train_features)`
and `.predict(features)`, so walk-forward validation can fit thresholds
/ centroids / HMM parameters on a TRAIN window only and apply the frozen
parameters to the TEST window -- no lookahead.

  1. PercentileRegimeClassifier  -- rule-based, vol percentile x trend sign
  2. KMeansRegimeClassifier      -- unsupervised clustering on vol/trend/mom
  3. HMMRegimeClassifier         -- Hidden Markov Model on returns/vol

`classify_regime(features, method=...)` remains as a convenience
function for the full-sample (in-sample) case: it fits and predicts on
the same data in one call. For walk-forward, use the classes directly
(see backtest.walk_forward_backtest, which does this for you).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1) Percentile-bucket rule (upgrade over a plain median split: uses
#    tertiles so "high vol" really means top third, not just above avg)
# ---------------------------------------------------------------------
class PercentileRegimeClassifier:
    def __init__(self, low_q: float = 0.33, high_q: float = 0.67):
        self.low_q = low_q
        self.high_q = high_q
        self.low_thresh_ = None
        self.high_thresh_ = None

    def fit(self, features: pd.DataFrame) -> "PercentileRegimeClassifier":
        vol = features["vol_20"].dropna()
        self.low_thresh_ = vol.quantile(self.low_q)
        self.high_thresh_ = vol.quantile(self.high_q)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.low_thresh_ is None:
            raise RuntimeError("Call .fit() before .predict()")
        vol = features["vol_20"]
        trending = features["ma_20"] > features["ma_100"]

        regime = pd.Series(index=features.index, dtype="object")
        regime[(vol <= self.low_thresh_) & trending] = "low_vol_trend"
        regime[(vol <= self.low_thresh_) & (~trending)] = "low_vol_meanrev"
        regime[(vol > self.low_thresh_) & (vol < self.high_thresh_)] = "mid_vol_neutral"
        regime[(vol >= self.high_thresh_) & trending] = "high_vol_trend"
        regime[(vol >= self.high_thresh_) & (~trending)] = "high_vol_defensive"
        return regime.ffill().bfill()


# ---------------------------------------------------------------------
# 2) K-means on volatility / trend strength / momentum
# ---------------------------------------------------------------------
class KMeansRegimeClassifier:
    COLS = ["vol_20", "ma_ratio_20_100", "mom_20"]

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler_ = None
        self.model_ = None
        self.rank_map_ = None
        self.name_map_ = None

    def fit(self, features: pd.DataFrame) -> "KMeansRegimeClassifier":
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        data = features[self.COLS].dropna()
        self.scaler_ = StandardScaler().fit(data.values)
        X = self.scaler_.transform(data.values)

        self.model_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state, n_init=10).fit(X)
        train_labels = self.model_.predict(X)

        # Rank clusters by mean vol (fit on TRAIN only) so labels are
        # interpretable and consistent when applied to a test window.
        tmp = pd.DataFrame({"cluster": train_labels, "vol": data["vol_20"].values})
        order = tmp.groupby("cluster")["vol"].mean().sort_values().index.tolist()
        self.rank_map_ = {cluster: rank for rank, cluster in enumerate(order)}

        if self.n_clusters == 3:
            self.name_map_ = {0: "kmeans_low_vol", 1: "kmeans_mid_vol", 2: "kmeans_high_vol"}
        else:
            self.name_map_ = {i: f"kmeans_cluster_{i}" for i in range(self.n_clusters)}
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict()")
        data = features[self.COLS].dropna()
        if len(data) == 0:
            return pd.Series(index=features.index, dtype="object")

        X = self.scaler_.transform(data.values)
        labels = self.model_.predict(X)
        vol_rank = pd.Series(labels, index=data.index).map(self.rank_map_)
        regime = vol_rank.map(self.name_map_)
        regime = regime.reindex(features.index)
        return regime.ffill().bfill()


# ---------------------------------------------------------------------
# 3) Hidden Markov Model on returns + volatility
# ---------------------------------------------------------------------
class HMMRegimeClassifier:
    COLS = ["ret_1d", "vol_20"]

    def __init__(self, n_states: int = 3, random_state: int = 42, n_iter: int = 200):
        self.n_states = n_states
        self.random_state = random_state
        self.n_iter = n_iter
        self.model_ = None
        self.rank_map_ = None
        self.name_map_ = None

    def fit(self, features: pd.DataFrame) -> "HMMRegimeClassifier":
        from hmmlearn.hmm import GaussianHMM

        data = features[self.COLS].dropna()
        X = data.values

        self.model_ = GaussianHMM(n_components=self.n_states, covariance_type="diag",
                                   n_iter=self.n_iter, random_state=self.random_state)
        self.model_.fit(X)
        train_states = self.model_.predict(X)

        tmp = pd.DataFrame({"state": train_states, "vol": data["vol_20"].values})
        order = tmp.groupby("state")["vol"].mean().sort_values().index.tolist()
        self.rank_map_ = {state: rank for rank, state in enumerate(order)}

        if self.n_states == 3:
            level_names = ["low_vol", "mid_vol", "high_vol"]
        else:
            level_names = [f"level_{i}" for i in range(self.n_states)]
        self.name_map_ = {i: f"hmm_{level_names[i]}" for i in range(self.n_states)}
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            raise RuntimeError("Call .fit() before .predict()")
        data = features[self.COLS].dropna()
        if len(data) == 0:
            return pd.Series(index=features.index, dtype="object")

        # decode a new sequence under the already-fitted model parameters
        # (no refitting -- this is what keeps it leakage-free on test data)
        states = self.model_.predict(data.values)
        vol_rank = pd.Series(states, index=data.index).map(self.rank_map_)
        regime = vol_rank.map(self.name_map_)
        regime = regime.reindex(features.index)
        return regime.ffill().bfill()


REGIME_CLASSIFIERS = {
    "percentile": PercentileRegimeClassifier,
    "kmeans": KMeansRegimeClassifier,
    "hmm": HMMRegimeClassifier,
}


def get_classifier(method: str = "percentile", **kwargs):
    if method not in REGIME_CLASSIFIERS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(REGIME_CLASSIFIERS)}")
    return REGIME_CLASSIFIERS[method](**kwargs)


def classify_regime(features: pd.DataFrame, method: str = "percentile", **kwargs) -> pd.Series:
    """
    Convenience wrapper for the FULL-SAMPLE / in-sample case: fits and
    predicts on the same data in one call. For walk-forward (fit on
    train, predict on test), use get_classifier(method) directly.
    """
    clf = get_classifier(method, **kwargs)
    clf.fit(features)
    return clf.predict(features)
