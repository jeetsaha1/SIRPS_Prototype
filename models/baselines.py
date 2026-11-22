import numpy as np
import pandas as pd


class ZeroBaseline:
    """Predicts zero excess return."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class ExpandingMeanBaseline:
    """Past realised mean, purged."""
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.mean(y)) if len(y) > 0 else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MomentumRankBaseline:
    """Ranks firms by mom_12_1 or ret_63d."""
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Uses mom_12_1 if available
        if isinstance(X, pd.DataFrame) and "mom_12_1" in X.columns:
            vals = X["mom_12_1"].fillna(0).values
            # Normalized rank centered around 0
            ranks = pd.Series(vals).rank(pct=True).values - 0.5
            return ranks * 0.04
        return np.zeros(len(X))
