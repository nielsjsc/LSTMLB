# Hybrid scaler: StandardScaler for counting stats, MinMaxScaler for rate stats
# with optional log1p transform on counting stats before scaling.
#
# Drop-in replacement for sklearn MinMaxScaler — supports fit, fit_transform,
# transform, inverse_transform, and is pickle-serializable via joblib.

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class HybridScaler:
    """Scaler that applies different strategies to rate vs counting features.

    Parameters
    ----------
    counting_indices : list[int]
        Column positions that correspond to counting stats (HR, 2B, …).
        These get StandardScaler (optionally preceded by log1p).
    rate_indices : list[int]
        Column positions that correspond to rate stats (BB%, AVG, …).
        These get MinMaxScaler(feature_range).
    feature_range : tuple[float, float]
        Range for the MinMaxScaler applied to rate stats.  Default ``(-1, 1)``.
    log_transform_counting : bool
        If True, apply ``log1p`` to counting columns before StandardScaler
        and ``expm1`` on the way back.  Compresses heavy right tails (HR, RBI).
    """

    def __init__(
        self,
        counting_indices: list[int],
        rate_indices: list[int],
        feature_range: tuple[float, float] = (-1, 1),
        log_transform_counting: bool = True,
    ):
        self.counting_indices = np.array(counting_indices, dtype=int)
        self.rate_indices = np.array(rate_indices, dtype=int)
        self.feature_range = feature_range
        self.log_transform_counting = log_transform_counting

        # Internal scalers — created lazily on fit()
        self._rate_scaler = MinMaxScaler(feature_range=feature_range)
        self._counting_scaler = StandardScaler()

        # Total number of features — set on fit
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split(self, X: np.ndarray):
        """Split data into rate and counting sub-arrays."""
        rate = X[:, self.rate_indices] if len(self.rate_indices) else np.empty((len(X), 0))
        counting = X[:, self.counting_indices] if len(self.counting_indices) else np.empty((len(X), 0))
        return rate, counting

    def _merge(self, rate: np.ndarray, counting: np.ndarray, n_samples: int) -> np.ndarray:
        """Reconstruct the full array from rate and counting sub-arrays."""
        out = np.empty((n_samples, self.n_features_in_), dtype=np.float64)
        if len(self.rate_indices):
            out[:, self.rate_indices] = rate
        if len(self.counting_indices):
            out[:, self.counting_indices] = counting
        return out

    def _log_forward(self, counting: np.ndarray) -> np.ndarray:
        """Apply log1p to counting columns (clamp to ≥ 0 first)."""
        if self.log_transform_counting and counting.size:
            return np.log1p(np.maximum(counting, 0.0))
        return counting

    def _log_inverse(self, counting: np.ndarray) -> np.ndarray:
        """Undo log1p: expm1."""
        if self.log_transform_counting and counting.size:
            return np.expm1(counting)
        return counting

    # ------------------------------------------------------------------
    # sklearn-compatible API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        rate, counting = self._split(X)
        counting = self._log_forward(counting)

        if rate.shape[1]:
            self._rate_scaler.fit(rate)
        if counting.shape[1]:
            self._counting_scaler.fit(counting)

        logger.info(
            f"HybridScaler fit: {len(self.rate_indices)} rate cols (MinMax{self.feature_range}), "
            f"{len(self.counting_indices)} counting cols (StandardScaler, "
            f"log1p={self.log_transform_counting})"
        )
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rate, counting = self._split(X)
        counting = self._log_forward(counting)

        rate_s = self._rate_scaler.transform(rate) if rate.shape[1] else rate
        counting_s = self._counting_scaler.transform(counting) if counting.shape[1] else counting

        return self._merge(rate_s, counting_s, n)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        rate_s, counting_s = self._split(X)

        rate = self._rate_scaler.inverse_transform(rate_s) if rate_s.shape[1] else rate_s
        counting = self._counting_scaler.inverse_transform(counting_s) if counting_s.shape[1] else counting_s
        counting = self._log_inverse(counting)

        return self._merge(rate, counting, n)

    # ------------------------------------------------------------------
    # Attributes that external code may inspect (MinMaxScaler compat)
    # ------------------------------------------------------------------

    @property
    def data_min_(self):
        """Return per-feature minimums (for legacy compatibility).

        Rate features come from MinMaxScaler.data_min_; counting features
        are approximated as ``mean - 3*scale`` from StandardScaler.
        """
        out = np.zeros(self.n_features_in_)
        if len(self.rate_indices):
            out[self.rate_indices] = self._rate_scaler.data_min_
        if len(self.counting_indices):
            out[self.counting_indices] = (
                self._counting_scaler.mean_ - 3 * self._counting_scaler.scale_
            )
        return out

    @property
    def data_max_(self):
        out = np.zeros(self.n_features_in_)
        if len(self.rate_indices):
            out[self.rate_indices] = self._rate_scaler.data_max_
        if len(self.counting_indices):
            out[self.counting_indices] = (
                self._counting_scaler.mean_ + 3 * self._counting_scaler.scale_
            )
        return out

    def __repr__(self):
        return (
            f"HybridScaler(counting={len(self.counting_indices)}, "
            f"rate={len(self.rate_indices)}, "
            f"log1p={self.log_transform_counting}, "
            f"range={self.feature_range})"
        )
