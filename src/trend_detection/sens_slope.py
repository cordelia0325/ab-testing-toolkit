"""
Sen's Slope Estimator

A robust, non-parametric estimator for the slope of a trend line.
Sen's slope is the median of all pairwise slopes, making it resistant to outliers.

References:
    - Sen, P. K. (1968). Estimates of the regression coefficient based on 
      Kendall's tau. Journal of the American Statistical Association.
    - Gilbert, R. O. (1987). Statistical Methods for Environmental Pollution Monitoring.

Author: Chuqiao Huang
"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SenSlopeResult:
    """Result container for Sen's slope estimation."""
    slope: float
    intercept: float
    conf_interval: Optional[Tuple[float, float]]
    n_slopes: int
    trend: str = ""
    
    def to_dict(self):
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'conf_interval': self.conf_interval,
            'n_slopes': self.n_slopes,
            'trend': self.trend
        }


def compute_all_slopes(x: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise slopes.
    
    For each pair (i, j) where i < j:
        slope_{ij} = (x_j - x_i) / (j - i)
    
    Args:
        x: Input time series array
    
    Returns:
        Sorted array of all pairwise slopes
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    if n < 2:
        return np.array([], dtype=np.float64)
    
    slopes = []
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        lag = np.arange(1, len(diff) + 1, dtype=float)
        s = diff / lag
        s = s[np.isfinite(s)]
        if s.size > 0:
            slopes.append(s)
    
    if not slopes:
        return np.array([], dtype=np.float64)
    
    all_slopes = np.concatenate(slopes)
    all_slopes.sort(kind='mergesort')
    
    return all_slopes


def compute_sens_slope(x: np.ndarray) -> float:
    """
    Compute Sen's slope estimator (median of all pairwise slopes).
    
    Args:
        x: Input time series array
    
    Returns:
        Sen's slope estimate
    """
    slopes = compute_all_slopes(x)
    
    if slopes.size == 0:
        return float('nan')
    
    return float(np.median(slopes))


def compute_intercept(x: np.ndarray, slope: float) -> float:
    """
    Compute intercept using median of (x_i - slope * i).
    
    Args:
        x: Input time series array
        slope: Estimated slope
    
    Returns:
        Intercept estimate
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x), dtype=float)
    intercepts = x - slope * t
    return float(np.median(intercepts))


def _confidence_interval_exact(n: int, alpha: float) -> Optional[float]:
    """
    Get C value from exact table for small samples (n <= 10).
    Used for confidence interval calculation.
    """
    # Small sample table for CI calculation
    from .mann_kendall import MK_SMALL_SAMPLE_TABLE
    
    table = MK_SMALL_SAMPLE_TABLE.get(int(n))
    if table is None:
        return None
    
    M = n * (n - 1) // 2
    valid_keys = sorted(k for k in table if (k % 2) == (M % 2))
    target = alpha / 2.0
    
    # Find smallest key where P <= alpha/2
    candidates = [k for k in valid_keys if table[k] <= target]
    if candidates:
        return float(min(candidates))
    
    # Fallback: largest key where P > alpha/2
    larger = [k for k in valid_keys if table[k] > target]
    return float(max(larger)) if larger else float(valid_keys[-1])


def _confidence_interval_normal(var_s: float, alpha: float) -> float:
    """
    Compute C value using normal approximation.
    
    C = z_{1-alpha/2} * sqrt(Var(S))
    """
    if not np.isfinite(var_s) or var_s <= 0:
        return 0.0
    
    z = norm.ppf(1 - alpha / 2.0)
    return float(z * np.sqrt(var_s))


def sens_slope_confidence_interval(
    slopes_sorted: np.ndarray,
    C: float
) -> Tuple[float, float]:
    """
    Compute confidence interval for Sen's slope.
    
    The interval is based on the rank positions in the sorted slope array.
    
    Args:
        slopes_sorted: Sorted array of all pairwise slopes
        C: Critical value (from exact table or normal approximation)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    M = slopes_sorted.size
    
    # Compute rank indices (1-based, then convert to 0-based)
    L = int(np.floor((M - C) / 2.0))
    U = int(np.ceil((M + C) / 2.0))
    
    # Clip to valid range [0, M-1]
    L = max(1, min(L, M)) - 1
    U = max(1, min(U, M)) - 1
    
    low, high = float(slopes_sorted[L]), float(slopes_sorted[U])
    
    # Ensure proper ordering
    if low > high:
        low, high = high, low
    
    return (low, high)


def sens_slope(
    x: np.ndarray,
    alpha: float = 0.05,
    var_s: Optional[float] = None
) -> SenSlopeResult:
    """
    Compute Sen's slope with confidence interval.
    
    Args:
        x: Input time series array
        alpha: Significance level for confidence interval (default 0.05 for 95% CI)
        var_s: Variance of S statistic (if None, uses normal approximation)
    
    Returns:
        SenSlopeResult object
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Compute all slopes
    slopes_sorted = compute_all_slopes(x)
    
    if slopes_sorted.size == 0:
        return SenSlopeResult(
            slope=float('nan'),
            intercept=float('nan'),
            conf_interval=None,
            n_slopes=0
        )
    
    # Sen's slope = median of all pairwise slopes
    slope = float(np.median(slopes_sorted))
    intercept = compute_intercept(x, slope)
    
    # Compute confidence interval
    conf_interval = None
    
    if 4 <= n <= 10:
        # Use exact table for small samples
        C = _confidence_interval_exact(n, alpha)
        if C is not None:
            conf_interval = sens_slope_confidence_interval(slopes_sorted, C)
    else:
        # Use normal approximation
        if var_s is None:
            # Compute variance without ties correction
            var_s = n * (n - 1) * (2 * n + 5) / 18.0
        
        C = _confidence_interval_normal(var_s, alpha)
        conf_interval = sens_slope_confidence_interval(slopes_sorted, C)
    
    # Determine trend based on CI
    if conf_interval is not None:
        lo, hi = conf_interval
        trend = ("significant_increase" if lo > 0 else
                 "significant_decrease" if hi < 0 else
                 "no_significant_trend")
    else:
        trend = "cannot_determine"
    
    return SenSlopeResult(
        slope=slope,
        intercept=intercept,
        conf_interval=conf_interval,
        n_slopes=slopes_sorted.size,
        trend=trend
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample data with known slope
    n = 20
    true_slope = 0.3
    t = np.arange(n)
    noise = np.random.normal(0, 0.5, n)
    data = 2.0 + true_slope * t + noise
    
    # Estimate slope
    result = sens_slope(data, alpha=0.05)
    
    print("Sen's Slope Results:")
    print(f"  True slope: {true_slope}")
    print(f"  Estimated slope: {result.slope:.4f}")
    print(f"  Intercept: {result.intercept:.4f}")
    print(f"  95% CI: [{result.conf_interval[0]:.4f}, {result.conf_interval[1]:.4f}]"
          if result.conf_interval else "  95% CI: N/A")
    print(f"  Number of pairwise slopes: {result.n_slopes}")
    print(f"  trend: {result.trend}")
