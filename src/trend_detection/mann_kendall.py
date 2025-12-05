"""
Mann-Kendall Trend Test with HR98 Autocorrelation Correction

This module implements the Mann-Kendall trend test for detecting monotonic trends
in time series data, with optional Hamed & Rao (1998) variance correction for
autocorrelated data.

References:
    - Mann, H. B. (1945). Nonparametric tests against trend. Econometrica.
    - Kendall, M. G. (1975). Rank Correlation Methods. Griffin, London.
    - Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test 
      for autocorrelated data. Journal of Hydrology.
    - Gilbert, R. O. (1987). Statistical Methods for Environmental Pollution Monitoring.

Author: Chuqiao Huang
"""

import numpy as np
from scipy.stats import norm, t as tdist, rankdata
from functools import lru_cache
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


# =============================================================================
# Small Sample Exact Tables (Gilbert 1987)
# =============================================================================

# Right-tail probability table for small samples (n=4 to 10) without ties
MK_SMALL_SAMPLE_TABLE = {
    4: {0: 0.625, 2: 0.375, 4: 0.167, 6: 0.042},
    5: {0: 0.592, 2: 0.408, 4: 0.242, 6: 0.117, 8: 0.042, 10: 0.0083},
    6: {1: 0.500, 3: 0.360, 5: 0.235, 7: 0.136, 9: 0.068, 11: 0.028, 13: 0.0083, 15: 0.0014},
    7: {1: 0.500, 3: 0.386, 5: 0.281, 7: 0.191, 9: 0.119, 11: 0.068, 13: 0.035, 
        15: 0.015, 17: 0.0054, 19: 0.0014, 21: 0.00020},
    8: {0: 0.548, 2: 0.452, 4: 0.360, 6: 0.274, 8: 0.199, 10: 0.138, 12: 0.089,
        14: 0.054, 16: 0.031, 18: 0.016, 20: 0.0071, 22: 0.0028, 24: 0.00087, 
        26: 0.00019, 28: 0.000025},
    9: {0: 0.540, 2: 0.460, 4: 0.381, 6: 0.306, 8: 0.238, 10: 0.179, 12: 0.130,
        14: 0.090, 16: 0.060, 18: 0.038, 20: 0.022, 22: 0.012, 24: 0.0063, 
        26: 0.0029, 28: 0.0012, 30: 0.00043, 32: 0.00012, 34: 0.000025, 36: 0.0000028},
    10: {1: 0.500, 3: 0.431, 5: 0.364, 7: 0.300, 9: 0.242, 11: 0.190, 13: 0.146,
         15: 0.108, 17: 0.078, 19: 0.054, 21: 0.036, 23: 0.023, 25: 0.014, 
         27: 0.0083, 29: 0.0046, 31: 0.0023, 33: 0.0011, 35: 0.00047, 37: 0.00018,
         39: 0.000058, 41: 0.000015, 43: 0.0000028, 45: 0.00000028}
}


@dataclass
class MKTestResult:
    """Result container for Mann-Kendall test."""
    p_value: Optional[float]
    trend: str
    direction: Optional[str]
    n: int
    S: Optional[int]
    Z: Optional[float]
    method: str
    var_s_raw: Optional[float] = None
    var_s_corrected: Optional[float] = None
    correction_factor: Optional[float] = None
    acf_used: Optional[List[Tuple[int, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'p_value': self.p_value,
            'trend': self.trend,
            'direction': self.direction,
            'n': self.n,
            'S': self.S,
            'Z': self.Z,
            'method': self.method,
            'var_s_raw': self.var_s_raw,
            'var_s_corrected': self.var_s_corrected,
            'correction_factor': self.correction_factor,
            'acf_used': self.acf_used
        }


# =============================================================================
# Core Functions
# =============================================================================

def compute_s_statistic(x: np.ndarray, eps: float = 0.0) -> int:
    """
    Compute the Mann-Kendall S statistic.
    
    S = sum_{i<j} sign(x_j - x_i)
    
    Args:
        x: Input time series array
        eps: Tolerance for treating small differences as zero (for numerical stability)
    
    Returns:
        S statistic (integer)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    S = 0
    
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        if eps > 0:
            S += np.sum(np.where(np.abs(diff) <= eps, 0, np.sign(diff)))
        else:
            S += int(np.sign(diff).sum())
    
    return int(S)


def compute_variance_with_ties(x: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute variance of S statistic accounting for ties.
    
    Var(S) = [n(n-1)(2n+5) - sum_t(t)(t-1)(2t+5)] / 18
    
    Args:
        x: Input time series array
    
    Returns:
        Tuple of (variance, tie_counts)
    """
    x = np.asarray(x)
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    
    # Tie correction term
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    
    return float(variance), counts


def _lookup_exact_p_value(n: int, s_abs: int) -> Optional[Tuple[float, int]]:
    """
    Look up exact p-value from small sample table.
    
    Args:
        n: Sample size
        s_abs: Absolute value of S statistic
    
    Returns:
        Tuple of (p_value, aligned_s) or None if not available
    """
    table = MK_SMALL_SAMPLE_TABLE.get(n)
    if table is None:
        return None
    
    M = n * (n - 1) // 2  # Maximum possible S
    
    # Align parity (S and M must have same parity)
    if (s_abs % 2) != (M % 2):
        s_abs -= 1
    
    base = 0 if (M % 2 == 0) else 1
    s_abs = max(s_abs, base)
    
    # Find the largest key <= s_abs with matching parity
    valid_keys = sorted(k for k in table if (k % 2) == (M % 2))
    k_use = max([k for k in valid_keys if k <= s_abs], default=valid_keys[0])
    
    return float(table[k_use]), int(s_abs)


@lru_cache(maxsize=None)
def _critical_correlation(n_eff: int, alpha: float) -> float:
    """Compute critical correlation value for significance testing."""
    df = max(3, int(n_eff) - 2)
    t_crit = tdist.ppf(1 - alpha, df)  # One-tailed
    return t_crit / np.sqrt(t_crit ** 2 + df)


def hr98_variance_correction(
    x: np.ndarray,
    var_raw: float,
    slope: Optional[float] = None,
    alpha_acf: float = 0.05,
    max_lag: Optional[int] = None,
    use_rank: bool = True
) -> Tuple[float, float, List[Tuple[int, float]]]:
    """
    Apply Hamed & Rao (1998) variance correction for autocorrelation.
    
    The correction inflates the variance when significant positive autocorrelation
    is detected, making the test more conservative.
    
    Args:
        x: Input time series array
        var_raw: Raw variance of S (without correction)
        slope: Estimated slope for detrending (if None, computed internally using Sen's slope)
        alpha_acf: Significance level for autocorrelation testing (one-tailed)
        max_lag: Maximum lag to consider (if None, determined by sample size)
        use_rank: Whether to use rank-based autocorrelation
    
    Returns:
        Tuple of (corrected_variance, correction_factor, list of (lag, rho) used)
    """
    n = len(x)
    
    # No correction for small samples or invalid variance
    if n <= 10 or var_raw <= 0:
        return var_raw, 1.0, []
    
    # Compute slope if not provided (using Sen's slope)
    if slope is None:
        from .sens_slope import compute_sens_slope
        slope = compute_sens_slope(x)
    
    # Detrend the series
    t = np.arange(n, dtype=float)
    residuals = x - slope * t
    
    # Use ranks for robustness if specified
    x_acf = rankdata(residuals, method="average") if use_rank else residuals
    
    # Determine lag range based on sample size
    if max_lag is None:
        if n < 14:
            max_lag = 0
        elif n < 21:
            max_lag = 1
        else:
            max_lag = int(min(10 * np.log10(n), n // 4))
    else:
        max_lag = min(int(max_lag), n // 2)
    
    # Compute correction factor
    gamma = 0.0
    acf_used = []
    
    for k in range(1, max_lag + 1):
        m = n - k
        if m < 5:
            break
        
        # Compute lag-k autocorrelation
        a, b = x_acf[:m], x_acf[k:]
        a_mean, b_mean = a.mean(), b.mean()
        da, db = a - a_mean, b - b_mean
        
        denominator = np.sqrt(np.dot(da, da) * np.dot(db, db))
        if denominator == 0:
            continue
        
        rho = float(np.dot(da, db) / denominator)
        if not np.isfinite(rho):
            continue
        
        # Only include significant positive autocorrelations
        if rho > _critical_correlation(m, alpha_acf):
            gamma += m * (m - 1) * (m - 2) * rho
            acf_used.append((k, float(rho)))
    
    # Compute correction factor (minimum 1.0)
    C = max(1.0, 1.0 + 2.0 * gamma / (n * (n - 1) * (n - 2)))
    var_corrected = var_raw * C
    
    return float(var_corrected), float(C), acf_used


# =============================================================================
# Main Test Function
# =============================================================================

def mann_kendall_test(
    series: np.ndarray,
    alpha: float = 0.05,
    use_hr98: bool = True,
    alpha_acf: float = 0.05,
    max_lag: Optional[int] = None,
    use_exact_table: bool = True,
    eps: float = 0.0
) -> MKTestResult:
    """
    Perform Mann-Kendall trend test with optional HR98 correction.
    
    Decision logic (see flowchart):
    - n < 4: Insufficient data
    - 4 <= n <= 10 (no ties): Use exact table lookup
    - n >= 11: Use normal approximation
      - 10 <= n <= 13: No HR98 correction
      - n >= 14: HR98 correction optional
        - 14 <= n <= 20: Only check lag-1
        - n >= 21: Use empirical lag selection
    
    Args:
        series: Input time series array
        alpha: Significance level (default 0.05)
        use_hr98: Whether to apply HR98 autocorrelation correction
        alpha_acf: Significance level for ACF testing in HR98
        max_lag: Maximum lag for HR98 (None for automatic)
        use_exact_table: Whether to use exact table for small samples
        eps: Tolerance for numerical stability
    
    Returns:
        MKTestResult object containing test results
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)
    
    # Validate input
    if n < 4 or not np.all(np.isfinite(x)):
        return MKTestResult(
            p_value=None, trend='insufficient_data', direction=None,
            n=n, S=None, Z=None, method='invalid_input'
        )
    
    # Compute S statistic
    S = compute_s_statistic(x, eps=eps)
    
    # Handle degenerate case (no trend signal)
    if S == 0:
        return MKTestResult(
            p_value=1.0, trend='no_trend', direction=None,
            n=n, S=0, Z=0.0, method='degenerate'
        )
    
    direction = 'increasing' if S > 0 else 'decreasing'
    s_abs = abs(S)
    
    # Compute variance
    var_raw, tie_counts = compute_variance_with_ties(x)
    has_ties = np.any(tie_counts > 1)
    
    # ========== Small Sample Path (Exact Table) ==========
    if use_exact_table and (4 <= n <= 10) and not has_ties:
        result = _lookup_exact_p_value(n, s_abs)
        if result is not None:
            p_value, _ = result
            
            if direction == 'increasing' and p_value < alpha:
                trend = 'significant_increase'
            elif direction == 'decreasing' and p_value < alpha:
                trend = 'significant_decrease'
            else:
                trend = 'no_trend'
            
            return MKTestResult(
                p_value=p_value, trend=trend, direction=direction,
                n=n, S=S, Z=None, method='exact_table',
                var_s_raw=var_raw
            )
    
    # ========== Normal Approximation Path ==========
    var_corrected = var_raw
    correction_factor = 1.0
    acf_used = []
    
    # Apply HR98 correction for larger samples
    if use_hr98 and n >= 14:
        var_corrected, correction_factor, acf_used = hr98_variance_correction(
            x, var_raw, alpha_acf=alpha_acf, max_lag=max_lag
        )
    
    # Compute Z statistic with continuity correction
    if var_corrected <= 0:
        return MKTestResult(
            p_value=1.0, trend='no_trend', direction=direction,
            n=n, S=S, Z=None, method='degenerate_variance',
            var_s_raw=var_raw
        )
    
    Z = (s_abs - 1) / np.sqrt(var_corrected)
    p_value = float(np.clip(1 - norm.cdf(Z), 0.0, 1.0))  # Right-tail
    
    # Determine trend
    if direction == 'increasing' and p_value < alpha:
        trend = 'significant_increase'
    elif direction == 'decreasing' and p_value < alpha:
        trend = 'significant_decrease'
    else:
        trend = 'no_significant_trend'
    
    method = 'normal_hr98' if use_hr98 and n >= 14 else 'normal'
    
    return MKTestResult(
        p_value=p_value, trend=trend, direction=direction,
        n=n, S=S, Z=Z, method=method,
        var_s_raw=var_raw, var_s_corrected=var_corrected,
        correction_factor=correction_factor, acf_used=acf_used
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def detect_trend(
    series: np.ndarray,
    alpha: float = 0.05,
    use_hr98: bool = True
) -> Dict[str, Any]:
    """
    Simple wrapper for trend detection.
    
    Args:
        series: Input time series
        alpha: Significance level
        use_hr98: Whether to use HR98 correction
    
    Returns:
        Dictionary with 'has_trend', 'direction', 'p_value', 'significant'
    """
    result = mann_kendall_test(series, alpha=alpha, use_hr98=use_hr98)
    
    return {
        'has_trend': result.trend != 'no_trend' and result.trend != 'insufficient_data',
        'direction': result.direction,
        'p_value': result.p_value,
        'significant': result.p_value is not None and result.p_value < alpha,
        'method': result.method
    }


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample data with trend
    n = 30
    t = np.arange(n)
    trend = 0.5 * t
    noise = np.random.normal(0, 2, n)
    data = trend + noise
    
    # Run test
    result = mann_kendall_test(data, alpha=0.05, use_hr98=True)
    
    print("Mann-Kendall Test Results:")
    print(f"  Sample size: {result.n}")
    print(f"  S statistic: {result.S}")
    print(f"  Z statistic: {result.Z:.4f}" if result.Z else "  Z statistic: N/A")
    print(f"  p-value: {result.p_value:.4f}" if result.p_value else "  p-value: N/A")
    print(f"  Trend: {result.trend}")
    print(f"  Direction: {result.direction}")
    print(f"  Method: {result.method}")
    
    if result.correction_factor and result.correction_factor > 1:
        print(f"  HR98 correction factor: {result.correction_factor:.4f}")
