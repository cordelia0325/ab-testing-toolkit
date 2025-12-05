"""
OLS Trend Detection with Newey-West HAC Standard Errors

This module implements ordinary least squares regression for trend detection,
with heteroskedasticity and autocorrelation consistent (HAC) standard errors
using the Newey-West estimator.

References:
    - Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, 
      heteroskedasticity and autocorrelation consistent covariance matrix.
      Econometrica.
    - Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation 
      consistent covariance matrix estimation. Econometrica.

Author: Chuqiao Huang
"""

import numpy as np
from scipy.stats import t as tdist
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class OLSResult:
    """Result container for OLS trend test."""
    slope: float
    intercept: float
    slope_se: float
    t_statistic: float
    p_value: float
    trend: str
    direction: str
    conf_interval: Tuple[float, float]
    r_squared: float
    n: int
    method: str
    lag: Optional[int] = None
    
    def to_dict(self):
        return {
            'slope': self.slope,
            'intercept': self.intercept,
            'slope_se': self.slope_se,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'trend': self.trend,
            'direction': self.direction,
            'conf_interval': self.conf_interval,
            'r_squared': self.r_squared,
            'n': self.n,
            'method': self.method,
            'lag': self.lag
        }


def _bartlett_kernel(j: int, lag: int) -> float:
    """
    Bartlett kernel weight for Newey-West estimator.
    
    w(j) = 1 - j/(lag+1) for j <= lag, 0 otherwise
    """
    if j > lag:
        return 0.0
    return 1.0 - j / (lag + 1)


def _select_lag(n: int, method: str = 'nw') -> int:
    """
    Select optimal lag for Newey-West estimator.
    
    Args:
        n: Sample size
        method: 'nw' for Newey-West rule, 'andrews' for Andrews rule
    
    Returns:
        Optimal lag
    """
    if method == 'nw':
        # Newey-West rule: floor(4 * (n/100)^(2/9))
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    else:
        # Andrews rule: floor(n^(1/3))
        lag = int(np.floor(n ** (1 / 3)))
    
    return max(0, min(lag, n - 2))


def ols_fit(y: np.ndarray, x: Optional[np.ndarray] = None) -> Tuple[float, float, np.ndarray]:
    """
    Fit OLS regression: y = intercept + slope * x
    
    Args:
        y: Dependent variable (time series)
        x: Independent variable (time index). If None, uses 0, 1, 2, ...
    
    Returns:
        Tuple of (intercept, slope, residuals)
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    
    if x is None:
        x = np.arange(n, dtype=np.float64)
    else:
        x = np.asarray(x, dtype=np.float64)
    
    # Add constant term
    X = np.column_stack([np.ones(n), x])
    
    # OLS: beta = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    
    intercept, slope = beta[0], beta[1]
    residuals = y - X @ beta
    
    return float(intercept), float(slope), residuals


def newey_west_se(
    residuals: np.ndarray,
    x: np.ndarray,
    lag: Optional[int] = None
) -> Tuple[float, int]:
    """
    Compute Newey-West HAC standard error for slope coefficient.
    
    Args:
        residuals: OLS residuals
        x: Independent variable (time index)
        lag: Number of lags (if None, uses automatic selection)
    
    Returns:
        Tuple of (standard_error_for_slope, lag_used)
    """
    n = len(residuals)
    
    if lag is None:
        lag = _select_lag(n)
    
    # Construct design matrix
    X = np.column_stack([np.ones(n), x])
    
    # Compute (X'X)^{-1}
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Compute S_0 = sum of u_i^2 * x_i * x_i'
    S = np.zeros((2, 2))
    
    # Lag-0 term
    for i in range(n):
        xi = X[i].reshape(-1, 1)
        S += (residuals[i] ** 2) * (xi @ xi.T)
    
    # Lag-j terms (j = 1, ..., lag)
    for j in range(1, lag + 1):
        w = _bartlett_kernel(j, lag)
        Gamma_j = np.zeros((2, 2))
        
        for i in range(j, n):
            xi = X[i].reshape(-1, 1)
            xi_j = X[i - j].reshape(-1, 1)
            Gamma_j += residuals[i] * residuals[i - j] * (xi @ xi_j.T)
        
        # Add both Gamma_j and Gamma_j' (for symmetry)
        S += w * (Gamma_j + Gamma_j.T)
    
    # HAC covariance matrix
    V_hac = XtX_inv @ S @ XtX_inv
    
    # Standard error for slope (second diagonal element)
    slope_se = np.sqrt(V_hac[1, 1])
    
    return float(slope_se), lag


def ols_standard_se(residuals: np.ndarray, x: np.ndarray) -> float:
    """
    Compute standard OLS standard error (assuming homoskedasticity).
    
    Args:
        residuals: OLS residuals
        x: Independent variable
    
    Returns:
        Standard error for slope
    """
    n = len(residuals)
    
    # Estimate sigma^2
    sigma2 = np.sum(residuals ** 2) / (n - 2)
    
    # Var(beta_1) = sigma^2 / sum((x - x_mean)^2)
    x_centered = x - x.mean()
    var_slope = sigma2 / np.sum(x_centered ** 2)
    
    return float(np.sqrt(var_slope))


def ols_trend_test(
    y: np.ndarray,
    alpha: float = 0.05,
    use_hac: bool = True,
    lag: Optional[int] = None
) -> OLSResult:
    """
    Perform OLS-based trend test with optional HAC standard errors.
    
    Tests H0: slope = 0 vs H1: slope != 0
    
    Args:
        y: Time series data
        alpha: Significance level
        use_hac: Whether to use Newey-West HAC standard errors
        lag: Lag for HAC (if None, uses automatic selection)
    
    Returns:
        OLSResult object
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    
    if n < 4:
        return OLSResult(
            slope=float('nan'), intercept=float('nan'),
            slope_se=float('nan'), t_statistic=float('nan'),
            p_value=float('nan'), trend='insufficient_data',
            direction='none', conf_interval=(float('nan'), float('nan')),
            r_squared=float('nan'), n=n, method='invalid'
        )
    
    # Time index
    x = np.arange(n, dtype=np.float64)
    
    # Fit OLS
    intercept, slope, residuals = ols_fit(y, x)
    
    # Compute standard error
    if use_hac:
        slope_se, lag_used = newey_west_se(residuals, x, lag)
        method = f'ols_hac_lag{lag_used}'
    else:
        slope_se = ols_standard_se(residuals, x)
        lag_used = None
        method = 'ols_standard'
    
    # t-statistic and p-value
    df = n - 2
    t_stat = slope / slope_se if slope_se > 0 else float('inf') * np.sign(slope)
    p_value = float(2 * (1 - tdist.cdf(abs(t_stat), df)))  # Two-tailed
    
    # Confidence interval
    t_crit = tdist.ppf(1 - alpha / 2, df)
    ci_low = slope - t_crit * slope_se
    ci_high = slope + t_crit * slope_se
    
    # R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # Determine trend
    direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'none'
    
    if p_value < alpha:
        trend = 'significant_increase' if slope > 0 else 'significant_decrease'
    else:
        trend = 'no_significant_trend'
    
    return OLSResult(
        slope=slope,
        intercept=intercept,
        slope_se=slope_se,
        t_statistic=t_stat,
        p_value=p_value,
        trend=trend,
        direction=direction,
        conf_interval=(ci_low, ci_high),
        r_squared=r_squared,
        n=n,
        method=method,
        lag=lag_used
    )


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate autocorrelated data with trend
    n = 50
    true_slope = 0.1
    t = np.arange(n)
    
    # AR(1) errors
    rho = 0.7
    errors = np.zeros(n)
    errors[0] = np.random.normal(0, 1)
    for i in range(1, n):
        errors[i] = rho * errors[i-1] + np.random.normal(0, 1)
    
    data = 5.0 + true_slope * t + errors
    
    # Test with standard OLS
    result_std = ols_trend_test(data, use_hac=False)
    print("Standard OLS Results:")
    print(f"  Slope: {result_std.slope:.4f} (true: {true_slope})")
    print(f"  SE: {result_std.slope_se:.4f}")
    print(f"  p-value: {result_std.p_value:.4f}")
    print(f"  Trend: {result_std.trend}")
    
    print()
    
    # Test with Newey-West HAC
    result_hac = ols_trend_test(data, use_hac=True)
    print("OLS with Newey-West HAC:")
    print(f"  Slope: {result_hac.slope:.4f} (true: {true_slope})")
    print(f"  SE: {result_hac.slope_se:.4f}")
    print(f"  p-value: {result_hac.p_value:.4f}")
    print(f"  Trend: {result_hac.trend}")
    print(f"  Lag used: {result_hac.lag}")
