"""
CUPED: Controlled-experiment Using Pre-Experiment Data

A variance reduction technique that uses pre-experiment data as a control
variate to reduce the variance of treatment effect estimates.

The key insight is that if we have a covariate X (e.g., pre-experiment metric)
that is correlated with the outcome Y but unaffected by the treatment,
we can use it to "explain away" some of the variance in Y.

References:
    - Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the 
      Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment 
      Data. WSDM '13.
    - Xie, H., & Aurisset, J. (2016). Improving the Sensitivity of Online 
      Controlled Experiments: Case Studies at Netflix.

Author: Chuqiao Huang
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from scipy import stats


@dataclass
class CUPEDResult:
    """Result container for CUPED analysis."""
    # Original estimates
    mean_treatment: float
    mean_control: float
    ate_raw: float  # Average Treatment Effect (raw)
    var_raw: float
    se_raw: float
    
    # CUPED-adjusted estimates
    ate_cuped: float
    var_cuped: float
    se_cuped: float
    
    # Variance reduction
    theta: float  # Adjustment coefficient
    variance_reduction: float  # 1 - var_cuped/var_raw
    
    # Statistical inference
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    
    # Sample sizes
    n_treatment: int
    n_control: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_treatment': self.mean_treatment,
            'mean_control': self.mean_control,
            'ate_raw': self.ate_raw,
            'var_raw': self.var_raw,
            'se_raw': self.se_raw,
            'ate_cuped': self.ate_cuped,
            'var_cuped': self.var_cuped,
            'se_cuped': self.se_cuped,
            'theta': self.theta,
            'variance_reduction': self.variance_reduction,
            't_stat': self.t_stat,
            'p_value': self.p_value,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_treatment': self.n_treatment,
            'n_control': self.n_control,
        }
    
    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "CUPED Analysis Results",
            "=" * 50,
            f"Sample sizes: Treatment={self.n_treatment}, Control={self.n_control}",
            "",
            "Raw Estimates:",
            f"  ATE (raw): {self.ate_raw:.6f}",
            f"  SE (raw):  {self.se_raw:.6f}",
            "",
            "CUPED-Adjusted Estimates:",
            f"  ATE (CUPED): {self.ate_cuped:.6f}",
            f"  SE (CUPED):  {self.se_cuped:.6f}",
            f"  theta:       {self.theta:.6f}",
            "",
            f"Variance Reduction: {self.variance_reduction*100:.2f}%",
            "",
            "Statistical Inference (CUPED):",
            f"  t-statistic: {self.t_stat:.4f}",
            f"  p-value:     {self.p_value:.4f}",
            f"  95% CI:      [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
        ]
        return "\n".join(lines)


def compute_theta(
    y: np.ndarray,
    x: np.ndarray,
    pooled: bool = True
) -> float:
    """
    Compute the optimal adjustment coefficient theta.
    
    theta = Cov(Y, X) / Var(X)
    
    This minimizes Var(Y - theta * X).
    
    Args:
        y: Post-experiment outcome
        x: Pre-experiment covariate
        pooled: If True, compute theta from pooled data (recommended)
    
    Returns:
        Optimal theta coefficient
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    # Remove NaN pairs
    mask = ~(np.isnan(y) | np.isnan(x))
    y = y[mask]
    x = x[mask]
    
    if len(y) < 2:
        return 0.0
    
    var_x = np.var(x, ddof=1)
    if var_x < 1e-10:
        return 0.0
    
    cov_xy = np.cov(y, x, ddof=1)[0, 1]
    theta = cov_xy / var_x
    
    return float(theta)


def cuped_adjust(
    y: np.ndarray,
    x: np.ndarray,
    theta: Optional[float] = None,
    x_mean: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """
    Apply CUPED adjustment to outcome variable.
    
    Y_adjusted = Y - theta * (X - E[X])
    
    Args:
        y: Post-experiment outcome
        x: Pre-experiment covariate
        theta: Adjustment coefficient (if None, computed from data)
        x_mean: Mean of X (if None, computed from data)
    
    Returns:
        Tuple of (adjusted_y, theta_used)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    if theta is None:
        theta = compute_theta(y, x)
    
    if x_mean is None:
        x_mean = np.nanmean(x)
    
    y_adjusted = y - theta * (x - x_mean)
    
    return y_adjusted, theta


def cuped_two_sample(
    y_treatment: np.ndarray,
    y_control: np.ndarray,
    x_treatment: np.ndarray,
    x_control: np.ndarray,
    alpha: float = 0.05,
    pooled_theta: bool = True
) -> CUPEDResult:
    """
    Perform CUPED analysis for a two-sample A/B test.
    
    This is the main function for applying CUPED to reduce variance
    in the treatment effect estimate.
    
    Args:
        y_treatment: Post-experiment outcomes for treatment group
        y_control: Post-experiment outcomes for control group
        x_treatment: Pre-experiment covariates for treatment group
        x_control: Pre-experiment covariates for control group
        alpha: Significance level for confidence intervals
        pooled_theta: If True, compute theta from pooled data (recommended)
    
    Returns:
        CUPEDResult object with all analysis results
    """
    # Convert to arrays
    y_t = np.asarray(y_treatment, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    x_t = np.asarray(x_treatment, dtype=float)
    x_c = np.asarray(x_control, dtype=float)
    
    # Remove NaN
    mask_t = ~(np.isnan(y_t) | np.isnan(x_t))
    mask_c = ~(np.isnan(y_c) | np.isnan(x_c))
    y_t, x_t = y_t[mask_t], x_t[mask_t]
    y_c, x_c = y_c[mask_c], x_c[mask_c]
    
    n_t, n_c = len(y_t), len(y_c)
    
    # Raw estimates
    mean_t = np.mean(y_t)
    mean_c = np.mean(y_c)
    ate_raw = mean_t - mean_c
    
    var_t = np.var(y_t, ddof=1)
    var_c = np.var(y_c, ddof=1)
    var_raw = var_t / n_t + var_c / n_c
    se_raw = np.sqrt(var_raw)
    
    # Compute theta (pooled or separate)
    if pooled_theta:
        y_pooled = np.concatenate([y_t, y_c])
        x_pooled = np.concatenate([x_t, x_c])
        theta = compute_theta(y_pooled, x_pooled)
        x_mean = np.mean(x_pooled)
    else:
        # Separate theta for each group (less common)
        theta_t = compute_theta(y_t, x_t)
        theta_c = compute_theta(y_c, x_c)
        theta = (theta_t + theta_c) / 2
        x_mean = (np.mean(x_t) * n_t + np.mean(x_c) * n_c) / (n_t + n_c)
    
    # CUPED adjustment
    y_t_adj = y_t - theta * (x_t - x_mean)
    y_c_adj = y_c - theta * (x_c - x_mean)
    
    # CUPED estimates
    mean_t_adj = np.mean(y_t_adj)
    mean_c_adj = np.mean(y_c_adj)
    ate_cuped = mean_t_adj - mean_c_adj
    
    var_t_adj = np.var(y_t_adj, ddof=1)
    var_c_adj = np.var(y_c_adj, ddof=1)
    var_cuped = var_t_adj / n_t + var_c_adj / n_c
    se_cuped = np.sqrt(var_cuped)
    
    # Variance reduction
    variance_reduction = 1 - var_cuped / var_raw if var_raw > 0 else 0.0
    
    # Statistical inference (using CUPED estimates)
    t_stat = ate_cuped / se_cuped if se_cuped > 0 else 0.0
    
    # Welch's degrees of freedom
    df = (var_t_adj/n_t + var_c_adj/n_c)**2 / (
        (var_t_adj/n_t)**2/(n_t-1) + (var_c_adj/n_c)**2/(n_c-1)
    )
    
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = ate_cuped - t_crit * se_cuped
    ci_upper = ate_cuped + t_crit * se_cuped
    
    return CUPEDResult(
        mean_treatment=mean_t,
        mean_control=mean_c,
        ate_raw=ate_raw,
        var_raw=var_raw,
        se_raw=se_raw,
        ate_cuped=ate_cuped,
        var_cuped=var_cuped,
        se_cuped=se_cuped,
        theta=theta,
        variance_reduction=variance_reduction,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treatment=n_t,
        n_control=n_c
    )


def cuped_from_dataframe(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    group_col: str,
    treatment_value: Any = 1,
    control_value: Any = 0,
    alpha: float = 0.05
) -> CUPEDResult:
    """
    Convenience function to run CUPED from a DataFrame.
    
    Args:
        df: DataFrame containing experiment data
        y_col: Column name for post-experiment outcome
        x_col: Column name for pre-experiment covariate
        group_col: Column name for treatment assignment
        treatment_value: Value indicating treatment group
        control_value: Value indicating control group
        alpha: Significance level
    
    Returns:
        CUPEDResult object
    
    Example:
        >>> result = cuped_from_dataframe(
        ...     df, 
        ...     y_col='revenue_post',
        ...     x_col='revenue_pre',
        ...     group_col='treatment',
        ...     treatment_value=1,
        ...     control_value=0
        ... )
    """
    treatment_mask = df[group_col] == treatment_value
    control_mask = df[group_col] == control_value
    
    return cuped_two_sample(
        y_treatment=df.loc[treatment_mask, y_col].values,
        y_control=df.loc[control_mask, y_col].values,
        x_treatment=df.loc[treatment_mask, x_col].values,
        x_control=df.loc[control_mask, x_col].values,
        alpha=alpha
    )


def estimate_sample_size_reduction(
    y: np.ndarray,
    x: np.ndarray
) -> Dict[str, float]:
    """
    Estimate potential sample size reduction from using CUPED.
    
    If CUPED reduces variance by factor (1 - r^2), then the same
    precision can be achieved with (1 - r^2) times the sample size.
    
    Args:
        y: Post-experiment outcome
        x: Pre-experiment covariate
    
    Returns:
        Dict with correlation, variance reduction, and sample size reduction
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    
    if len(y) < 2:
        return {'correlation': 0.0, 'variance_reduction': 0.0, 'sample_size_reduction': 0.0}
    
    # Correlation between Y and X
    corr = np.corrcoef(y, x)[0, 1]
    
    # Theoretical variance reduction = r^2
    var_reduction = corr ** 2
    
    # Sample size reduction (same precision with fewer samples)
    sample_reduction = var_reduction
    
    return {
        'correlation': float(corr),
        'variance_reduction': float(var_reduction),
        'sample_size_reduction': float(sample_reduction)
    }


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    n_treatment = 1000
    n_control = 1000
    
    # Generate correlated pre/post data
    # Pre-experiment metric (same distribution for both groups)
    x_treatment = np.random.normal(100, 20, n_treatment)
    x_control = np.random.normal(100, 20, n_control)
    
    # Post-experiment metric (treatment has +5 effect)
    # Y = 0.8 * X + noise + treatment_effect
    noise_t = np.random.normal(0, 15, n_treatment)
    noise_c = np.random.normal(0, 15, n_control)
    
    treatment_effect = 5.0
    y_treatment = 0.8 * x_treatment + noise_t + treatment_effect
    y_control = 0.8 * x_control + noise_c
    
    # Run CUPED analysis
    result = cuped_two_sample(
        y_treatment=y_treatment,
        y_control=y_control,
        x_treatment=x_treatment,
        x_control=x_control,
        alpha=0.05
    )
    
    print(result.summary())
    print()
    print(f"True treatment effect: {treatment_effect}")
    
    # Estimate potential sample size reduction
    y_all = np.concatenate([y_treatment, y_control])
    x_all = np.concatenate([x_treatment, x_control])
    reduction = estimate_sample_size_reduction(y_all, x_all)
    print(f"\nPotential sample size reduction: {reduction['sample_size_reduction']*100:.1f}%")
