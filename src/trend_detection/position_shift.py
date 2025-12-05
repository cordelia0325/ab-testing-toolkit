"""
Hodges-Lehmann Position Shift Estimator

A robust, non-parametric method for detecting level shifts between two time windows.
Uses the Hodges-Lehmann estimator (median of all pairwise differences) to measure
the shift between a baseline period (B) and a study period (S).

Key Features:
    - Two-sample Hodges-Lehmann shift estimation
    - Flexible window splitting (symmetric/asymmetric)
    - Bootstrap confidence intervals
    - Signal-to-noise ratio (SNR) calculation
    - Business threshold comparison

References:
    - Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based on 
      rank tests. Annals of Mathematical Statistics.
    - Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric 
      Statistical Methods (3rd ed.). Wiley.

Author: Chuqiao Huang
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PositionShiftResult:
    """Result container for position shift analysis."""
    ok: bool
    reason: str
    
    # Window information
    B_range: Tuple[Optional[str], Optional[str]]
    S_range: Tuple[Optional[str], Optional[str]]
    n_B: int
    n_S: int
    
    # Core statistics
    med_B: float
    med_S: float
    med_diff: float
    hl_shift: float
    direction: str
    
    # Scale and threshold
    sigma: float
    snr: float
    level_threshold: float
    pass_threshold: bool
    
    # Optional: Bootstrap CI
    hl_ci: Optional[Tuple[float, float]] = None
    hl_conf_level: Optional[float] = None
    hl_trend: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'ok': self.ok,
            'reason': self.reason,
            'B_range': self.B_range,
            'S_range': self.S_range,
            'n_B': self.n_B,
            'n_S': self.n_S,
            'med_B': self.med_B,
            'med_S': self.med_S,
            'med_diff': self.med_diff,
            'hl_shift': self.hl_shift,
            'direction': self.direction,
            'sigma': self.sigma,
            'snr': self.snr,
            'level_threshold': self.level_threshold,
            'pass_threshold': self.pass_threshold,
        }
        if self.hl_ci is not None:
            result['hl_ci'] = self.hl_ci
            result['hl_conf_level'] = self.hl_conf_level
            result['hl_trend'] = self.hl_trend
        return result


# ========== Window Splitting ==========

def split_half(s: pd.Series) -> Dict[str, pd.Series]:
    """
    Split a time series into two halves.
    
    Args:
        s: Input series (sorted in ascending order)
    
    Returns:
        Dict with 'B' (first half) and 'S' (second half)
    """
    n = len(s)
    mid = n // 2
    return {"B": s.iloc[:mid], "S": s.iloc[mid:]}


def split_windows(
    series: pd.Series,
    end_date: Optional[pd.Timestamp] = None,
    win: Optional[int] = None,
    gap: int = 0,
    win_S: Optional[int] = None,
    win_B: Optional[int] = None
) -> Dict[str, Any]:
    """
    Split time series into baseline (B) and study (S) windows.
    
    Supports two calling conventions:
        1) Symmetric: split_windows(series, end_date, win=7, gap=0)
        2) Asymmetric: split_windows(series, end_date, win_S=7, win_B=14, gap=0)
    
    Window layout (time flows left to right):
        [...older data...][B window][gap][S window]
                                              ^end_date
    
    Args:
        series: Input time series with datetime index
        end_date: End date of S window (default: last date in series)
        win: Window length for both B and S (symmetric mode)
        gap: Number of days gap between B and S windows
        win_S: Length of study window (asymmetric mode)
        win_B: Length of baseline window (asymmetric mode)
    
    Returns:
        Dict containing:
            - B: Baseline window series
            - S: Study window series
            - B_range: (start_date, end_date) of B window
            - S_range: (start_date, end_date) of S window
            - removed_dates: Dates not included in either window
    """
    # Clean and prepare series
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()]
    s = s.sort_index()

    if s.empty:
        return dict(
            B=s, S=s,
            B_range=(None, None),
            S_range=(None, None),
            removed_dates=[]
        )

    # Parse window lengths
    if win_S is None and win_B is None:
        win = 7 if win is None else int(win)
        win_S = win_B = win
    else:
        # If only one is given, use it; fallback to win; then default to 7
        win_S = int(win_S if win_S is not None else (win if win is not None else 7))
        win_B = int(win_B if win_B is not None else (win if win is not None else 7))

    # Determine S window end date
    S_end = (s.index.max().normalize() if end_date is None
             else pd.to_datetime(end_date).normalize())
    
    # S window (study period)
    S_start = S_end - pd.Timedelta(days=win_S - 1)
    S = s.loc[(s.index.normalize() >= S_start) & (s.index.normalize() <= S_end)]

    # B window (baseline period, with gap before S)
    B_end = S_start - pd.Timedelta(days=gap + 1)
    B_start = B_end - pd.Timedelta(days=win_B - 1)
    B = s.loc[(s.index.normalize() >= B_start) & (s.index.normalize() <= B_end)]

    # Track dates that were truncated (not in B or S)
    kept = B.index.union(S.index)
    removed_dates = list(s.index.difference(kept))

    return dict(
        B=B, S=S,
        B_range=(B_start.date() if pd.notna(B_start) else None,
                 B_end.date() if pd.notna(B_end) else None),
        S_range=(S_start.date() if pd.notna(S_start) else None,
                 S_end.date() if pd.notna(S_end) else None),
        removed_dates=removed_dates
    )


# ========== Hodges-Lehmann Estimator ==========

def hodges_lehmann_2sample(S: np.ndarray, B: np.ndarray) -> float:
    """
    Two-sample Hodges-Lehmann shift estimator.
    
    Computes the median of all pairwise differences (s_i - b_j),
    where s_i in S and b_j in B.
    
    Complexity: O(n_S * n_B), which is fast for typical window sizes (7x7 or 14x14).
    
    Args:
        S: Study period samples
        B: Baseline period samples
    
    Returns:
        Hodges-Lehmann shift estimate (median of all pairwise differences)
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    
    if S.size == 0 or B.size == 0:
        return float('nan')
    
    # Compute all pairwise differences
    diffs = (S[:, None] - B[None, :]).ravel()
    return float(np.median(diffs))


def bootstrap_hl_ci(
    S: np.ndarray,
    B: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
    seed: int = 42
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for Hodges-Lehmann shift.
    
    Resamples S and B independently with replacement, computes HL for each
    bootstrap sample, and returns the (alpha/2, 1-alpha/2) quantiles.
    
    Args:
        S: Study period samples
        B: Baseline period samples
        alpha: Significance level (default 0.05 for 95% CI)
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    nS, nB = S.size, B.size
    
    if nS < 2 or nB < 2:
        return (float('nan'), float('nan'))
    
    rng = default_rng(seed)
    boots = np.empty(n_bootstrap, dtype=float)
    
    for t in range(n_bootstrap):
        S_resample = S[rng.integers(0, nS, nS)]
        B_resample = B[rng.integers(0, nB, nB)]
        boots[t] = hodges_lehmann_2sample(S_resample, B_resample)
    
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (lo, hi)


# ========== Robust Scale Estimation ==========

def robust_sigma(
    series: pd.Series,
    end_date: Optional[pd.Timestamp] = None,
    days: int = 30
) -> float:
    """
    Estimate robust scale (sigma) using MAD from the tail of a time series.
    
    sigma = 1.4826 * MAD, where MAD = median(|x_i - median(x)|)
    
    The factor 1.4826 makes MAD consistent with standard deviation for normal data.
    
    Args:
        series: Input time series
        end_date: End date for the estimation window (default: last date)
        days: Number of days to include in estimation
    
    Returns:
        Robust sigma estimate
    """
    s = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(s.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        s.index = pd.to_datetime(s.index, errors='coerce')
        s = s[~s.index.isna()]
    s = s.sort_index()
    
    if s.empty:
        return float('nan')
    
    # Determine end date
    if end_date is None:
        end_ts = s.index.max().normalize()
    else:
        end_ts = pd.to_datetime(end_date).normalize()
    
    # Get tail data
    start_ts = end_ts - pd.Timedelta(days=days - 1)
    tail = s.loc[(s.index.normalize() >= start_ts) & (s.index.normalize() <= end_ts)]
    
    if tail.size < 3:
        return float('nan')
    
    # Compute MAD-based sigma
    med = np.median(tail.values)
    mad = np.median(np.abs(tail.values - med))
    sigma = 1.4826 * mad
    
    return float(sigma) if sigma > 0 else float('nan')


# ========== Main Position Shift Analysis ==========

def position_shift(
    series: pd.Series,
    *,
    end_date: Optional[str] = None,
    win_S: int = 7,
    win_B: int = 7,
    gap: int = 0,
    sigma: Optional[float] = None,
    sigma_days: int = 30,
    biz_threshold: Optional[float] = None,
    use_bootstrap_ci: bool = False,
    ci_alpha: float = 0.05,
    ci_n_bootstrap: int = 2000
) -> PositionShiftResult:
    """
    Analyze position shift between baseline and study windows.
    
    This function compares two time periods using robust non-parametric methods:
        - Hodges-Lehmann shift estimator
        - Median differences
        - Signal-to-noise ratio (SNR)
        - Optional bootstrap confidence intervals
    
    Args:
        series: Input time series with datetime index
        end_date: End date of study window (default: last date in series)
        win_S: Length of study window in days
        win_B: Length of baseline window in days
        gap: Number of days gap between windows
        sigma: Known scale parameter (if None, estimated from data)
        sigma_days: Days to use for sigma estimation
        biz_threshold: Business-defined minimum detectable effect
        use_bootstrap_ci: Whether to compute bootstrap CI for HL shift
        ci_alpha: Significance level for CI (default 0.05 for 95% CI)
        ci_n_bootstrap: Number of bootstrap iterations
    
    Returns:
        PositionShiftResult object containing:
            - ok: Whether analysis succeeded
            - med_B, med_S, med_diff: Median statistics
            - hl_shift: Hodges-Lehmann shift estimate
            - direction: 'up', 'down', or 'flat'
            - sigma: Robust scale estimate
            - snr: Signal-to-noise ratio (|hl_shift| / sigma)
            - level_threshold: max(sigma, biz_threshold)
            - pass_threshold: Whether |hl_shift| >= threshold
            - hl_ci, hl_trend: Bootstrap CI results (if requested)
    """
    # Split into windows
    parts = split_windows(
        series,
        end_date=end_date,
        win_S=win_S,
        win_B=win_B,
        gap=gap
    )
    B_ser = parts['B'].dropna()
    S_ser = parts['S'].dropna()
    B = B_ser.values
    S = S_ser.values
    
    # Default result for failures
    def make_error_result(reason: str) -> PositionShiftResult:
        return PositionShiftResult(
            ok=False, reason=reason,
            B_range=parts['B_range'], S_range=parts['S_range'],
            n_B=int(B.size), n_S=int(S.size),
            med_B=float('nan'), med_S=float('nan'), med_diff=float('nan'),
            hl_shift=float('nan'), direction='unknown',
            sigma=float('nan'), snr=float('nan'),
            level_threshold=float('nan'), pass_threshold=False
        )
    
    # Validate sample sizes
    if B.size < 3 or S.size < 3:
        return make_error_result(
            f'Insufficient data points (B={B.size}, S={S.size}, need >=3 each)'
        )
    
    # Compute median difference and direction
    med_B = float(np.median(B))
    med_S = float(np.median(S))
    med_diff = float(med_S - med_B)
    direction = 'up' if med_diff > 0 else ('down' if med_diff < 0 else 'flat')
    
    # Compute Hodges-Lehmann shift
    hl_shift = hodges_lehmann_2sample(S, B)
    
    # Estimate sigma (robust scale)
    if sigma is None or not np.isfinite(sigma):
        s_end = parts['S_range'][1]
        s_end_ts = pd.to_datetime(s_end) if s_end is not None else None
        sigma = robust_sigma(pd.Series(series), end_date=s_end_ts, days=sigma_days)
    sigma = float(sigma) if np.isfinite(sigma) else float('nan')
    
    # Compute threshold and SNR
    if np.isfinite(sigma):
        level_thr = max(sigma, biz_threshold or 0.0)
        snr = abs(hl_shift) / sigma if sigma > 0 else float('nan')
    else:
        level_thr = biz_threshold if biz_threshold else float('nan')
        snr = float('nan')
    
    pass_thr = (abs(hl_shift) >= level_thr) if np.isfinite(level_thr) else False
    
    # Build result
    result = PositionShiftResult(
        ok=True, reason='',
        B_range=parts['B_range'], S_range=parts['S_range'],
        n_B=int(B.size), n_S=int(S.size),
        med_B=med_B, med_S=med_S, med_diff=med_diff,
        hl_shift=float(hl_shift), direction=direction,
        sigma=sigma,
        snr=float(snr) if np.isfinite(snr) else float('nan'),
        level_threshold=float(level_thr) if np.isfinite(level_thr) else float('nan'),
        pass_threshold=bool(pass_thr)
    )
    
    # Optional: Bootstrap CI
    if use_bootstrap_ci:
        lo, hi = bootstrap_hl_ci(S, B, alpha=ci_alpha, n_bootstrap=ci_n_bootstrap)
        result.hl_ci = (lo, hi)
        result.hl_conf_level = 1 - ci_alpha
        
        # Determine trend based on CI
        if np.isfinite(lo) and lo > 0:
            result.hl_trend = "significant_increase"
        elif np.isfinite(hi) and hi < 0:
            result.hl_trend = "significant_decrease"
        else:
            result.hl_trend = "no_significant_trend"
    
    return result


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate sample data with a level shift
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    # Baseline period: mean=10, Study period: mean=12 (shift of +2)
    baseline = np.random.normal(10, 1, 15)
    study = np.random.normal(12, 1, 15)
    data = pd.Series(np.concatenate([baseline, study]), index=dates)
    
    # Analyze position shift
    result = position_shift(
        data,
        win_S=7,
        win_B=7,
        gap=0,
        use_bootstrap_ci=True,
        ci_alpha=0.05
    )
    
    print("Position Shift Analysis Results:")
    print(f"  Baseline window: {result.B_range} (n={result.n_B})")
    print(f"  Study window: {result.S_range} (n={result.n_S})")
    print(f"  Median B: {result.med_B:.4f}")
    print(f"  Median S: {result.med_S:.4f}")
    print(f"  Median diff: {result.med_diff:.4f}")
    print(f"  HL shift: {result.hl_shift:.4f}")
    print(f"  Direction: {result.direction}")
    print(f"  Sigma: {result.sigma:.4f}")
    print(f"  SNR: {result.snr:.4f}")
    print(f"  Threshold: {result.level_threshold:.4f}")
    print(f"  Pass threshold: {result.pass_threshold}")
    if result.hl_ci:
        print(f"  95% CI: [{result.hl_ci[0]:.4f}, {result.hl_ci[1]:.4f}]")
        print(f"  trend: {result.hl_trend}")
