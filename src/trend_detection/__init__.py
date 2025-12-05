"""
Trend Detection Module

This module provides statistical methods for detecting monotonic trends
in time series data, particularly useful for A/B testing metrics.

Available methods:
    - Mann-Kendall test with HR98 autocorrelation correction
    - Sen's slope estimator (robust, non-parametric)
    - OLS with Newey-West HAC standard errors
    - Hodges-Lehmann position shift estimator
    - Excel-based trend analysis tool

Author: Chuqiao Huang
"""

from .mann_kendall import (
    mann_kendall_test,
    detect_trend,
    compute_s_statistic,
    hr98_variance_correction,
    MKTestResult
)

from .sens_slope import (
    sens_slope,
    compute_sens_slope,
    compute_all_slopes,
    SenSlopeResult
)

from .ols_hac import (
    ols_trend_test,
    ols_fit,
    newey_west_se,
    OLSResult
)

from .position_shift import (
    position_shift,
    PositionShiftResult,
    hodges_lehmann_2sample,
    bootstrap_hl_ci,
    split_windows,
    robust_sigma
)

from .trend_analysis import (
    mk_trend_platform,
    analyze_trend_from_excel
)

from .unified import TrendDetector

__all__ = [
    # Mann-Kendall
    'mann_kendall_test',
    'detect_trend',
    'compute_s_statistic',
    'hr98_variance_correction',
    'MKTestResult',
    
    # Sen's slope
    'sens_slope',
    'compute_sens_slope',
    'compute_all_slopes',
    'SenSlopeResult',
    
    # OLS HAC
    'ols_trend_test',
    'ols_fit',
    'newey_west_se',
    'OLSResult',
    
    # Position shift (Hodges-Lehmann)
    'position_shift',
    'PositionShiftResult',
    'hodges_lehmann_2sample',
    'bootstrap_hl_ci',
    'split_windows',
    'robust_sigma',
    
    # Trend analysis (Excel integration)
    'mk_trend_platform',
    'analyze_trend_from_excel',
    
    # Unified
    'TrendDetector'
]
