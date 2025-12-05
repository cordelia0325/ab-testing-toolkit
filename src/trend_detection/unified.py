"""
Unified Trend Detector

Combines multiple trend detection methods and provides a consistent interface
for detecting trends in A/B testing metrics.

The detector follows a decision tree based on sample size:
- n < 4: Insufficient data
- 4 <= n <= 10: Use exact table lookup (Mann-Kendall)
- n >= 11: Use normal approximation with optional HR98 correction

Author: Chuqiao Huang
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .mann_kendall import mann_kendall_test, MKTestResult
from .sens_slope import sens_slope, SenSlopeResult
from .ols_hac import ols_trend_test, OLSResult


@dataclass
class TrendResult:
    """Unified result container for trend detection."""
    # Primary results (from Mann-Kendall)
    has_trend: bool
    trend: str
    direction: Optional[str]
    p_value: Optional[float]
    
    # Sample info
    n: int
    method: str
    
    # Slope estimates
    mk_result: Optional[MKTestResult] = None
    sen_result: Optional[SenSlopeResult] = None
    ols_result: Optional[OLSResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_trend': self.has_trend,
            'trend': self.trend,
            'direction': self.direction,
            'p_value': self.p_value,
            'n': self.n,
            'method': self.method,
            'slope_sen': self.sen_result.slope if self.sen_result else None,
            'slope_ols': self.ols_result.slope if self.ols_result else None,
            'slope_ci': self.sen_result.conf_interval if self.sen_result else None
        }
    
    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=" * 50,
            "Trend Detection Results",
            "=" * 50,
            f"Sample size: {self.n}",
            f"Method: {self.method}",
            f"Has trend: {self.has_trend}",
            f"Direction: {self.direction or 'N/A'}",
            f"p-value: {self.p_value:.4f}" if self.p_value else "p-value: N/A",
        ]
        
        if self.sen_result and np.isfinite(self.sen_result.slope):
            lines.append(f"Sen's slope: {self.sen_result.slope:.6f}")
            if self.sen_result.conf_interval:
                ci = self.sen_result.conf_interval
                lines.append(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
        
        if self.ols_result and np.isfinite(self.ols_result.slope):
            lines.append(f"OLS slope: {self.ols_result.slope:.6f}")
            lines.append(f"  SE (HAC): {self.ols_result.slope_se:.6f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class TrendDetector:
    """
    Unified trend detector for A/B testing metrics.
    
    Combines:
    - Mann-Kendall test (non-parametric, robust to outliers)
    - Sen's slope (robust slope estimation)
    - OLS with HAC (parametric, handles autocorrelation)
    
    Example:
        >>> detector = TrendDetector(alpha=0.05)
        >>> result = detector.detect(data)
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        use_hr98: bool = True,
        use_ols: bool = True,
        alpha_acf: float = 0.05
    ):
        """
        Initialize trend detector.
        
        Args:
            alpha: Significance level for trend tests
            use_hr98: Whether to use HR98 autocorrelation correction
            use_ols: Whether to also run OLS trend test
            alpha_acf: Significance level for ACF testing in HR98
        """
        self.alpha = alpha
        self.use_hr98 = use_hr98
        self.use_ols = use_ols
        self.alpha_acf = alpha_acf
    
    def detect(
        self,
        series: np.ndarray,
        compute_slope: bool = True
    ) -> TrendResult:
        """
        Detect trend in time series.
        
        Args:
            series: Input time series
            compute_slope: Whether to compute slope estimates
        
        Returns:
            TrendResult object
        """
        x = np.asarray(series, dtype=np.float64)
        n = len(x)
        
        # Run Mann-Kendall test
        mk_result = mann_kendall_test(
            x,
            alpha=self.alpha,
            use_hr98=self.use_hr98,
            alpha_acf=self.alpha_acf
        )
        
        # Run Sen's slope if requested
        sen_result = None
        if compute_slope:
            sen_result = sens_slope(
                x,
                alpha=self.alpha,
                var_s=mk_result.var_s_corrected or mk_result.var_s_raw
            )
        
        # Run OLS with HAC if requested
        ols_result = None
        if self.use_ols and n >= 4:
            ols_result = ols_trend_test(
                x,
                alpha=self.alpha,
                use_hac=True
            )
        
        # Determine overall result
        has_trend = mk_result.trend not in ['no_trend', 'insufficient_data']
        
        return TrendResult(
            has_trend=has_trend,
            trend=mk_result.trend,
            direction=mk_result.direction,
            p_value=mk_result.p_value,
            n=n,
            method=mk_result.method,
            mk_result=mk_result,
            sen_result=sen_result,
            ols_result=ols_result
        )
    
    def detect_batch(
        self,
        series_list: List[np.ndarray],
        names: Optional[List[str]] = None
    ) -> Dict[str, TrendResult]:
        """
        Detect trends in multiple time series.
        
        Args:
            series_list: List of time series arrays
            names: Optional names for each series
        
        Returns:
            Dictionary mapping names to TrendResult objects
        """
        if names is None:
            names = [f"series_{i}" for i in range(len(series_list))]
        
        results = {}
        for name, series in zip(names, series_list):
            results[name] = self.detect(series)
        
        return results
    
    @staticmethod
    def compare_methods(series: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Compare results from different trend detection methods.
        
        Useful for cross-validation and method selection.
        
        Args:
            series: Input time series
            alpha: Significance level
        
        Returns:
            Dictionary with results from each method
        """
        x = np.asarray(series, dtype=np.float64)
        
        # Mann-Kendall without HR98
        mk_basic = mann_kendall_test(x, alpha=alpha, use_hr98=False)
        
        # Mann-Kendall with HR98
        mk_hr98 = mann_kendall_test(x, alpha=alpha, use_hr98=True)
        
        # Sen's slope
        sen = sens_slope(x, alpha=alpha)
        
        # OLS standard
        ols_std = ols_trend_test(x, alpha=alpha, use_hac=False)
        
        # OLS with HAC
        ols_hac = ols_trend_test(x, alpha=alpha, use_hac=True)
        
        return {
            'mann_kendall_basic': {
                'trend': mk_basic.trend,
                'p_value': mk_basic.p_value,
                'method': mk_basic.method
            },
            'mann_kendall_hr98': {
                'trend': mk_hr98.trend,
                'p_value': mk_hr98.p_value,
                'correction_factor': mk_hr98.correction_factor,
                'method': mk_hr98.method
            },
            'sens_slope': {
                'slope': sen.slope,
                'conf_interval': sen.conf_interval
            },
            'ols_standard': {
                'slope': ols_std.slope,
                'p_value': ols_std.p_value,
                'se': ols_std.slope_se
            },
            'ols_hac': {
                'slope': ols_hac.slope,
                'p_value': ols_hac.p_value,
                'se': ols_hac.slope_se,
                'lag': ols_hac.lag
            },
            'consensus': _compute_consensus(
                [mk_basic.trend, mk_hr98.trend, ols_std.trend, ols_hac.trend]
            )
        }


def _compute_consensus(trends: List[str]) -> str:
    """Compute consensus trend from multiple methods."""
    significant_up = sum(1 for t in trends if t == 'significant_increase')
    significant_down = sum(1 for t in trends if t == 'significant_decrease')
    no_trend = sum(1 for t in trends if t == 'no_trend')
    
    total = len(trends)
    
    if significant_up >= total / 2:
        return 'likely_increase'
    elif significant_down >= total / 2:
        return 'likely_decrease'
    elif no_trend >= total / 2:
        return 'likely_no_trend'
    else:
        return 'inconclusive'


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    # Generate test data with trend
    n = 30
    t = np.arange(n)
    trend_component = 0.3 * t
    noise = np.random.normal(0, 1, n)
    data = 10 + trend_component + noise
    
    # Create detector and run
    detector = TrendDetector(alpha=0.05, use_hr98=True)
    result = detector.detect(data)
    
    print(result.summary())
    
    # Compare methods
    print("\nMethod Comparison:")
    comparison = TrendDetector.compare_methods(data)
    for method, res in comparison.items():
        print(f"\n{method}:")
        for k, v in res.items():
            print(f"  {k}: {v}")
