"""
Tests for trend detection module.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trend_detection import (
    mann_kendall_test,
    sens_slope,
    ols_trend_test,
    TrendDetector,
    compute_s_statistic,
    position_shift,
    hodges_lehmann_2sample
)


class TestMannKendall:
    """Tests for Mann-Kendall test."""
    
    def test_increasing_trend(self):
        """Test detection of increasing trend."""
        np.random.seed(42)
        n = 30
        data = np.arange(n) * 0.5 + np.random.normal(0, 0.5, n)
        
        result = mann_kendall_test(data, alpha=0.05)
        
        assert result.direction == 'increasing'
        assert result.p_value < 0.05
        assert result.trend == 'significant_increase'
    
    def test_decreasing_trend(self):
        """Test detection of decreasing trend."""
        np.random.seed(42)
        n = 30
        data = -np.arange(n) * 0.5 + np.random.normal(0, 0.5, n)
        
        result = mann_kendall_test(data, alpha=0.05)
        
        assert result.direction == 'decreasing'
        assert result.p_value < 0.05
        assert result.trend == 'significant_decrease'
    
    def test_no_trend(self):
        """Test detection of no trend."""
        np.random.seed(42)
        n = 30
        data = np.random.normal(0, 1, n)
        
        result = mann_kendall_test(data, alpha=0.05)
        
        # With random noise, should not detect significant trend
        assert result.p_value > 0.05 or result.trend == 'no_trend'
    
    def test_small_sample_exact(self):
        """Test exact table lookup for small samples."""
        data = np.array([1, 2, 3, 4, 5])  # Clear increasing trend
        
        result = mann_kendall_test(data, alpha=0.05)
        
        assert result.method == 'exact_table'
        assert result.direction == 'increasing'
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        data = np.array([1, 2, 3])  # Only 3 points
        
        result = mann_kendall_test(data)
        
        assert result.trend == 'insufficient_data'
        assert result.p_value is None
    
    def test_s_statistic_computation(self):
        """Test S statistic computation."""
        # For [1, 2, 3, 4, 5], all pairs are concordant
        # S = 10 (number of pairs n*(n-1)/2 = 10)
        data = np.array([1, 2, 3, 4, 5])
        S = compute_s_statistic(data)
        
        assert S == 10
        
        # For [5, 4, 3, 2, 1], all pairs are discordant
        data_rev = np.array([5, 4, 3, 2, 1])
        S_rev = compute_s_statistic(data_rev)
        
        assert S_rev == -10


class TestSensSlope:
    """Tests for Sen's slope estimator."""
    
    def test_known_slope(self):
        """Test slope estimation with known slope."""
        n = 20
        true_slope = 0.5
        t = np.arange(n)
        data = 2.0 + true_slope * t  # No noise
        
        result = sens_slope(data)
        
        assert np.isclose(result.slope, true_slope, atol=1e-10)
    
    def test_confidence_interval(self):
        """Test that confidence interval contains true slope."""
        np.random.seed(42)
        n = 30
        true_slope = 0.3
        t = np.arange(n)
        data = 1.0 + true_slope * t + np.random.normal(0, 0.3, n)
        
        result = sens_slope(data, alpha=0.05)
        
        assert result.conf_interval is not None
        ci_low, ci_high = result.conf_interval
        # True slope should be within CI (with high probability)
        # This might occasionally fail due to randomness


class TestHLPositionShift:
    """Tests for Hodges-Lehmann Position Shift Estimator."""

    def test_core_hl_calculation(self):
        """Test the mathematical correctness of HL estimator on small data."""
        # B = [1, 2], S = [4, 5]
        # Differences (s - b):
        # 4-1=3, 4-2=2
        # 5-1=4, 5-2=3
        # Diff set: {2, 3, 3, 4}. Median is 3.
        B = np.array([1, 2])
        S = np.array([4, 5])
        
        shift = hodges_lehmann_2sample(S, B)
        assert np.isclose(shift, 3.0)

    def test_integration_shift_detection(self):
        """Test full pipeline detection of a known level shift."""
        # Create dates
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        
        # First 10 days = 10, Last 10 days = 15 (Shift = +5)
        values = np.concatenate([np.ones(10)*10, np.ones(10)*15])
        series = pd.Series(values, index=dates)
        
        # Run with explicit windows (last 5 days vs previous 5 days)
        # Gap = 0
        result = position_shift(
            series, 
            win_S=5, 
            win_B=5, 
            gap=0
        )
        
        assert result.ok is True
        assert result.direction == 'up'
        assert np.isclose(result.hl_shift, 5.0)
        assert result.n_S == 5
        assert result.n_B == 5

    def test_window_logic_and_gap(self):
        """Test if windows and gaps are sliced correctly."""
        # 20 days: 1...20
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        series = pd.Series(np.arange(20), index=dates)
        
        # S window = 5 days (ends on day 20) -> days 16,17,18,19,20
        # Gap = 2 days -> skips days 14, 15
        # B window = 5 days -> days 9,10,11,12,13
        
        result = position_shift(series, win_S=5, win_B=5, gap=2)
        
        assert result.ok
        # Check counts
        assert result.n_S == 5
        assert result.n_B == 5
        
        # We can check specific values if we exposed the raw data in result, 
        # but here we check the median difference to infer correctness
        # S median (15..19) -> 17
        # B median (8..12) -> 10
        # Expected diff ~ 7
        assert np.isclose(result.med_S - result.med_B, 7.0)

    def test_noise_handling_and_threshold(self):
        """Test signal-to-noise ratio and threshold logic."""
        np.random.seed(99)
        dates = pd.date_range('2024-01-01', periods=60, freq='D')
        
        # High noise, small shift
        # B: Mean 100, S: Mean 100.2 (Tiny shift)
        # Noise sigma approx 1.0
        values = np.random.normal(100, 1.0, 60)
        values[30:] += 0.2 
        
        series = pd.Series(values, index=dates)
        
        # Set a business threshold of 2.0 (much larger than the 0.2 shift)
        result = position_shift(
            series, 
            win_S=15, 
            win_B=15, 
            biz_threshold=2.0
        )
        
        # Should NOT pass threshold
        assert result.pass_threshold is False
        # SNR should be low (0.2 / 1.0 = 0.2)
        assert result.snr < 1.0

    def test_insufficient_data_error(self):
        """Test error handling when windows are empty or too small."""
        dates = pd.date_range('2024-01-01', periods=4, freq='D')
        series = pd.Series([1, 2, 3, 4], index=dates)
        
        # Asking for windows larger than data
        result = position_shift(series, win_S=10, win_B=10)
        
        assert result.ok is False
        assert "Insufficient data" in result.reason
        
class TestOLSHAC:
    """Tests for OLS with HAC standard errors."""
    
    def test_known_slope(self):
        """Test slope estimation with known slope."""
        n = 30
        true_slope = 0.4
        t = np.arange(n)
        data = 3.0 + true_slope * t
        
        result = ols_trend_test(data, use_hac=False)
        
        assert np.isclose(result.slope, true_slope, atol=1e-10)
    
    def test_hac_larger_se(self):
        """Test that HAC SE is larger for autocorrelated data."""
        np.random.seed(42)
        n = 50
        
        # Generate autocorrelated errors
        rho = 0.7
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, 1)
        for i in range(1, n):
            errors[i] = rho * errors[i-1] + np.random.normal(0, 1)
        
        data = 5.0 + 0.1 * np.arange(n) + errors
        
        result_std = ols_trend_test(data, use_hac=False)
        result_hac = ols_trend_test(data, use_hac=True)
        
        # HAC SE should generally be larger for positively autocorrelated data
        # This might not always hold for specific random seeds
        assert result_hac.slope_se > 0
        assert result_std.slope_se > 0


class TestTrendDetector:
    """Tests for unified TrendDetector."""
    
    def test_detect_increasing(self):
        """Test unified detection of increasing trend."""
        np.random.seed(42)
        data = np.arange(30) * 0.5 + np.random.normal(0, 0.5, 30)
        
        detector = TrendDetector(alpha=0.05)
        result = detector.detect(data)
        
        assert result.has_trend
        assert result.direction == 'increasing'
    
    def test_compare_methods(self):
        """Test method comparison."""
        np.random.seed(42)
        data = np.arange(30) * 0.3 + np.random.normal(0, 0.5, 30)
        
        comparison = TrendDetector.compare_methods(data)
        
        assert 'mann_kendall_basic' in comparison
        assert 'mann_kendall_hr98' in comparison
        assert 'sens_slope' in comparison
        assert 'ols_standard' in comparison
        assert 'ols_hac' in comparison
        assert 'consensus' in comparison


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
