"""
Utility functions for A/B Testing Toolkit.
"""

import numpy as np
from typing import Optional


def format_p_value(p: float, decimals: int = 4, tiny: float = 1e-4) -> str:
    """Format p-value for display."""
    if not np.isfinite(p):
        return "N/A"
    return f"<{tiny:.{decimals}f}" if p < tiny else f"{p:.{decimals}f}"


def generate_trend_data(
    n: int = 30,
    slope: float = 0.1,
    intercept: float = 0.0,
    noise_std: float = 1.0,
    ar_coef: float = 0.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic time series data with trend.
    
    Args:
        n: Number of observations
        slope: True slope of trend
        intercept: Intercept
        noise_std: Standard deviation of noise
        ar_coef: AR(1) coefficient for autocorrelated errors
        seed: Random seed
    
    Returns:
        Time series array
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(n, dtype=float)
    trend = intercept + slope * t
    
    # Generate errors
    if ar_coef == 0:
        errors = np.random.normal(0, noise_std, n)
    else:
        errors = np.zeros(n)
        errors[0] = np.random.normal(0, noise_std)
        for i in range(1, n):
            errors[i] = ar_coef * errors[i-1] + np.random.normal(0, noise_std * np.sqrt(1 - ar_coef**2))
    
    return trend + errors
