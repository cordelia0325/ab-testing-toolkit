"""
A/B Testing Toolkit

Statistical methods for trend detection and variance reduction in large-scale A/B testing.

Modules:
    - trend_detection: Mann-Kendall, Sen's slope, OLS with HAC
    - variance_reduction: CUPED, CUPAC (coming soon)

Author: Chuqiao Huang
"""

from . import trend_detection
from . import variance_reduction

__version__ = '0.1.0'
__author__ = 'Chuqiao Huang'
