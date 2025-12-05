"""
Variance Reduction Module

Methods for reducing variance in A/B test treatment effect estimates,
allowing experiments to reach statistical significance with fewer samples.

Available methods:
    - CUPED: Controlled-experiment Using Pre-Experiment Data
    - CUPAC: CUPED with ML-based covariate adjustment (cross-fitting)

Author: Chuqiao Huang
"""

from .cuped import (
    cuped_two_sample,
    cuped_from_dataframe,
    cuped_adjust,
    compute_theta,
    estimate_sample_size_reduction,
    CUPEDResult
)

from .cupac import (
    cupac_two_sample,
    cupac_from_dataframe,
    cross_fit_predict,
    compare_models,
    print_model_comparison,
    CUPACResult
)

__all__ = [
    # CUPED
    'cuped_two_sample',
    'cuped_from_dataframe',
    'cuped_adjust',
    'compute_theta',
    'estimate_sample_size_reduction',
    'CUPEDResult',
    
    # CUPAC
    'cupac_two_sample',
    'cupac_from_dataframe',
    'cross_fit_predict',
    'compare_models',
    'print_model_comparison',
    'CUPACResult',
]
