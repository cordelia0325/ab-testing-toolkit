# A/B Testing Toolkit

Statistical methods for trend detection and variance reduction in large-scale online experiments.

*Inspired by work at Kuaishou Technology's Causal Inference & Experimentation Platform.*

## Overview

This toolkit provides robust statistical methods for:

1. **Trend Detection**: Identifying metric drift between treatment and control groups
2. **Variance Reduction**: Improving experimental sensitivity with fewer samples (CUPED/CUPAC)

## Features

### Trend Detection

| Method | Description | Use Case |
|--------|-------------|----------|
| **Mann-Kendall** | Non-parametric test for monotonic trends | Robust to outliers, distribution-free |
| **Mann-Kendall + HR98** | With autocorrelation correction | Time series with serial dependence |
| **Sen's Slope** | Robust slope estimator (median of pairwise slopes) | Outlier-resistant trend estimation |
| **OLS + Newey-West HAC** | Parametric test with HAC standard errors | When normality holds approximately |
| **Hodges-Lehmann** | Two-sample position shift estimator | Comparing baseline vs study periods |

#### Decision Tree for Method Selection

```
Sample Size (n)
    │
    ├── n < 4: Insufficient data
    │
    ├── 4 ≤ n ≤ 10 (no ties): Exact table lookup
    │
    └── n ≥ 11: Normal approximation
            │
            ├── 10 ≤ n ≤ 13: No HR98 correction
            │
            └── n ≥ 14: HR98 correction (optional)
                    │
                    ├── 14 ≤ n ≤ 20: Check lag-1 only
                    │
                    └── n ≥ 21: Use empirical lag selection
```

### Variance Reduction

| Method | Description |
|--------|-------------|
| **CUPED** | Controlled-experiment Using Pre-Experiment Data |
| **CUPAC** | CUPED with ML-based covariate adjustment (cross-fitting) |

## Installation

```bash
git clone https://github.com/[username]/ab-testing-toolkit.git
cd ab-testing-toolkit
pip install -r requirements.txt
```

## Quick Start

### Basic Trend Detection

```python
import numpy as np
from src.trend_detection import TrendDetector

# Generate sample data
np.random.seed(42)
data = np.cumsum(np.random.randn(30)) + np.arange(30) * 0.1

# Create detector and run
detector = TrendDetector(alpha=0.05, use_hr98=True)
result = detector.detect(data)

print(result.summary())
```

### Mann-Kendall Test

```python
from src.trend_detection import mann_kendall_test

result = mann_kendall_test(data, alpha=0.05, use_hr98=True)

print(f"S statistic: {result.S}")
print(f"p-value: {result.p_value:.4f}")
print(f"Trend: {result.trend}")
print(f"Direction: {result.direction}")
```

### Sen's Slope Estimation

```python
from src.trend_detection import sens_slope

result = sens_slope(data, alpha=0.05)

print(f"Slope: {result.slope:.6f}")
print(f"95% CI: [{result.conf_interval[0]:.6f}, {result.conf_interval[1]:.6f}]")
```

### OLS with Newey-West HAC

```python
from src.trend_detection import ols_trend_test

result = ols_trend_test(data, alpha=0.05, use_hac=True)

print(f"Slope: {result.slope:.6f}")
print(f"SE (HAC): {result.slope_se:.6f}")
print(f"p-value: {result.p_value:.4f}")
```

### Compare All Methods

```python
from src.trend_detection import TrendDetector

comparison = TrendDetector.compare_methods(data, alpha=0.05)

for method, res in comparison.items():
    print(f"{method}: {res}")
```

### Hodges-Lehmann Position Shift

```python
import pandas as pd
from src.trend_detection import position_shift

# Time series with datetime index
dates = pd.date_range('2024-01-01', periods=30, freq='D')
data = pd.Series([...], index=dates)

# Compare last 7 days (S) vs previous 7 days (B)
result = position_shift(
    data,
    win_S=7,           # Study window length
    win_B=7,           # Baseline window length  
    gap=0,             # Gap between windows
    use_bootstrap_ci=True,
    ci_alpha=0.05
)

print(f"HL shift: {result.hl_shift:.4f}")
print(f"Direction: {result.direction}")
print(f"SNR: {result.snr:.4f}")
print(f"95% CI: [{result.hl_ci[0]:.4f}, {result.hl_ci[1]:.4f}]")
print(f"Verdict: {result.hl_trend}")
```

### CUPED (Variance Reduction)

```python
from src.variance_reduction import cuped_two_sample

# Pre-experiment data (e.g., last week's metrics)
x_treatment = [...]  # Pre-experiment values for treatment
x_control = [...]    # Pre-experiment values for control

# Post-experiment data
y_treatment = [...]  # Post-experiment values for treatment
y_control = [...]    # Post-experiment values for control

result = cuped_two_sample(
    y_treatment=y_treatment,
    y_control=y_control,
    x_treatment=x_treatment,
    x_control=x_control,
    alpha=0.05
)

print(f"ATE (raw):   {result.ate_raw:.4f} ± {result.se_raw:.4f}")
print(f"ATE (CUPED): {result.ate_cuped:.4f} ± {result.se_cuped:.4f}")
print(f"Variance reduction: {result.variance_reduction*100:.1f}%")
```

### CUPAC (ML-based Variance Reduction)

```python
from src.variance_reduction import cupac_two_sample, compare_models
from sklearn.ensemble import RandomForestRegressor

# Multiple pre-experiment features
X_treatment = np.column_stack([x1_treatment, x2_treatment, x3_treatment])
X_control = np.column_stack([x1_control, x2_control, x3_control])

# Single model
result = cupac_two_sample(
    y_treatment=y_treatment,
    y_control=y_control,
    X_treatment=X_treatment,
    X_control=X_control,
    model=RandomForestRegressor(n_estimators=100),
    n_folds=5
)

print(result.summary())

# Compare multiple models
results = compare_models(
    y_treatment, y_control,
    X_treatment, X_control
)
# Compares: LinearRegression, Ridge, DecisionTree, RandomForest
```

### Analyze Trends from Excel

```python
from src.trend_detection import analyze_trend_from_excel

# Analyze trend from Excel file with visualization
result = analyze_trend_from_excel(
    'metrics.xlsx',
    sheet_name=0,
    date_col='date',        # Auto-detects if None
    name_col='metric_name',
    value_col='value',
    start_date='2024-01-01',
    end_date='2024-01-31',
    alpha=0.10,
    run_hr98=True,
    run_sen=True,
    show_plot=True
)

# Or use the platform API directly with a Series
from src.trend_detection import mk_trend_platform

result = mk_trend_platform(
    series=my_series,  # pd.Series with datetime index
    alpha=0.10,
    run_hr98=True
)
```

## Project Structure

```
ab-testing-toolkit/
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── trend_detection/
│   │   ├── __init__.py
│   │   ├── mann_kendall.py      # Mann-Kendall with HR98
│   │   ├── sens_slope.py        # Sen's slope estimator
│   │   ├── ols_hac.py           # OLS with Newey-West HAC
│   │   ├── position_shift.py    # Hodges-Lehmann shift
│   │   ├── trend_analysis.py    # Excel integration tool
│   │   └── unified.py           # Unified detector
│   │
│   ├── variance_reduction/
│   │   ├── __init__.py
│   │   ├── cuped.py             # CUPED implementation
│   │   └── cupac.py             # CUPAC with cross-fitting
│   │
│   └── utils/
│       └── __init__.py
│
├── notebooks/
│   ├── trend_detection_demo.ipynb
│   └── variance_reduction_demo.ipynb
│
├── tests/
│   └── test_trend_detection.py
│
└── data/
    └── synthetic/
```

## Methods

### Mann-Kendall Test

The Mann-Kendall test is a non-parametric test for detecting monotonic trends.

**S statistic:**
$$S = \sum_{i<j} \text{sign}(x_j - x_i)$$

**Variance (with ties correction):**
$$\text{Var}(S) = \frac{n(n-1)(2n+5) - \sum_t t(t-1)(2t+5)}{18}$$

**Z statistic (normal approximation):**
$$Z = \frac{|S| - 1}{\sqrt{\text{Var}(S)}}$$

### HR98 Autocorrelation Correction

For autocorrelated data, the variance is inflated by a factor C:

$$\text{Var}^*(S) = C \cdot \text{Var}(S)$$

where:
$$C = 1 + \frac{2}{n(n-1)(n-2)} \sum_{k=1}^{L} (n-k)(n-k-1)(n-k-2) \rho_k$$

### Sen's Slope

The Sen's slope estimator is the median of all pairwise slopes:

$$\hat{\beta} = \mathrm{median}\left\\{ \frac{x_j - x_i}{j - i} : i < j \right\\}$$

### Newey-West HAC Estimator

The HAC covariance matrix uses Bartlett kernel weights:

$$\hat{V} = (X'X)^{-1} \hat{S} (X'X)^{-1}$$

where:
$$\hat{S} = \sum_{j=-L}^{L} w(j) \hat{\Gamma}_j$$

### Hodges-Lehmann Position Shift

The two-sample Hodges-Lehmann estimator measures the shift between baseline (B) and study (S) periods:

$$\hat{\Delta} = \text{median}\{ s_i - b_j : s_i \in S, b_j \in B \}$$

**Window Layout:**
```
[...older data...][B window][gap][S window]
                                      ^end_date
```

**Signal-to-Noise Ratio:**
$$\text{SNR} = \frac{|\hat{\Delta}|}{\sigma}$$

where σ is estimated using MAD (Median Absolute Deviation):
$$\sigma = 1.4826 \times \text{MAD}$$

**Verdict (based on bootstrap CI):**
- CI lower bound > 0 → significant_increase
- CI upper bound < 0 → significant_decrease  
- CI spans 0 → not_significant

### CUPED (Variance Reduction)

CUPED uses pre-experiment data X to reduce variance in post-experiment outcome Y:

$$Y_{adj} = Y - \theta (X - \bar{X})$$

**Optimal theta:**
$$\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

**Variance reduction:**
$$\frac{\text{Var}(Y_{adj})}{\text{Var}(Y)} = 1 - \rho^2_{XY}$$

where ρ is the correlation between X and Y.

### CUPAC (ML-based Variance Reduction)

CUPAC extends CUPED by using ML predictions as the covariate:

$$Y_{adj} = Y - \theta (\hat{Y} - \bar{\hat{Y}})$$

where $\hat{Y}$ is predicted using cross-fitting:

1. Split data into K folds
2. For each fold k: train model on all other folds, predict for fold k
3. Use out-of-fold predictions as covariate

**Supported models:**
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest

## References

1. Mann, H. B. (1945). Nonparametric tests against trend. *Econometrica*.
2. Kendall, M. G. (1975). *Rank Correlation Methods*. Griffin, London.
3. Hamed, K. H., & Rao, A. R. (1998). A modified Mann-Kendall trend test for autocorrelated data. *Journal of Hydrology*.
4. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *JASA*.
5. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.
6. Gilbert, R. O. (1987). *Statistical Methods for Environmental Pollution Monitoring*.
7. Hodges, J. L., & Lehmann, E. L. (1963). Estimates of location based on rank tests. *Annals of Mathematical Statistics*.
8. Deng, A., et al. (2013). Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data (CUPED).

## License

MIT License

## Author

Chuqiao Huang  
Email: chuqiaohuang2025@gmail.com

---

*Note: This toolkit was developed for educational and research purposes. The methods implemented are based on well-established statistical techniques.*
