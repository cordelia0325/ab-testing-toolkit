"""
CUPAC: CUPED with Predicted covariates using Adaptive models

An extension of CUPED that uses machine learning models to construct
more predictive covariates, potentially achieving greater variance reduction.

Key differences from CUPED:
    1. Uses ML models (not just linear) to predict Y from covariates
    2. Employs cross-fitting to avoid overfitting bias
    3. Can combine multiple pre-experiment features

Cross-fitting procedure:
    1. Split data into K folds
    2. For each fold k:
       - Train model on all data except fold k
       - Predict Y_hat for fold k
    3. Use predictions as the covariate in CUPED adjustment

References:
    - Poyarkov, A., et al. (2016). Boosted Decision Tree Regression Adjustment 
      for Variance Reduction in Online Controlled Experiments. KDD '16.
    - Deng, A., & Shi, X. (2016). Data-Driven Metric Development for Online 
      Controlled Experiments. KDD '16.

Author: Chuqiao Huang
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any, Union, Callable
from dataclasses import dataclass
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin, clone


@dataclass
class CUPACResult:
    """Result container for CUPAC analysis."""
    # Original estimates
    mean_treatment: float
    mean_control: float
    ate_raw: float
    var_raw: float
    se_raw: float
    
    # CUPAC-adjusted estimates
    ate_cupac: float
    var_cupac: float
    se_cupac: float
    
    # Adjustment info
    theta: float
    variance_reduction: float
    model_name: str
    n_folds: int
    
    # Model performance (on pooled data)
    r_squared: float  # Rﾂｲ of predictions
    
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
            'ate_cupac': self.ate_cupac,
            'var_cupac': self.var_cupac,
            'se_cupac': self.se_cupac,
            'theta': self.theta,
            'variance_reduction': self.variance_reduction,
            'model_name': self.model_name,
            'n_folds': self.n_folds,
            'r_squared': self.r_squared,
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
            "CUPAC Analysis Results",
            "=" * 50,
            f"Model: {self.model_name} ({self.n_folds}-fold cross-fitting)",
            f"Sample sizes: Treatment={self.n_treatment}, Control={self.n_control}",
            "",
            "Raw Estimates:",
            f"  ATE (raw): {self.ate_raw:.6f}",
            f"  SE (raw):  {self.se_raw:.6f}",
            "",
            "CUPAC-Adjusted Estimates:",
            f"  ATE (CUPAC): {self.ate_cupac:.6f}",
            f"  SE (CUPAC):  {self.se_cupac:.6f}",
            f"  theta:       {self.theta:.6f}",
            "",
            f"Model Rﾂｲ: {self.r_squared:.4f}",
            f"Variance Reduction: {self.variance_reduction*100:.2f}%",
            "",
            "Statistical Inference (CUPAC):",
            f"  t-statistic: {self.t_stat:.4f}",
            f"  p-value:     {self.p_value:.4f}",
            f"  95% CI:      [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
        ]
        return "\n".join(lines)


def cross_fit_predict(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate out-of-fold predictions using cross-fitting.
    
    This avoids overfitting bias by ensuring each prediction
    is made by a model that never saw that observation.
    
    Args:
        model: Sklearn-compatible regressor
        X: Feature matrix
        y: Target variable
        n_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility
    
    Returns:
        Array of out-of-fold predictions (same length as y)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    n = len(y)
    predictions = np.zeros(n)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        
        # Clone model to avoid state leakage
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        predictions[test_idx] = model_clone.predict(X_test)
    
    return predictions


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Rﾂｲ (coefficient of determination)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


def cupac_two_sample(
    y_treatment: np.ndarray,
    y_control: np.ndarray,
    X_treatment: np.ndarray,
    X_control: np.ndarray,
    model: Optional[BaseEstimator] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42
) -> CUPACResult:
    """
    Perform CUPAC analysis for a two-sample A/B test.
    
    Uses cross-fitting to generate predictions, then applies CUPED
    adjustment using predictions as the covariate.
    
    Args:
        y_treatment: Post-experiment outcomes for treatment group
        y_control: Post-experiment outcomes for control group
        X_treatment: Pre-experiment features for treatment group (n_samples, n_features)
        X_control: Pre-experiment features for control group
        model: Sklearn regressor (default: LinearRegression)
        n_folds: Number of folds for cross-fitting
        alpha: Significance level for confidence intervals
        random_state: Random seed
    
    Returns:
        CUPACResult object with all analysis results
    """
    # Default model
    if model is None:
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    
    # Convert to arrays
    y_t = np.asarray(y_treatment, dtype=float)
    y_c = np.asarray(y_control, dtype=float)
    X_t = np.asarray(X_treatment, dtype=float)
    X_c = np.asarray(X_control, dtype=float)
    
    # Ensure 2D
    if X_t.ndim == 1:
        X_t = X_t.reshape(-1, 1)
    if X_c.ndim == 1:
        X_c = X_c.reshape(-1, 1)
    
    n_t, n_c = len(y_t), len(y_c)
    
    # Raw estimates
    mean_t = np.mean(y_t)
    mean_c = np.mean(y_c)
    ate_raw = mean_t - mean_c
    
    var_t = np.var(y_t, ddof=1)
    var_c = np.var(y_c, ddof=1)
    var_raw = var_t / n_t + var_c / n_c
    se_raw = np.sqrt(var_raw)
    
    # Pool data for cross-fitting
    y_pooled = np.concatenate([y_t, y_c])
    X_pooled = np.vstack([X_t, X_c])
    
    # Cross-fit predictions
    y_hat = cross_fit_predict(
        model, X_pooled, y_pooled, 
        n_folds=n_folds, 
        random_state=random_state
    )
    
    # Split predictions back
    y_hat_t = y_hat[:n_t]
    y_hat_c = y_hat[n_t:]
    
    # Compute Rﾂｲ of predictions
    r_squared = compute_r_squared(y_pooled, y_hat)
    
    # Compute theta using predictions as covariate
    # theta = Cov(Y, Y_hat) / Var(Y_hat)
    var_yhat = np.var(y_hat, ddof=1)
    if var_yhat > 1e-10:
        cov_y_yhat = np.cov(y_pooled, y_hat, ddof=1)[0, 1]
        theta = cov_y_yhat / var_yhat
    else:
        theta = 0.0
    
    y_hat_mean = np.mean(y_hat)
    
    # CUPAC adjustment
    y_t_adj = y_t - theta * (y_hat_t - y_hat_mean)
    y_c_adj = y_c - theta * (y_hat_c - y_hat_mean)
    
    # CUPAC estimates
    mean_t_adj = np.mean(y_t_adj)
    mean_c_adj = np.mean(y_c_adj)
    ate_cupac = mean_t_adj - mean_c_adj
    
    var_t_adj = np.var(y_t_adj, ddof=1)
    var_c_adj = np.var(y_c_adj, ddof=1)
    var_cupac = var_t_adj / n_t + var_c_adj / n_c
    se_cupac = np.sqrt(var_cupac)
    
    # Variance reduction
    variance_reduction = 1 - var_cupac / var_raw if var_raw > 0 else 0.0
    
    # Statistical inference
    t_stat = ate_cupac / se_cupac if se_cupac > 0 else 0.0
    
    # Welch's degrees of freedom
    df = (var_t_adj/n_t + var_c_adj/n_c)**2 / (
        (var_t_adj/n_t)**2/(n_t-1) + (var_c_adj/n_c)**2/(n_c-1)
    )
    
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = ate_cupac - t_crit * se_cupac
    ci_upper = ate_cupac + t_crit * se_cupac
    
    # Get model name
    model_name = model.__class__.__name__
    
    return CUPACResult(
        mean_treatment=mean_t,
        mean_control=mean_c,
        ate_raw=ate_raw,
        var_raw=var_raw,
        se_raw=se_raw,
        ate_cupac=ate_cupac,
        var_cupac=var_cupac,
        se_cupac=se_cupac,
        theta=theta,
        variance_reduction=variance_reduction,
        model_name=model_name,
        n_folds=n_folds,
        r_squared=r_squared,
        t_stat=t_stat,
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_treatment=n_t,
        n_control=n_c
    )


def cupac_from_dataframe(
    df: pd.DataFrame,
    y_col: str,
    feature_cols: List[str],
    group_col: str,
    model: Optional[BaseEstimator] = None,
    treatment_value: Any = 1,
    control_value: Any = 0,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42
) -> CUPACResult:
    """
    Convenience function to run CUPAC from a DataFrame.
    
    Args:
        df: DataFrame containing experiment data
        y_col: Column name for post-experiment outcome
        feature_cols: List of column names for pre-experiment features
        group_col: Column name for treatment assignment
        model: Sklearn regressor (default: LinearRegression)
        treatment_value: Value indicating treatment group
        control_value: Value indicating control group
        n_folds: Number of folds for cross-fitting
        alpha: Significance level
        random_state: Random seed
    
    Returns:
        CUPACResult object
    
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> result = cupac_from_dataframe(
        ...     df,
        ...     y_col='revenue_post',
        ...     feature_cols=['revenue_pre', 'visits_pre', 'days_active'],
        ...     group_col='treatment',
        ...     model=RandomForestRegressor(n_estimators=100),
        ...     n_folds=5
        ... )
    """
    treatment_mask = df[group_col] == treatment_value
    control_mask = df[group_col] == control_value
    
    return cupac_two_sample(
        y_treatment=df.loc[treatment_mask, y_col].values,
        y_control=df.loc[control_mask, y_col].values,
        X_treatment=df.loc[treatment_mask, feature_cols].values,
        X_control=df.loc[control_mask, feature_cols].values,
        model=model,
        n_folds=n_folds,
        alpha=alpha,
        random_state=random_state
    )


def compare_models(
    y_treatment: np.ndarray,
    y_control: np.ndarray,
    X_treatment: np.ndarray,
    X_control: np.ndarray,
    models: Optional[Dict[str, BaseEstimator]] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42
) -> Dict[str, CUPACResult]:
    """
    Compare multiple models for CUPAC analysis.
    
    Args:
        y_treatment, y_control: Post-experiment outcomes
        X_treatment, X_control: Pre-experiment features
        models: Dict of {name: model} (default: LR, Ridge, RF, DT)
        n_folds: Number of folds for cross-fitting
        alpha: Significance level
        random_state: Random seed
    
    Returns:
        Dict of {model_name: CUPACResult}
    """
    if models is None:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'DecisionTree': DecisionTreeRegressor(max_depth=5, random_state=random_state),
            'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state),
        }
    
    results = {}
    for name, model in models.items():
        result = cupac_two_sample(
            y_treatment=y_treatment,
            y_control=y_control,
            X_treatment=X_treatment,
            X_control=X_control,
            model=model,
            n_folds=n_folds,
            alpha=alpha,
            random_state=random_state
        )
        results[name] = result
    
    return results


def print_model_comparison(results: Dict[str, CUPACResult]) -> None:
    """Print a comparison table of model results."""
    print("\nModel Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'R-squared':>8} {'Var Reduction':>15} {'SE':>10} {'p-value':>10}")
    print("-" * 70)
    
    for name, result in sorted(results.items(), key=lambda x: -x[1].variance_reduction):
        print(f"{name:<20} {result.r_squared:>8.4f} {result.variance_reduction*100:>14.2f}% "
              f"{result.se_cupac:>10.6f} {result.p_value:>10.4f}")


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    n_treatment = 1000
    n_control = 1000
    
    # Generate multiple pre-experiment features
    # Feature 1: Previous metric value
    x1_treatment = np.random.normal(100, 20, n_treatment)
    x1_control = np.random.normal(100, 20, n_control)
    
    # Feature 2: User activity level
    x2_treatment = np.random.normal(50, 10, n_treatment)
    x2_control = np.random.normal(50, 10, n_control)
    
    # Feature 3: Days since registration
    x3_treatment = np.random.exponential(30, n_treatment)
    x3_control = np.random.exponential(30, n_control)
    
    # Combine features
    X_treatment = np.column_stack([x1_treatment, x2_treatment, x3_treatment])
    X_control = np.column_stack([x1_control, x2_control, x3_control])
    
    # Post-experiment outcome (non-linear relationship + treatment effect)
    def generate_outcome(X, treatment_effect=0):
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
        noise = np.random.normal(0, 10, len(X))
        return 0.5 * x1 + 0.3 * x2 + 0.1 * np.sqrt(x3) + noise + treatment_effect
    
    treatment_effect = 5.0
    y_treatment = generate_outcome(X_treatment, treatment_effect)
    y_control = generate_outcome(X_control, 0)
    
    # Compare different models
    print("Comparing CUPAC models...")
    results = compare_models(
        y_treatment=y_treatment,
        y_control=y_control,
        X_treatment=X_treatment,
        X_control=X_control,
        n_folds=5
    )
    
    print_model_comparison(results)
    
    print(f"\nTrue treatment effect: {treatment_effect}")
    
    # Show best model details
    best_model = max(results.items(), key=lambda x: x[1].variance_reduction)
    print(f"\nBest model: {best_model[0]}")
    print(best_model[1].summary())
