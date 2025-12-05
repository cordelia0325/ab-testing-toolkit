"""
Trend Analysis Tool for Excel Data

This module integrates Mann-Kendall Trend Test and Sen's Slope Estimator
to analyze time series data directly from Excel files.

Dependencies:
    - pandas, numpy, matplotlib
    - mann_kendall (local module)
    - sens_slope (local module)
    
Author: Chuqiao Huang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from typing import Optional, Dict, Any, Union, Tuple, List

# === Import Local Statistical Modules ===
# Ensure these files are in the same directory or Python path
try:
    from .mann_kendall import mann_kendall_test
    from .sens_slope import sens_slope
except ImportError:
    # Fallback for direct execution
    from mann_kendall import mann_kendall_test
    from sens_slope import sens_slope

# =============================================================================
# 1. IO & Data Loading (Input Layer)
# =============================================================================

def _validate_window(y: np.ndarray, *, min_len: int = 4, require_no_nan: bool = True) -> Tuple[bool, Optional[str]]:
    """Validate if the data window is sufficient for testing."""
    y = np.asarray(y, dtype=float)
    if require_no_nan and (not np.all(np.isfinite(y))):
        return False, "Target window contains NaN or Inf values."
    if y.size < min_len:
        return False, f"Target window length must be >= {min_len}."
    return True, None

def _to_series_from_excel(
    xlsx_path: str,
    sheet_name: Union[str, int] = 0,
    date_col: Optional[Union[str, int]] = None,
    name_col: Optional[Union[str, int]] = None,
    value_col: Optional[Union[str, int]] = None
) -> Tuple[pd.Series, str, pd.DataFrame]:
    """
    Read Excel file and extract the time series.
    
    Returns:
        (series, metric_label, df_raw)
    """
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception as e:
        raise IOError(f"Failed to read Excel file: {e}")
        
    cols = list(df.columns)

    def _resolve_col(spec, role, candidates_regex):
        """Resolve column name by index, exact name, or regex."""
        # By Index
        if isinstance(spec, int):
            if spec < 0 or spec >= len(cols):
                raise IndexError(f"{role} column index out of bounds: {spec}")
            return cols[spec]
        
        # By Name
        if isinstance(spec, str):
            if spec in df.columns:
                return spec
            # Try stripped matching
            trimmed = {c.strip(): c for c in df.columns}
            if spec.strip() in trimmed:
                return trimmed[spec.strip()]
            raise KeyError(f"{role} column not found: '{spec}'. Available: {cols}")

        # Auto-detect via Regex
        pattern = re.compile(candidates_regex, re.I)
        for c in cols:
            if isinstance(c, str) and pattern.search(c.strip()):
                return c
        raise KeyError(f"Auto-detection failed for {role}. Available: {cols}")

    # Regex patterns for auto-detection
    date_name = _resolve_col(date_col, "Date", r"date|time|day|dt|period|æ¥æ|??")
    name_name = _resolve_col(name_col, "Metric Name", r"metric|name|label|target|æ?")
    # Matches 'value', 'diff', 'uplift', 'ratio', etc.
    value_name = _resolve_col(value_col, "Value", r"value|diff|rel|ratio|change|improv|uplift|y|çž?|æå")

    # Data Cleaning
    df = df.copy()
    df[date_name] = pd.to_datetime(df[date_name], errors="coerce")

    # Check for bad dates
    bad_date_mask = df[date_name].isna()
    if bad_date_mask.any():
        bad_rows = df.loc[bad_date_mask].head(5).to_dict(orient="records")
        raise ValueError(f"Date column contains invalid values ({bad_date_mask.sum()} rows). Examples: {bad_rows}")

    df = df.sort_values(by=date_name)

    # Extract Metric Label (take first non-null)
    if df[name_name].notna().any():
        metric_label = str(df[name_name].dropna().iloc[0])
    else:
        metric_label = str(name_name)

    # Process Value Series
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    s = pd.Series(df[value_name].values, index=df[date_name])
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s.index = pd.to_datetime(s.index)
    
    # Note: We do not dropna here; let the platform logic decide how to handle gaps
    return s, metric_label, df


# =============================================================================
# 2. Logic Layer: The Platform (mk_trend_platform)
# =============================================================================

def mk_trend_platform(
    xlsx_path: Optional[str] = None,
    series: Optional[pd.Series] = None,
    *,
    # Excel options
    sheet_name: Union[str, int] = 0,
    date_col: Any = None, 
    name_col: Any = None, 
    value_col: Any = None,
    # Window options
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    # Stats options
    alpha: float = 0.10,
    run_hr98: bool = True,
    run_sen: bool = True
) -> Dict[str, Any]:
    """
    Core logic simulating the A/B platform backend.
    
    1. Loads data (from Series or Excel).
    2. Slices the target time window.
    3. Runs Mann-Kendall test (switching between Exact/HR98 based on N).
    4. Runs Sen's Slope.
    5. Returns structured dictionary.
    """
    
    # --- 1) Load Full Data ---
    if series is not None:
        s_all = pd.Series(series, dtype="float64").copy()
        
        # Parse Index
        idx_try = pd.to_datetime(s_all.index, errors="coerce")
        if idx_try.isna().all():
            # Synthetic index if no dates provided
            idx = pd.date_range("2000-01-01", periods=len(s_all), freq="D")
        elif idx_try.isna().any():
            return {"ok": False, "message": "Series index contains invalid dates."}
        else:
            idx = idx_try
            
        if pd.Index(idx).has_duplicates:
            return {"ok": False, "message": "Series index contains duplicate dates."}
            
        s_all.index = pd.to_datetime(idx)
        s_all = s_all.sort_index()
        metric_label = "Provided Series"
        
    elif xlsx_path:
        try:
            s_all, metric_label, _ = _to_series_from_excel(
                xlsx_path, sheet_name, date_col, name_col, value_col
            )
        except Exception as e:
            return {"ok": False, "message": f"Excel load failed: {str(e)}"}
    else:
        return {"ok": False, "message": "Must provide either 'series' or 'xlsx_path'."}

    if s_all.empty:
        return {"ok": False, "message": "Input series is empty."}

    # --- 2) Window Slicing ---
    data_min = s_all.index.min().normalize()
    data_max = s_all.index.max().normalize()

    # Resolve dates
    dt_end = pd.to_datetime(end_date).normalize() if end_date else data_max
    dt_start = pd.to_datetime(start_date).normalize() if start_date else data_min

    # Window Validation
    dummy_window = {"metric": metric_label, "start": data_min.date(), "end": data_max.date(), "n": 0}
    
    if dt_start > dt_end:
        return {"ok": False, "message": f"Start date ({dt_start.date()}) cannot be after end date ({dt_end.date()}).", "window": dummy_window}
    
    # Out of bounds check (optional warning logic could go here, but we fail strict)
    if dt_start < data_min:
         return {"ok": False, "message": f"Window start ({dt_start.date()}) is before data start ({data_min.date()}).", "window": dummy_window}
    if dt_end > data_max:
         return {"ok": False, "message": f"Window end ({dt_end.date()}) is after data end ({data_max.date()}).", "window": dummy_window}

    # Slice
    mask = (s_all.index.normalize() >= dt_start) & (s_all.index.normalize() <= dt_end)
    s_target = s_all.loc[mask].sort_index()
    
    # Drop NaNs for the calculation part
    s_target_clean = s_target.dropna()
    n = len(s_target_clean)

    if n == 0:
        return {"ok": False, "message": "Target window is empty.", "window": dummy_window}

    # Validate constraints (N >= 4, etc)
    ok_win, msg_win = _validate_window(s_target_clean.values, min_len=4)
    if not ok_win:
        return {
            "ok": False, 
            "message": msg_win, 
            "window": {"metric": metric_label, "start": dt_start.date(), "end": dt_end.date(), "n": n}
        }

    # --- 3) Statistical Testing ---
    
    # Decision Logic:
    # If 4 <= N <= 10: Use Exact Table (small sample), skip HR98 to avoid overfitting
    # If N > 10: Use HR98 (autocorrelation correction) if requested
    
    if 4 <= n <= 10:
        # mann_kendall_test handles exact table lookup automatically for N<=10
        mk_res = mann_kendall_test(s_target_clean.values, alpha=alpha, use_hr98=False)
    else:
        mk_res = mann_kendall_test(s_target_clean.values, alpha=alpha, use_hr98=run_hr98)

    # Sen's Slope
    sen_res = None
    if run_sen:
        # If MK calculated a corrected variance, pass it to Sen's Slope for better CI
        var_for_ci = mk_res.var_s_corrected if mk_res.var_s_corrected else mk_res.var_s_raw
        sen_res = sens_slope(s_target_clean.values, alpha=alpha, var_s=var_for_ci)

    # --- 4) Structure Output ---
    
    # Helper to format p-value
    def _fmt_p(p):
        if p is None: return "N/A"
        return "< 0.001" if p < 0.001 else f"{p:.4f}"

    return {
        "ok": True,
        "window": {
            "metric": metric_label,
            "start": dt_start.date(),
            "end": dt_end.date(),
            "n": n
        },
        "mk": {
            "S": mk_res.S,
            "Z": mk_res.Z,
            "p_value": mk_res.p_value,
            "p_str": _fmt_p(mk_res.p_value),
            "trend": mk_res.trend,
            "direction": mk_res.direction,
            "method": mk_res.method
        },
        "sen": {
            "sen_slope": sen_res.slope if sen_res else None,
            "intercept": sen_res.intercept if sen_res else None,
            "ci": sen_res.conf_interval if sen_res else None,
        },
        "hr98": {
            "var_raw": mk_res.var_s_raw,
            "var_hr": mk_res.var_s_corrected,
            "correction_factor": mk_res.correction_factor,
            "acf_used": mk_res.acf_used
        },
        "data": {
            "all": s_all,
            "target": s_target  # Note: this might include NaNs if originally present, for plotting continuity
        }
    }


# =============================================================================
# 3. Application Layer: Analysis & Visualization (analyze_trend_from_excel)
# =============================================================================

def analyze_trend_from_excel(
    xlsx_path: str,   
    *,
    # Read xlsx
    sheet_name: Union[str, int] = 0,
    date_col: Any = None, 
    name_col: Any = None, 
    value_col: Any = None,
    # Target Window
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    # Statistics
    alpha: float = 0.10,
    run_hr98: bool = True,
    run_sen: bool = True,
    # Visualization
    show_plot: bool = True
) -> Dict[str, Any]:  
    """
    Primary Entry Point:
    1. Calls platform logic (load -> calc).
    2. Prints textual summary.
    3. Generates visualization plots.
    """
    
    # 1. Execute Logic
    res = mk_trend_platform(
        xlsx_path=xlsx_path, sheet_name=sheet_name,
        date_col=date_col, name_col=name_col, value_col=value_col,
        start_date=start_date, end_date=end_date,
        alpha=alpha, run_hr98=run_hr98, run_sen=run_sen
    )

    if not res.get("ok"):
        print(f"[MK Error] {res.get('message')}")
        return res

    # 2. Print Summary
    W = res["window"]
    MK = res["mk"]
    SEN = res["sen"]
    HR = res["hr98"]

    print("\n=== Trend Detection Results ===")
    print(f"Window: {W['start']} to {W['end']} | N={W['n']} | Metric: {W['metric']}")
    
    z_val = MK.get("Z")
    z_str = f"{z_val:.3f}" if (z_val is not None and np.isfinite(z_val)) else "NA"
    
    # Show Variance (Corrected if HR98 triggered, else Raw)
    var_val = HR.get('var_hr') if HR.get('var_hr') is not None else HR.get('var_raw')
    var_str = f"{var_val:.3f}" if (var_val is not None and np.isfinite(var_val)) else "NA"
    
    # Show ACFs used
    acf_list = HR.get('acf_used', []) or []
    acf_str = "[" + ", ".join(f"(lag={k}, rho={r:.3f})" for k, r in acf_list) + "]"
    
    print(f"Stats: S={MK['S']}, Z={z_str}, Var={var_str}, ACF_adj={acf_str}")
    
    if SEN.get("sen_slope") is not None and SEN.get("ci") is not None:
        slope = SEN["sen_slope"]
        low, high = SEN["ci"]
        # Assuming metric is relative (e.g. 0.05 for 5%), display as %
        print(f"Sen's Slope: {slope*100:.4g}% per day")
        print(f"Confidence Interval ({int((1-alpha)*100)}%): [{low*100:.4g}%, {high*100:.4g}%]")
    
    print(f"Result: p={MK['p_str']} | Trend: {MK['trend']} | Method: {MK['method']}")

    # 3. Visualization
    if show_plot:
        try:
            # Setup Plot
            s_all = res["data"]["all"]
            s_target = res["data"]["target"].dropna() # Clean for plotting points

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
            
            # --- Top Plot: Global Context ---
            # Plot all data in gray
            axes[0].plot(s_all.index, s_all.values * 100, '-', color='0.7', lw=1.5, label='All Data', zorder=1)
            # Highlight target window in blue
            axes[0].plot(s_target.index, s_target.values * 100, '-', color='#1f77b4', lw=2.0, label='Target Window', zorder=2)
            
            axes[0].axhline(0.0, color='k', linestyle=':', alpha=0.3)
            axes[0].set_ylabel("Uplift / Value (%)")
            axes[0].set_title(f"Global Context: {W['metric']}")
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.2)

            # --- Bottom Plot: Detailed Analysis ---
            # Points and Line
            axes[1].plot(s_target.index, s_target.values * 100, 'o-', ms=5, color='#1f77b4', lw=2.0, label='Observed', zorder=3)
            
            # Mean Line
            mean_v = float(s_target.mean())
            axes[1].axhline(mean_v * 100, color='#1f77b4', linestyle='--', alpha=0.5, label=f"Mean: {mean_v*100:.2f}%")
            axes[1].axhline(0.0, color='k', linestyle=':', alpha=0.3)

            # Sen's Slope Trend Line
            slope = SEN.get("sen_slope")
            if slope is not None and np.isfinite(slope):
                # Calculate X axis as days relative to window start
                # Note: sens_slope logic assumes equidistant steps, or uses actual x differences
                # Here we map dates to "Days since start" to match the slope calculation context
                t_start = s_target.index.min()
                days_diff = (s_target.index - t_start).days.values
                
                # Re-estimate intercept based on Median(y - slope*x) for visual consistency
                # (The sens_slope module provides an intercept, but we recalculate to match this specific x-axis basis)
                intercept_visual = np.median(s_target.values - slope * days_diff)
                
                y_hat = (intercept_visual + slope * days_diff) * 100 # Convert to %
                
                axes[1].plot(s_target.index, y_hat, '-', color='#ff7f0e', lw=2.5, 
                             label=f"Trend: {slope*100:.3f}%/day")

            # Y-Axis Auto-Scaling with padding
            vals = s_target.values * 100
            y_min, y_max = np.min(vals), np.max(vals)
            y_range = max(1.0, y_max - y_min)
            axes[1].set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)

            # Labels
            axes[1].set_ylabel("Uplift / Value (%)")
            axes[1].set_xlabel("Date")
            
            title_txt = f"Target Window ({W['start']} ~ {W['end']}) | N={W['n']}\n"
            title_txt += f"Mann-Kendall: {MK['trend']} (p={MK['p_str']})"
            axes[1].set_title(title_txt)
            
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.2)

            plt.show()
            
        except Exception as e:
            print(f"Visualization warning: {e}")

    return res

if __name__ == "__main__":
    print("Module loaded. Call 'analyze_trend_from_excel(path)' to run.")
