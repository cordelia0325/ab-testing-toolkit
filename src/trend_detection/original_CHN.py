# 1) imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from scipy.stats import norm, t as tdist, rankdata
from functools import lru_cache

# === 核心层：M-K/HR98 Trend test ===

# 2) 常量
# -------- 小样本右尾精确表 (Gilbert 1987) --------
MK_SMALLSAMPLE_RIGHTTAIL = {
    4: {0:0.625, 2:0.375, 4:0.167, 6:0.042},
    5: {0:0.592, 2:0.408, 4:0.242, 6:0.117, 8:0.042, 10:0.0083},
    6: {1:0.500, 3:0.360, 5:0.235, 7:0.136, 9:0.068, 11:0.028, 13:0.0083, 15:0.0014},
    7: {1:0.500, 3:0.386, 5:0.281, 7:0.191, 9:0.119, 11:0.068, 13:0.035, 15:0.015, 17:0.0054, 19:0.0014, 21:0.00020},
    8: {0:0.548, 2:0.452, 4:0.360, 6:0.274, 8:0.199, 10:0.138, 12:0.089, 14:0.054, 16:0.031, 18:0.016, 20:0.0071, 22:0.0028, 24:0.00087, 26:0.00019, 28:0.000025},
    9: {0:0.540, 2:0.460, 4:0.381, 6:0.306, 8:0.238, 10:0.179, 12:0.130, 14:0.090, 16:0.060, 18:0.038, 20:0.022, 22:0.012, 24:0.0063, 26:0.0029, 28:0.0012, 30:0.00043, 32:0.00012, 34:0.000025, 36:0.0000028},
    10:{1:0.500, 3:0.431, 5:0.364, 7:0.300, 9:0.242, 11:0.190, 13:0.146, 15:0.108, 17:0.078, 19:0.054, 21:0.036, 23:0.023, 25:0.014, 27:0.0083, 29:0.0046, 31:0.0023, 33:0.0011, 35:0.00047, 37:0.00018, 39:0.000058, 41:0.000015, 43:0.0000028, 45:0.00000028}
}

# 3) 通用工具

# p 值格式化 (print 结果时用)
def _fmt_p(p: float, decimals: int = 3, tiny: float = 1e-3) -> str:
    if not np.isfinite(p):
        return "NA"
    return f"<{tiny:.{decimals}f}" if p < tiny else f"{p:.{decimals}f}"

# 4) M-K 基础工具
def _mk_S(x, eps=0.0):
    """O(n^2) 计算 S；|Δ|<=eps 视作 0，抗微小抖动"""
    x = np.asarray(x)
    S = 0
    for i in range(len(x)-1):
        d = x[i+1:] - x[i]
        if eps > 0:
            S += np.sum(np.where(np.abs(d) <= eps, 0, np.sign(d)))
        else:
            S += np.sign(d).sum()
    return int(S)

def _var_s_with_ties(x):
    """含 ties 的 Var(S)"""
    x = np.asarray(x)
    n = len(x)
    _, count = np.unique(x, return_counts=True)
    tie_term = np.sum(count*(count-1)*(2*count+5))
    var = (n*(n-1)*(2*n+5) - tie_term) / 18.0
    return float(var), count

def _table_right_tail(n, s_abs):
    """查表：返回右尾 p 及对齐后的 |S|；若无表或无键返回 None"""
    col = MK_SMALLSAMPLE_RIGHTTAIL.get(n)
    if not col: 
        return None
    M = n*(n-1)//2
    # 奇偶对齐 (S 与 M 同奇偶)
    if (s_abs%2) != (M%2):
        s_abs -= 1
    base = 0 if (M % 2 == 0) else 1
    s_abs = max(s_abs, base)
    # 取 <= s_abs 的最大键（保守）
    keys = sorted(k for k in col if (k%2) == (M%2))
    k_use = max([k for k in keys if k <= s_abs], default=keys[0])
    return float(col[k_use]), int(s_abs)

# 5) Sen 以及置信区间
def _sen_slope(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return np.array([], float), float('nan')
    pieces = []
    for i in range(n-1):
        d = x[i+1:] - x[i]
        h = np.arange(1, len(d)+1, dtype=float)
        s = d / h
        s = s[np.isfinite(s)]
        if s.size:
            pieces.append(s)
    if not pieces:
        return np.array([], float), float('nan')
    slopes = np.concatenate(pieces)
    slopes.sort(kind='mergesort')
    sen_hat = float(np.median(slopes))
    return slopes, sen_hat

def _sen_C_from_exact_table(n, alpha):
    """
    给 n∈[4,10] 且无并列时，从精确右尾表取 C：
    规则：与 M=nC2 同奇偶的支持点中，取 P_right<=alpha/2 的“最小 C”。
    找不到则取刚刚超过 alpha/2 的最大键（更保守）。
    """
    col = MK_SMALLSAMPLE_RIGHTTAIL.get(int(n))
    if not col:
        return None
    M = n*(n-1)//2
    keys = sorted(k for k in col if (k%2) == (M%2))
    target = alpha/2.0
    cand = [k for k in keys if col[k] <= target]
    if cand:
        return float(min(cand))         # 满足条件的最小 C
    bigger = [k for k in keys if col[k] > target]
    return float(max(bigger)) if bigger else float(keys[-1])
  
def _sen_C_from_var(var, alpha):
    """
    正态近似：C = z * sqrt(varS)
    注：传入的 var 可以是 HR98 修正后的 var_hr，也可以是原始 var0（未修正）（根据最后传入的var）。
    """
    if not np.isfinite(var) or var <= 0:
        return 0.0
    z = norm.ppf(1-alpha/2.0)
    return float(z*np.sqrt(var))

def _sen_confidence_interval(slopes_sorted, C):
    slopes = np.asarray(slopes_sorted, float)
    M = slopes.size
    # 秩次（1-based）
    L = int(np.floor((M-C)/2.0))
    U = int(np.ceil((M+C)/2.0))
    # 再裁剪到 [0, M-1] (0-based)
    L = max(1, min(L, M))-1
    U = max(1, min(U, M))-1
    low, high = float(slopes[L]), float(slopes[U])
    if low > high: 
        low, high = high, low
    return (low, high)

# 6) HR98 工具

# 口径说明：
# - alpha：用于 M-K 检验的“单尾”显著性阈值（右尾 p_right < alpha 判显著）。
# - CI：使用同一个 alpha 构造“双侧 (1 - alpha) 置信区间”，即 z_{1 - alpha/2}。
#   * 例如 alpha=0.10 → 90% CI；alpha=0.05 → 95% CI。
# - alpha_acf：用于 HR98 自相关筛选的“单尾”阈值（只关心正自相关）。
#   * 单尾（正向）是因为正自相关会低估 Var(S)，需要膨胀；负自相关通常不膨胀。
#   * alpha_acf 越小 → 入选滞后越少 → 方差膨胀越弱；越大则越保守（膨胀更强）。

@lru_cache(maxsize=None)
def _rcrit(n_eff, alpha_acf):
    df = max(3, int(n_eff)-2)
    tcrit = tdist.ppf(1-alpha_acf, df)  # 单尾
    return tcrit/np.sqrt(tcrit**2 + df)

def _hr98_var(x, var0, sen_hat, *, alpha_acf=0.05, lag_set=None, max_lag=None, use_rank=True, round_decimals=12):
    """Hamed & Rao (1998) 自相关修正：返回 var_hr, gamma, acf_used"""
    n = len(x)
    if n <= 10 or var0 <= 0:
        return var0, 0.0, [], sen_hat

    # 若外部未算 sen_hat（比如 run_hr98=True 但 run_sen=False），这里再算一次
    if sen_hat is None:
        _, sen_hat = _sen_slope(x)

    t = np.arange(n, dtype=float)
    resid = np.round(x-sen_hat*t, round_decimals)
    
    x_acf = rankdata(resid, method="average") if use_rank else resid
    
    # 选择滞后
    if lag_set is not None:
        lags = sorted({k for k in lag_set if 1 <= k < n})
    else:
        if max_lag is None:
            if n < 14: max_lag = 0
            elif n < 21: max_lag = 1
            else: max_lag = int(min(10*np.log10(n), n//4))
        else:
            max_lag = min(int(max_lag), n//2)
        lags = list(range(1, max_lag+1))

    gamma, acf_used = 0.0, []
    for k in lags:
        m = n - k
        if m < 5: break
        a, b = x_acf[:m], x_acf[k:]
        # 手写 Pearson，避免创建 2x2 矩阵
        am, bm = a.mean(), b.mean()
        da, db = a-am, b-bm
        den = np.sqrt(np.dot(da,da)*np.dot(db,db))
        if den == 0: continue
        rho = float(np.dot(da,db)/den)
        if not np.isfinite(rho): continue
        if rho > _rcrit(m, alpha_acf):   # 只纳入正且显著的自相关，使得检测更保守
            gamma += m*(m-1)*(m-2)*rho
            acf_used.append((k, float(rho)))

    C = max(1.0, 1.0 + 2.0*gamma/(n*(n-1)*(n-2)))
    return float(var0*C), float(gamma), acf_used, sen_hat

# 7) 主检验
def mk_test(series, *, alpha=0.10, alpha_acf=0.05, run_hr98=True, run_sen=True, 
            max_lag=None, lag_set=None, use_rank=True, eps=0.0, round_decimals=12):
    """
    统一版 Mann–Kendall：
      - n∈[4,10] 且无并列 → 精确查表（右尾）
      - 否则 → 正态近似；若 hr98=True → HR98 自相关修正
    """
    x = np.asarray(series, dtype=np.float64)
    n = len(x)

    # 兜底校验，避免直接调用 mk_test 时 NaN/长度不足引发异常
    if (n < 4) or (not np.all(np.isfinite(x))):
        return dict(p_value=None, trend='数据不足或存在缺失/非数值(NaN/Inf)', direction=None, method='invalid_input',
        n=n, S=None, Z=None, var_s_raw=None, var_s_hr=None, gamma=0.0, acf_used=[])

    S = _mk_S(x, eps=eps)
    if S == 0:
        return dict(p_value=1.0, trend='无明显趋势', direction=None, 
                    n=n, S=0, Z=0.0, method='degenerate')

    direction = 'increasing' if S > 0 else 'decreasing'
    s_abs = abs(S)
    
    var0, count = _var_s_with_ties(x)
    has_ties = np.any(count > 1)

    sen_hat, slopes_sorted = None, None

    C_exact = None
    sen_ci = None
    
    # ---------------- 小样本查表路径 ----------------
    if (n in MK_SMALLSAMPLE_RIGHTTAIL) and (not has_ties):
        out = _table_right_tail(n, s_abs)
        if out is not None:
            p_value, used_s = out

            if run_sen:
                slopes_sorted, sen_hat = _sen_slope(x)
                C_exact = _sen_C_from_exact_table(n, alpha)
                sen_ci = _sen_confidence_interval(slopes_sorted, C_exact)
                
            trend = ('显著上升▲' if (direction=='increasing' and p_value < alpha)
                     else '显著下降▼' if (direction=='decreasing' and p_value < alpha)
                     else '无明显趋势')
            return dict(p_value=p_value, trend=trend, direction=direction, 
                        sen_slope=sen_hat, conf_int=sen_ci,
                        method='exact_lookup_no_ties',
                        n=n, S=S, Z=None, used_S_for_table=used_s, var_s_raw=None, var_s_hr=None,
                        gamma=0.0, acf_used=[])

    # ---------------- 正态近似路径（含 HR98 可选） ----------------
    var = var0
    gamma, acf_used = 0.0, []

    # 若需要输出 Sen（置信区间等），先在外面算一次，给 HR98 复用
    if run_sen:
        slopes_sorted, sen_hat = _sen_slope(x)
        
    # 若 sen_hat 已有就直接用；没有才在内部计算一次（保证不重复计算 Sen）
    if run_hr98:
        var, gamma, acf_used, sen_used = _hr98_var(x, var0, sen_hat, alpha_acf=alpha_acf, 
                                                   lag_set=lag_set, max_lag=max_lag, 
                                                   use_rank=use_rank, round_decimals=round_decimals)
        if sen_hat is None:  # 仅在未计算过时接收 HR98 内部的计算结果
            sen_hat = sen_used
        
    # 避免 var<=0 的极端情况
    if var <= 0:
        return dict(p_value=1.0, trend='无明显趋势',
                    direction=direction, method='degenerate_var',
                    n=n, S=S, Z=None, var_s_raw=var0, var_s_hr=None, gamma=0.0, acf_used=[])
     
    # 统一用 |S| 计算 Z（连续性校正）
    Z = (s_abs - 1)/np.sqrt(var)
    p_value = float(np.clip(1 - norm.cdf(Z), 0.0, 1.0))      # 右尾

    C_norm = None
    if run_sen:
        C_norm = _sen_C_from_var(var, alpha)
        sen_ci = _sen_confidence_interval(slopes_sorted, C_norm)
    
    trend = ('显著上升▲' if (direction=='increasing' and p_value < alpha)
             else '显著下降▼' if (direction=='decreasing' and p_value < alpha)
             else '无明显趋势')

    return dict(p_value=p_value, trend=trend, direction=direction,
                sen_slope=sen_hat, conf_int=sen_ci,
                method=('normal_approx_with_ties_hr98' if run_hr98 else 'normal_approx_with_ties'),
                n=n, S=S, Z=Z, var_s_raw=var0, var_s_hr=(var if run_hr98 else None),
                gamma=gamma, acf_used=acf_used)

# 8) IO / 平台 / 可视化

# === 入口层：平台规则 & 读数 ===
# ---------------- 检验窗口规则 ----------------
def _validate_window(y, *, min_len=4, require_no_nan=True):
    y = np.asarray(y, dtype=float)
    if require_no_nan and (not np.all(np.isfinite(y))):
        return False, "目标窗口内存在 NaN/Inf。"
    if y.size < min_len:
        return False, f"目标窗口长度需≥{min_len}天。"
    return True, None

# ---------------- 读取表格数据 ----------------
def _to_series_from_excel(xlsx_path: str,
                          sheet_name=0,      # 表的序号，默认为第一张（或是唯一的那一张）。
                          date_col=None,     # 可传列名（推荐）、列号、或留空自动识别
                          name_col=None,     # 指标中文名列
                          value_col=None):     # 相对提升列
    """
    读取 xlsx（单表）：包含日期、指标中文名、相对提升（可能有 NaN）。
    返回：(series, metric_label, df_raw)
      - series: DatetimeIndex 升序、去 NaN 的相对提升序列
      - metric_label: 指标中文名（整列取第一个非空做标签）
      - df_raw: 原始 DataFrame（便于调试）
    支持列名 / 列号 / 自动识别（模糊匹配）。
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    cols = list(df.columns)

    def _resolve_col(spec, role, candidates_regex):
        """把‘列名或列号或None’解析成实际列名；None 时走自动识别"""
        # 列号
        if isinstance(spec, int):
            if spec < 0 or spec >= len(cols):
                raise IndexError(f"{role} 列号越界：{spec}（共有 {len(cols)} 列）")
            return cols[spec]
        # 明确列名
        if isinstance(spec, str):
            if spec in df.columns:
                return spec
            # 试下去空白匹配
            trimmed = [c.strip() for c in df.columns]
            mapping = dict(zip(trimmed, df.columns))
            if spec.strip() in mapping:
                return mapping[spec.strip()]
            raise KeyError(f"{role} 列名不存在：{spec}；现有列：{cols}")

        # 自动识别（按正则在列名里模糊匹配）
        pattern = re.compile(candidates_regex, re.I)
        for c in cols:
            if isinstance(c, str) and pattern.search(c.strip()):
                return c
        raise KeyError(f"自动识别失败：找不到 {role} 列。现有列：{cols}")

    # 自动识别模式的候选正则（可按你的表头再加关键字）
    date_name = _resolve_col(date_col, "日期", r"date|日期|时间|day|dt")
    name_name = _resolve_col(name_col, "指标中文名", r"指标名|指标中文名|metric|name")
    value_name= _resolve_col(value_col, "相对提升", r"相对|提升|diff|rel|ratio|变化|增幅|improv|value|y|目标值")

    # 规范数据
    df = df.copy()
    df[date_name] = pd.to_datetime(df[date_name], errors="coerce")

    # 显式检查是否存在非法日期
    bad_date_mask = df[date_name].isna()
    if bad_date_mask.any():
        # 列出前若干条坏行，清晰报错
        bad_rows = df.loc[bad_date_mask].head(10)
        examples = bad_rows[[date_name]].to_dict(orient="records")
        raise ValueError(f"日期列存在无法解析的值（共 {bad_date_mask.sum()} 行），例如：{examples}。")

    df = df.sort_values(by=date_name)

    # 指标中文名：整列一般相同，取第一个非空
    metric_label = str(df[name_name].dropna().iloc[0]) if df[name_name].notna().any() else str(name_name)

    # 相对提升序列（去 NaN）
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    s = pd.Series(df[value_name].values, index=df[date_name])
    s = s.replace([float("inf"), float("-inf")], pd.NA) 
    s.index = pd.to_datetime(s.index)             # 保底：确保是 DatetimeIndex
    s = s.sort_index()

    return s, metric_label, df

# === 模拟 AB 平台 ===
def mk_trend_platform(xlsx_path: Optional[str] = None,
                      series: Optional[pd.Series] = None,   # 支持两种入口：传 xlsx_path 或直接传 series（优先使用 series，便于测试）。
                      *,
                      # 读取xlsx
                      sheet_name: int = 0,    # 表的序号，默认为第一张（或是唯一的那张）。
                      date_col=None, name_col=None, value_col=None,
                      # 窗口
                      start_date: Optional[pd.Timestamp] = None,  # 目标检测窗口的开始日期，None 默认用 xlsx 中的最早一天
                      end_date: Optional[pd.Timestamp] = None,    # 目标检测窗口的结束日期，None 默认用 xlsx 中的最后一天
                      # 显著性
                      alpha: float = 0.10,
                      # 控制项 （默认使用HR98修正，并且计算Sen's slope以及置信区间）
                      run_hr98: bool = True,
                      run_sen: bool = True
                      ) -> Dict[str, Any]:

    # ---------------- 1) 读取全量数据 ----------------
    # 取数据 series 优先
    if series is not None:
        s_all = pd.Series(series, dtype="float64").copy()

        # 先尝试把索引解析成日期
        idx_try = pd.to_datetime(s_all.index, errors="coerce")
        
        if idx_try.isna().all():
            # 索引完全不是日期：按“日粒度等间隔”自动构造合成日期
            # 起始随便定一个“固定锚点”，不影响检验（关键是等间隔 & 有序）
            idx = pd.date_range("2000-01-01", periods=len(s_all), freq="D")
        elif idx_try.isna().any():
            # 部分可解析、部分不行：给出清晰报错，避免悄悄丢点
            return dict(ok=False, message="series 索引部分不是有效日期：请全部用 DatetimeIndex，或改为纯数值索引让函数按天自动补索引。")
        else:
            idx = idx_try
            
        # 索引去重检查（避免重复日期导致切窗&排序歧义）
        if pd.Index(idx).has_duplicates:
            return dict(ok=False, message="series 索引存在重复时间点：请先去重或聚合。")
            
        s_all.index = pd.to_datetime(s_all.index, errors="coerce")
        s_all = s_all.sort_index()
        metric_label = "metric"
        
    else:
        if not xlsx_path:
            return dict(ok=False, message="必须提供 series 或 xlsx_path。")
        try:
            s_all, metric_label, _df = _to_series_from_excel(
                xlsx_path,
                sheet_name=sheet_name,
                date_col=date_col, name_col=name_col, value_col=value_col
            )
        except Exception as e:
            return dict(ok=False, message=f"读取 xlsx 失败：{e}。")
      
    if s_all.empty:
        return dict(ok=False, message="空序列：没有可用数据点。")

    # ---------------- 2) 切目标窗，校验（长度≥4、窗口内无缺失数据） ----------------
    # 默认窗口为全部区间
    data_min = s_all.index.min().normalize()
    data_max = s_all.index.max().normalize()

    end_date = data_max if end_date is None else pd.to_datetime(end_date).normalize()
    start_date = data_min if start_date is None else pd.to_datetime(start_date).normalize()
    # 备选：若未给 start_date，可让上层传入窗口长度；这里示例：默认14天
    # start_date = end_date - pd.Timedelta(days=13)
   
    # 判断时间先后是否有效
    if start_date > end_date:
        return dict(ok=False,
                    message=f"目标窗口起止时间顺序错误：开始日期 {start_date.date()} 不应晚于结束日期 {end_date.date()}。",
                    window=dict(metric=metric_label, start=data_min.date(), end=data_max.date(), n=0)
                   )
        
    # 做越界检查
    if start_date < data_min:
        return dict(ok=False,
                    message=f"目标窗口开始时间超出范围，应不早于 {data_min.date()}。",
                    window=dict(metric=metric_label, start=data_min.date(), end=data_max.date(), n=0)
                   )
    if end_date > data_max:
        return dict(ok=False,
                    message=f"目标窗口结束时间超出范围，应不晚于 {data_max.date()}。",
                    window=dict(metric=metric_label, start=data_min.date(), end=data_max.date(), n=0)
                   )
    
    # 生成目标窗
    mask = (s_all.index.normalize() >= start_date) & (s_all.index.normalize() <= end_date)
    s_target = s_all.loc[mask].sort_index()

    n = len(s_target)

    if s_target.empty:
        return dict(ok=False, message="目标窗口内无数据。", window=dict(metric=metric_label, start=start_date.date(), end=end_date.date(), n=0))

    ok, msg = _validate_window(s_target, min_len=4, require_no_nan=True)
    if not ok:
        return dict(ok=False, message=msg, window=dict(metric=metric_label, start=start_date.date(), end=end_date.date(), n=n))

    # ---------------- 3) 分流做 M-K 趋势检验 ----------------
    # 4–10 天：走精确表（如果存在并列项，则自动回退正态近似）
    if 4 <= n <= 10:
        mk_res = mk_test(s_target, alpha=alpha,
                         run_hr98=False, run_sen=run_sen)  # 小样本不做 HR98
    else:
        # >10 天：走 HR98
        mk_res = mk_test(s_target, alpha=alpha, run_hr98=run_hr98, run_sen=run_sen)

    # ---------------- 4) 统一结果结构 ----------------
    # 输出所有信息，方便调用 analyze_trend_from_excel 帮助人为调取信息分析
    out: Dict[str, Any] = dict(
        ok=True,
        window=dict(metric=metric_label,
                    start=start_date.date(), end=end_date.date(), n=n),
        mk=dict(
            S=mk_res.get("S"), Z=mk_res.get("Z"),
            p_value=mk_res.get("p_value"),
            p_str=_fmt_p(mk_res.get("p_value")) if mk_res.get("p_value") is not None else "nan",      # 目标数据
            trend=mk_res.get("trend"),  # 目标数据
            method=mk_res.get("method"),
            direction=mk_res.get("direction"),
        ),
        sen=dict(
            sen_slope=mk_res.get("sen_slope"), # 当不计算 Sen’s slope 时，sen_slope/ci 统一为 None
            ci=mk_res.get("conf_int"),
        ),
        hr98=dict(
            var_raw=mk_res.get("var_s_raw"),
            var_hr=mk_res.get("var_s_hr"),
            gamma=mk_res.get("gamma"),
            acf_used=mk_res.get("acf_used"),
        ),
        data=dict(all=s_all, target=s_target),
    )
    '''
    # 只输出核心信息（p值，检测结果，Sen's slope，置信区间），如果只是调用 mk_trend_platform 模拟平台行为，就只需要这个
    out : Dict[str, Any] = dict(
        mk=dict(
            p_str=_fmt_p(mk_res.get("p_value")) if mk_res.get("p_value") is not None else "nan", 
            trend=mk_res.get("trend"),
        ),
        sen=dict(
            sen_slope=mk_res.get("sen_slope"), 
            ci=mk_res.get("conf_int"), 
        ),
    )
    '''
    return out

# === 内部分析用 ===
def analyze_trend_from_excel(xlsx_path: str,   
                             *,
                             # 读取 xlsx
                             sheet_name = 0,    # 表的序号，默认为第一张（或是唯一的那一张）。
                             date_col = None, name_col = None, value_col =None,  # 既可传列名，也可传列号；None 表示自动识别
                             # 目标窗口
                             start_date: Optional[pd.Timestamp] = None,  # 开始日期；None 用 xlsx 中的最初一天
                             end_date: Optional[pd.Timestamp] = None,  # 结束日期；None 用 xlsx 中的最后一天
                             # 显著性水平
                             alpha: float = 0.10,
                             # 控制项 （默认使用 HR98 修正，并且计算 Sen's slope 以及置信区间）
                             run_hr98: bool = True,
                             run_sen: bool = True,
                             # 可视化
                             show_plot: bool = True
                            )-> Dict[str, Any]:  
    """
    内部查看/联调用途：调用平台 -> 打印结果 -> 可视化
    """
    res = mk_trend_platform(xlsx_path=xlsx_path, sheet_name=sheet_name,
                            date_col=date_col, name_col=name_col, value_col=value_col,
                            start_date=start_date, end_date=end_date,
                            alpha=alpha, run_hr98=run_hr98, run_sen=run_sen)

    if not res.get("ok"):
        print("[MK] 失败：", res.get("message"))
        return res

    # ------- 打印检测结果 -------
    W, MK, SEN, HR = res["window"], res["mk"], res["sen"], res["hr98"]
    print("\n=== 目标窗检测结果 ===")
    print(f"检测区间: {W['start']} ~ {W['end']} | n={W['n']} | 指标: {W['metric']}")
    
    z = MK.get("Z")
    z_str = f"{z:.3f}" if (z is not None and np.isfinite(z)) else "NA"
    
    var = HR.get('var_hr') if HR.get('var_hr') is not None else HR.get('var_raw')
    var_str = f"{var:.3f}" if (var is not None and np.isfinite(var)) else "NA"
    
    acf_str = "[" + ", ".join(f"(lag={k}, rho={r:.3f})" for k, r in HR.get('acf_used', [])) + "]"
    
    print(f"S={MK['S']}, Z={z_str}, Var={var_str}, used_rhos={acf_str}")
    
    if SEN.get("sen_slope") is not None and SEN.get("ci") is not None:
        low, high = SEN["ci"]
        print(f"Sen's slope={SEN['sen_slope']*100:.4g}%, CI=[{low*100:.4g}%, {high*100:.4g}%]")
    print(f"p={MK['p_str']}, trend={MK['trend']} ({MK['method']})")

    # 5) 画图
    if show_plot:
        plt.rcParams['font.sans-serif'] = ['STHeiti']
        plt.rcParams['axes.unicode_minus'] = False

        s_all, s_target = res["data"]["all"], res["data"]["target"]
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), tight_layout=True)

        # 上图：全局灰线 + 目标窗蓝线
        axes[0].plot(s_all.index, s_all.values * 100, '-', color='0.6', lw=1.5, label='全部数据', zorder=1)
        axes[0].plot(s_target.index, s_target.values * 100, '-', color='#1f77b4', lw=1.8, label='目标窗', zorder=3)

        axes[0].set_xlabel("日期")
        axes[0].set_ylabel("相对提升（%）")
        axes[0].axhline(0.0, color='k', linestyle=':', alpha=0.35)
        axes[0].set_title(
            f"全局 + 目标窗（{s_target.index.min().date()}~{s_target.index.max().date()}） | n = {len(s_target)} | 指标：{W['metric']}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)

        # 下图：仅目标窗 + Sen 趋势 + 均值
        axes[1].plot(s_target.index, s_target.values * 100, 'o-', ms=5.0, color='#1f77b4', lw=2.0, label='目标窗', zorder=3)
        axes[1].axhline(0.0, color='k', linestyle=':', alpha=0.35)

        mean_v = float(s_target.mean()) * 100.0
        axes[1].axhline(mean_v, color='#1f77b4', linestyle='--', alpha=0.6, label=f"均值 {mean_v:.4f}%")

        # 画 Sen 趋势线（用 Sen slope 与中位截距）
        sen_hat = SEN.get("sen_slope")
        if (sen_hat is not None) and np.isfinite(sen_hat):
            x_days = (s_target.index - s_target.index.min()).days.values.astype(float)
            intercept_med = float(np.median(s_target.values - sen_hat * x_days))
            y_hat = (intercept_med + sen_hat * x_days) * 100.0  # 转百分比显示
        else:
            y_hat = None

        if y_hat is not None:
            axes[1].plot(s_target.index, y_hat, "-", color='darkorange', lw=1.8, label=f"Sen's slope = {sen_hat*100:.4g}%/天")
            
        # 自适应 y 轴
        y_candidates = [float(s_target.min()*100), float(s_target.max()*100)]
        if y_hat is not None and len(y_hat) > 0:
            y_candidates += [float(np.nanmin(y_hat)), float(np.nanmax(y_hat))]
        ymin, ymax = min(y_candidates), max(y_candidates)
        rng = ymax - ymin if np.isfinite(ymax - ymin) and ymax > ymin else 1.0
        axes[1].set_ylim(ymin - 0.10*rng, ymax + 0.10*rng)

        # 标题 + MK 结果
        tbits1 = [f"目标窗（{s_target.index.min().date()}~{s_target.index.max().date()}） | n = {len(s_target)} | 指标：{W['metric']}"]
        mk_part = f"\nMann–Kendall: {MK.get('trend')} (p={MK.get('p_str')})"; tbits1.append(mk_part)
        axes[1].set_title(" ".join(tbits1));
        
        axes[1].set_xlabel("日期")
        axes[1].set_ylabel("相对提升（%）")
        axes[1].legend()
        axes[1].grid(True, axis='y', alpha=0.3)
        plt.show()

    return res

if __name__ == '__main__':
    df = pd.read_excel("social_produce_traffic_holdout_uidfirst_base5,exp6_LT7.xlsx")
    s = df.set_index("时间").loc["2025-01-12":"2025-01-25", "相对提升"]
    out = mk_test(s, run_hr98=True, run_sen=True)
    print(f"该序列的 p值为{out["p_value"]}, 检测结果是{out["trend"]}。拟合出的 Sen 斜率是{out["sen_slope"]}, 其置信区间为{out["conf_int"]}。")

if __name__ == '__main__':
    out = mk_test("social_produce_traffic_holdout_uidfirst_base5,exp6_LT7.xlsx", 
                            start_date="2025-01-12",        
                            end_date="2025-01-25", 
                            run_hr98=True, run_sen=True)
    print(f"该序列的 p值为{out["mk"]["p_str"]}, 检测结果是{out["mk"]["trend"]}。拟合出的 Sen 斜率是{out["sen"]["sen_slope"]}, 其置信区间为{out["sen"]["ci"]}。")

if __name__ == '__main__':
    out = analyze_trend_from_excel("social_produce_traffic_holdout_uidfirst_base5,exp6_LT7.xlsx", 
                                   start_date="2025-01-12",
                                   end_date="2025-01-25", 
                                   run_hr98=True, run_sen=True,
                                   show_plot=True)
