"""
analytics_engine.py
Cached Analytics Backend
All expensive computations live here and are memoised with st.cache_data
so they are only executed once per unique dataset.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _df_hash(df: pd.DataFrame) -> str:
    """Fast hash of a DataFrame for cache-key purposes."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def _find(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Return the first column whose lowercased name contains any keyword."""
    cl = {c.lower(): c for c in df.columns}
    for kw in keywords:
        for name_l, name in cl.items():
            if kw in name_l:
                return name
    return None


def _num(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _find(df, hints)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        return col
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None


def _cat(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _find(df, hints)
    if col and not pd.api.types.is_numeric_dtype(df[col]):
        return col
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cats[0] if cats else None


def _date(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(kw in col.lower() for kw in ["date", "time", "month", "year", "day"]):
            try:
                pd.to_datetime(df[col].dropna().astype(str).head(20), errors="raise")
                return col
            except Exception:
                pass
    return None


def _fmt(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:,.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Classify every column into one of:
    integer | float | boolean | datetime | categorical | string

    Returns:
        dict mapping column_name → type_label
    """
    types: Dict[str, str] = {}
    bool_vals = {"yes", "no", "true", "false", "1", "0", "y", "n"}

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_bool_dtype(s):
            types[col] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        elif pd.api.types.is_integer_dtype(s):
            types[col] = "integer"
        elif pd.api.types.is_float_dtype(s):
            types[col] = "float"
        elif hasattr(s, "cat"):                        # already category
            types[col] = "categorical"
        else:
            sample = s.dropna().astype(str).str.strip().str.lower().head(200)
            if sample.isin(bool_vals).mean() > 0.8:
                types[col] = "boolean"
            else:
                ratio = s.nunique() / max(len(s), 1)
                types[col] = "categorical" if ratio < 0.2 else "string"

    return types


# ─────────────────────────────────────────────────────────────────────────────
# KPI COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract top-level business KPIs from the dataset.

    Returns a flat dict with string values ready for display.
    """
    sales_col  = _num(df, ["price", "sales", "revenue", "amount", "total", "value"])
    qty_col    = _num(df, ["quantity", "qty", "units", "count", "volume"])
    cust_col   = _cat(df, ["customer", "client", "buyer", "user", "cust"])
    prod_col   = _cat(df, ["product", "item", "goods", "sku", "category"])
    order_col  = _cat(df, ["order", "order_id", "invoice", "transaction"])
    city_col   = _cat(df, ["city", "region", "location", "state", "area", "zone"])
    date_col   = _date(df)

    kpis: Dict[str, Any] = {}

    kpis["total_rows"]   = len(df)
    kpis["total_cols"]   = len(df.columns)
    kpis["memory_mb"]    = round(df.memory_usage(deep=True).sum() / 1024**2, 1)
    kpis["completeness"] = round((1 - df.isna().sum().sum() / max(df.size, 1)) * 100, 1)

    if sales_col:
        total  = df[sales_col].sum()
        orders = df[order_col].nunique() if order_col else len(df)
        kpis["total_revenue"]     = _fmt(total)
        kpis["total_revenue_raw"] = float(total)
        kpis["aov"]               = _fmt(total / max(orders, 1))
        kpis["aov_raw"]           = float(total / max(orders, 1))
        kpis["sales_col"]         = sales_col

    kpis["total_orders"]      = f"{df[order_col].nunique():,}" if order_col else f"{len(df):,}"
    kpis["unique_customers"]  = f"{df[cust_col].nunique():,}" if cust_col else "N/A"
    kpis["unique_products"]   = f"{df[prod_col].nunique():,}" if prod_col else "N/A"
    kpis["unique_regions"]    = f"{df[city_col].nunique():,}" if city_col else "N/A"

    kpis["cust_col"]   = cust_col
    kpis["prod_col"]   = prod_col
    kpis["city_col"]   = city_col
    kpis["qty_col"]    = qty_col
    kpis["date_col"]   = date_col
    kpis["order_col"]  = order_col

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION METRICS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def distribution_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Compute per-column distribution statistics (numeric only).
    Returns a dict of {col: {mean, median, std, min, max, q1, q3, skew, outliers}}.
    """
    result: Dict[str, Dict] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3  = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr     = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr else 0
        result[col] = {
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "max":      round(float(s.max()), 4),
            "q1":       round(q1, 4),
            "q3":       round(q3, 4),
            "skew":     round(float(s.skew()), 4),
            "outliers": outliers,
            "count":    int(len(s)),
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute Pearson correlation matrix for numeric columns.
    Returns None if fewer than 2 numeric columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return None
    return df[num_cols].corr().round(3)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def category_summaries(df: pd.DataFrame, top_n: int = 15) -> Dict[str, pd.DataFrame]:
    """
    Value-count summary for every categorical / string column.

    Returns {col: DataFrame[value, count, pct]}
    """
    result: Dict[str, pd.DataFrame] = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vc = df[col].value_counts(dropna=True).head(top_n)
        total = vc.sum()
        frame = pd.DataFrame({
            "value": vc.index,
            "count": vc.values,
            "pct":   (vc.values / max(total, 1) * 100).round(1),
        })
        result[col] = frame
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TREND ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def trend_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Compute monthly revenue trend.
    Returns DataFrame[month, revenue, mom_growth_pct, ma3] or None if no date col.
    """
    date_col  = _date(df)
    sales_col = _num(df, ["price", "sales", "revenue", "amount", "total"])

    if not date_col or not sales_col:
        return None

    tmp = df[[date_col, sales_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])

    monthly = (tmp.resample("ME", on=date_col)[sales_col]
                  .sum()
                  .reset_index())
    monthly.columns = ["month", "revenue"]
    monthly["ma3"]          = monthly["revenue"].rolling(3, min_periods=1).mean()
    monthly["mom_growth"]   = monthly["revenue"].pct_change() * 100
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# GROUPED AGGREGATIONS (for dashboard charts)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def group_by_col(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg: str = "sum",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Generic groupby helper used by dashboard and query engine.

    Args:
        df:        Source DataFrame
        group_col: Column to group by
        value_col: Column to aggregate
        agg:       Aggregation function (sum | mean | count | max | min)
        top_n:     Return only the top N rows by value

    Returns:
        DataFrame sorted descending, capped at top_n rows.
    """
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    agg_map = {"sum": "sum", "mean": "mean", "count": "count",
               "max": "max", "min": "min"}
    agg_fn = agg_map.get(agg, "sum")

    result = (df.groupby(group_col, observed=True)[value_col]
                .agg(agg_fn)
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
    result.columns = [group_col, value_col]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATASET QUALITY SCORE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def quality_score(df: pd.DataFrame) -> Tuple[int, str, str]:
    """
    Return (score 0-100, label, colour).
    Penalises missing values, duplicate rows, and constant columns.
    """
    rows, cols = df.shape
    if rows == 0:
        return 0, "No Data", "red"

    missing_ratio = df.isna().sum().sum() / max(df.size, 1)
    dup_ratio     = df.duplicated().sum() / max(rows, 1)
    const_cols    = sum(df[c].nunique() <= 1 for c in df.columns)

    score = 100
    score -= missing_ratio * 40
    score -= dup_ratio     * 20
    score -= const_cols    *  5
    score  = max(0, min(100, round(score)))

    if score >= 80:
        return score, "Good 🟢",     "green"
    elif score >= 60:
        return score, "Fair 🟡",     "orange"
    elif score >= 40:
        return score, "Poor 🟠",     "darkorange"
    return score, "Critical 🔴", "red"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN PROFILE TABLE
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-column summary table for display in the UI.

    Returns DataFrame with columns:
    Column | Type | Missing | Missing% | Unique | Completeness% | Sample Values
    """
    col_types = detect_column_types(df)
    rows = []
    for col in df.columns:
        s       = df[col]
        missing = int(s.isna().sum())
        total   = len(s)
        sample  = ", ".join(str(v) for v in s.dropna().unique()[:3])
        rows.append({
            "Column":        col,
            "Type":          col_types.get(col, "unknown"),
            "Missing":       missing,
            "Missing %":     f"{missing/total*100:.1f}%",
            "Unique":        int(s.nunique()),
            "Completeness":  f"{(total-missing)/total*100:.1f}%",
            "Sample Values": sample[:60],
        })
    return pd.DataFrame(rows)
