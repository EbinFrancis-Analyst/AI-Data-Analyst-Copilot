"""
analytics_engine.py
Cached Analytics Backend — all heavy computations cached with st.cache_data.
Handles datasets up to 500k rows efficiently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Column finders ─────────────────────────────────────────────────────────────

def _find(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    cl = {c.lower(): c for c in df.columns}
    for kw in kws:
        for nl, nc in cl.items():
            if kw in nl:
                return nc
    return None

def _num(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _find(df, hints)
    if c and pd.api.types.is_numeric_dtype(df[c]):
        return c
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None

def _cat(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    c = _find(df, hints)
    if c and not pd.api.types.is_numeric_dtype(df[c]):
        return c
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return cats[0] if cats else None

def _date(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(k in col.lower() for k in ["date", "time", "month", "year", "day"]):
            try:
                pd.to_datetime(df[col].dropna().head(10), errors="raise")
                return col
            except Exception:
                pass
    return None

def _fmt(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:,.2f}"


# ── Core analytics (all cached) ────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level KPIs. Cached per unique df."""
    sales_col = _num(df, ["price", "sales", "revenue", "amount", "total", "value"])
    qty_col   = _num(df, ["quantity", "qty", "units", "count", "volume"])
    cust_col  = _cat(df, ["customer", "client", "buyer", "user", "cust"])
    prod_col  = _cat(df, ["product", "item", "goods", "sku", "category"])
    order_col = _cat(df, ["order", "order_id", "invoice", "transaction"])
    city_col  = _cat(df, ["city", "region", "location", "state", "area", "zone"])
    date_col  = _date(df)

    kpis: Dict[str, Any] = {
        "total_rows":      len(df),
        "total_cols":      len(df.columns),
        "memory_mb":       round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "completeness":    round((1 - df.isna().sum().sum() / max(df.size, 1)) * 100, 1),
        "sales_col":  sales_col,
        "qty_col":    qty_col,
        "cust_col":   cust_col,
        "prod_col":   prod_col,
        "order_col":  order_col,
        "city_col":   city_col,
        "date_col":   date_col,
        "total_revenue":   "N/A",
        "total_revenue_raw": 0.0,
        "aov":             "N/A",
        "aov_raw":         0.0,
        "total_orders":    f"{len(df):,}",
        "unique_customers":"N/A",
        "unique_products": "N/A",
        "unique_regions":  "N/A",
    }

    if sales_col:
        total  = df[sales_col].sum()
        orders = df[order_col].nunique() if order_col else len(df)
        kpis["total_revenue"]     = _fmt(float(total))
        kpis["total_revenue_raw"] = float(total)
        kpis["aov"]               = _fmt(float(total / max(orders, 1)))
        kpis["aov_raw"]           = float(total / max(orders, 1))

    if order_col:
        kpis["total_orders"] = f"{df[order_col].nunique():,}"

    if cust_col:
        kpis["unique_customers"] = f"{df[cust_col].nunique():,}"

    if prod_col:
        kpis["unique_products"] = f"{df[prod_col].nunique():,}"

    if city_col:
        kpis["unique_regions"] = f"{df[city_col].nunique():,}"

    # Top performers
    if city_col and sales_col:
        grp = df.groupby(city_col, observed=True)[sales_col].sum()
        kpis["top_city"]     = str(grp.idxmax())
        kpis["top_city_rev"] = _fmt(float(grp.max()))

    if prod_col and sales_col:
        grp = df.groupby(prod_col, observed=True)[sales_col].sum()
        kpis["top_product"]     = str(grp.idxmax())
        kpis["top_product_rev"] = _fmt(float(grp.max()))

    return kpis


@st.cache_data(show_spinner=False)
def quality_score(df: pd.DataFrame) -> Tuple[int, str, str]:
    """Return (score 0-100, label, colour_hex)."""
    if len(df) == 0:
        return 0, "No Data 🔴", "#FF1744"
    missing_r = df.isna().sum().sum() / max(df.size, 1)
    dup_r     = df.duplicated().sum() / max(len(df), 1)
    const_c   = sum(df[c].nunique() <= 1 for c in df.columns)
    score = max(0, min(100, round(100 - missing_r * 40 - dup_r * 20 - const_c * 5)))
    if score >= 80: return score, "Good 🟢",     "#4CAF50"
    if score >= 60: return score, "Fair 🟡",     "#FFA500"
    if score >= 40: return score, "Poor 🟠",     "#FF5722"
    return score, "Critical 🔴", "#FF1744"


@st.cache_data(show_spinner=False)
def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column summary table."""
    rows = []
    for col in df.columns:
        s       = df[col]
        missing = int(s.isna().sum())
        total   = len(s)
        sample  = ", ".join(str(v) for v in s.dropna().unique()[:3])
        dtype   = str(s.dtype)
        rows.append({
            "Column":        col,
            "Type":          dtype,
            "Missing":       missing,
            "Missing %":     f"{missing/total*100:.1f}%",
            "Unique":        int(s.nunique()),
            "Completeness":  f"{(total-missing)/total*100:.1f}%",
            "Sample Values": sample[:60],
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def distribution_metrics(df: pd.DataFrame) -> Dict[str, Dict]:
    """Per-column numeric distribution stats."""
    result: Dict[str, Dict] = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr     = q3 - q1
        result[col] = {
            "mean":     round(float(s.mean()), 4),
            "median":   round(float(s.median()), 4),
            "std":      round(float(s.std()), 4),
            "min":      round(float(s.min()), 4),
            "max":      round(float(s.max()), 4),
            "q1":       round(q1, 4),
            "q3":       round(q3, 4),
            "skew":     round(float(s.skew()), 4),
            "outliers": int(((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()) if iqr else 0,
            "count":    int(len(s)),
        }
    return result


@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Pearson correlation matrix for numeric columns."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num) < 2:
        return None
    return df[num].corr().round(3)


@st.cache_data(show_spinner=False)
def group_by_col(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    agg: str = "sum",
    top_n: int = 15,
) -> pd.DataFrame:
    """Generic cached groupby used by dashboard + NL engine."""
    if group_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    fn = {"sum": "sum", "mean": "mean", "count": "count",
          "max": "max", "min": "min"}.get(agg, "sum")
    result = (df.groupby(group_col, observed=True)[value_col]
                .agg(fn)
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index())
    result.columns = [group_col, value_col]
    return result


@st.cache_data(show_spinner=False)
def trend_analysis(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Monthly revenue trend with MA3."""
    date_col  = _date(df)
    sales_col = _num(df, ["price", "sales", "revenue", "amount", "total"])
    if not date_col or not sales_col:
        return None
    tmp = df[[date_col, sales_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return None
    monthly = (tmp.resample("ME", on=date_col)[sales_col]
                  .sum().reset_index())
    monthly.columns = ["month", "revenue"]
    monthly["ma3"]       = monthly["revenue"].rolling(3, min_periods=1).mean()
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100
    return monthly
