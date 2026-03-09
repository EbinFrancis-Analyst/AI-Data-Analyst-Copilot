"""
analytics_engine.py  v5
Cached Analytics Backend

New in v5:
  BUG-1  resolve_column_ambiguity(df, keywords)
           Returns one column name, a list of candidates (ambiguous), or None (no match).
  BUG-2  get_dataset_schema(df)
           Returns a validated schema string — column names that EXIST in df only.
  BUG-3  clean_numeric_columns(df)
           Coerces columns that are ≥70% numeric values to float, strips $ / commas first.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-3 — Numeric Type Fixer
# Must run BEFORE any analytics so dirty "object" columns become float.
# ─────────────────────────────────────────────────────────────────────────────

def clean_numeric_columns(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """
    For every object column whose values are ≥ *threshold* numeric (after
    stripping common non-numeric noise like '$', ',', '%', whitespace),
    coerce the column to float with pd.to_numeric(errors='coerce').

    This fixes columns loaded as object that contain dirty values such as:
        100, 200, "missing", "$400", "1,200.50"

    Args:
        df:         Source DataFrame (not mutated — a copy is returned).
        threshold:  Fraction of non-null values that must parse as numeric.

    Returns:
        New DataFrame with offending object columns coerced to float.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            continue
        # Strip common currency/formatting characters before testing
        cleaned = s.str.replace(r"[$,€£¥%\s]", "", regex=True)
        numeric  = pd.to_numeric(cleaned, errors="coerce")
        ratio    = numeric.notna().sum() / len(cleaned)
        if ratio >= threshold:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[$,€£¥%\s]", "", regex=True),
                errors="coerce",
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BUG-2 — Dataset Schema Validator
# ─────────────────────────────────────────────────────────────────────────────

# Maps pandas dtype categories to human-readable type labels
_DTYPE_LABELS: Dict[str, str] = {
    "int64": "integer", "int32": "integer", "int16": "integer", "int8": "integer",
    "float64": "float",  "float32": "float",
    "bool":   "boolean",
    "object": "string",  "string": "string",
    "category": "categorical",
}


def get_dataset_schema(df: pd.DataFrame) -> str:
    """
    Return a validated, human-readable schema string listing ONLY columns
    that actually exist in *df*.  Nothing is fabricated.

    Example output:
        Dataset Schema (12 columns):
        • order_id       — integer
        • customer_name  — string
        • net_margin     — float
        • city           — categorical

    This string is used as a guard-rail: any code that needs to reference
    columns should first call this and match against df.columns.
    """
    lines = [f"Dataset Schema ({len(df.columns)} columns):"]
    for col in df.columns:
        raw   = str(df[col].dtype)
        label = _DTYPE_LABELS.get(raw, raw)
        # Datetime variants
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            label = "datetime"
        lines.append(f"  • {col:<30} — {label}")
    return "\n".join(lines)


def validate_column(df: pd.DataFrame, col_name: str) -> bool:
    """
    Return True only if *col_name* is an actual column in *df*.
    Use this before every column reference to prevent schema hallucination.
    """
    return col_name in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# BUG-1 — Column Ambiguity Resolver
# ─────────────────────────────────────────────────────────────────────────────

# Return type:  str           → unambiguous single match
#               List[str]     → multiple candidates (caller must ask user)
#               None          → no match found
AmbiguityResult = Union[str, List[str], None]


def resolve_column_ambiguity(
    df: pd.DataFrame,
    keywords: List[str],
    dtype_filter: Optional[str] = None,
) -> AmbiguityResult:
    """
    Find all columns in *df* that match any keyword in *keywords*.

    Rules
    -----
    1. Collect every column whose lowercased name contains any keyword.
    2. Optionally filter by dtype_filter: "numeric" | "categorical" | "datetime".
    3. If exactly ONE match  → return the column name (str).
    4. If MULTIPLE matches   → return the full list (List[str])  ← caller must
                               show the user a disambiguation message.
    5. If NO match           → return None.

    Args:
        df:           The DataFrame to inspect.
        keywords:     List of keyword strings to search for.
        dtype_filter: Optional dtype constraint.

    Returns:
        str | List[str] | None
    """
    cl = {c.lower(): c for c in df.columns}
    candidates: List[str] = []

    for kw in keywords:
        kw_lower = kw.lower()
        for col_lower, col_orig in cl.items():
            if kw_lower in col_lower and col_orig not in candidates:
                candidates.append(col_orig)

    # Apply dtype filter
    if dtype_filter == "numeric":
        candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    elif dtype_filter == "categorical":
        candidates = [c for c in candidates
                      if not pd.api.types.is_numeric_dtype(df[c])
                      and not pd.api.types.is_datetime64_any_dtype(df[c])]
    elif dtype_filter == "datetime":
        candidates = [c for c in candidates
                      if pd.api.types.is_datetime64_any_dtype(df[c])]

    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return candidates   # ambiguous — return all so caller can ask user


def ambiguity_message(candidates: List[str]) -> str:
    """
    Build the user-facing disambiguation message for NL chart generator and
    any other caller that receives a List[str] from resolve_column_ambiguity().
    """
    bullets = "\n".join(f"  • {c}" for c in candidates)
    return (
        f"Multiple columns match your request:\n{bullets}\n\n"
        f"Please specify which column you want by including its exact name."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal column finders  (updated to use resolve_column_ambiguity)
# ─────────────────────────────────────────────────────────────────────────────

def _find(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    """
    Return the FIRST unambiguous match, or None if ambiguous / not found.
    Ambiguous results are silently resolved by taking the first candidate
    (safe for internal KPI computation where a best-effort pick is fine).
    """
    result = resolve_column_ambiguity(df, kws)
    if isinstance(result, list):
        return result[0]   # best-effort for internal use
    return result          # str or None


def _num(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    result = resolve_column_ambiguity(df, hints, dtype_filter="numeric")
    if isinstance(result, list):
        return result[0]
    if isinstance(result, str):
        return result
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None


def _cat(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    result = resolve_column_ambiguity(df, hints, dtype_filter="categorical")
    if isinstance(result, list):
        return result[0]
    if isinstance(result, str):
        return result
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


# ─────────────────────────────────────────────────────────────────────────────
# Core analytics — all @st.cache_data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute top-level business KPIs. Cached per unique df hash."""
    sales_col = _num(df, ["price", "sales", "revenue", "amount", "total", "value"])
    qty_col   = _num(df, ["quantity", "qty", "units", "volume"])
    cust_col  = _cat(df, ["customer", "client", "buyer", "user", "cust"])
    prod_col  = _cat(df, ["product", "item", "goods", "sku", "category"])
    order_col = _cat(df, ["order", "order_id", "invoice", "transaction"])
    city_col  = _cat(df, ["city", "region", "location", "state", "area", "zone"])
    date_col  = _date(df)

    # --- schema guard: only reference columns that exist ---
    def _safe(col: Optional[str]) -> Optional[str]:
        return col if col and validate_column(df, col) else None

    sales_col = _safe(sales_col)
    qty_col   = _safe(qty_col)
    cust_col  = _safe(cust_col)
    prod_col  = _safe(prod_col)
    order_col = _safe(order_col)
    city_col  = _safe(city_col)
    date_col  = _safe(date_col)

    kpis: Dict[str, Any] = {
        "total_rows":        len(df),
        "total_cols":        len(df.columns),
        "memory_mb":         round(df.memory_usage(deep=True).sum() / 1024**2, 1),
        "completeness":      round((1 - df.isna().sum().sum() / max(df.size, 1)) * 100, 1),
        "schema":            get_dataset_schema(df),
        "sales_col":         sales_col,
        "qty_col":           qty_col,
        "cust_col":          cust_col,
        "prod_col":          prod_col,
        "order_col":         order_col,
        "city_col":          city_col,
        "date_col":          date_col,
        "total_revenue":     "N/A",
        "total_revenue_raw": 0.0,
        "aov":               "N/A",
        "aov_raw":           0.0,
        "total_orders":      f"{len(df):,}",
        "unique_customers":  "N/A",
        "unique_products":   "N/A",
        "unique_regions":    "N/A",
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
        kpis["unique_products"]  = f"{df[prod_col].nunique():,}"
    if city_col:
        kpis["unique_regions"]   = f"{df[city_col].nunique():,}"

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
        rows.append({
            "Column":        col,
            "Type":          str(s.dtype),
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
    """Cached generic groupby. Validates columns exist before operating."""
    # BUG-2 guard: never operate on columns that don't exist
    if not validate_column(df, group_col) or not validate_column(df, value_col):
        logger.warning("group_by_col: column not found — %s / %s", group_col, value_col)
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
    """Monthly revenue trend with 3-month moving average."""
    date_col  = _date(df)
    sales_col = _num(df, ["price", "sales", "revenue", "amount", "total"])
    if not date_col or not sales_col:
        return None
    if not validate_column(df, date_col) or not validate_column(df, sales_col):
        return None
    tmp = df[[date_col, sales_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return None
    monthly = tmp.resample("ME", on=date_col)[sales_col].sum().reset_index()
    monthly.columns = ["month", "revenue"]
    monthly["ma3"]        = monthly["revenue"].rolling(3, min_periods=1).mean()
    monthly["mom_growth"] = monthly["revenue"].pct_change() * 100
    return monthly
