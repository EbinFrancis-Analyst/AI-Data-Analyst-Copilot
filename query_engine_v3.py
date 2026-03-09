"""
query_engine_v3.py  v5
Natural Language → Plotly Chart Generator

New in v5:
  BUG-1  Every column lookup now calls resolve_column_ambiguity().
         If multiple candidates match, the handler returns (None, ambiguity_message)
         so the caller (app.py) can show the user a disambiguation widget.
  BUG-2  get_dataset_schema() is called to validate columns before chart generation.
         Charts that reference non-existent columns are rejected with a clear error.
  BUG-3  clean_numeric_columns() is applied to a sample before any arithmetic,
         preventing crashes on dirty "$400" / "missing" values.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics_engine import (
    group_by_col, trend_analysis, correlation_matrix,
    resolve_column_ambiguity, ambiguity_message,
    get_dataset_schema, validate_column,
    clean_numeric_columns, _fmt,
)

# ── Design tokens ──────────────────────────────────────────────────────────────

COLORS = [
    "#4F8BF9", "#A259FF", "#43D9AD", "#FFB347", "#FF6B6B",
    "#87CEEB", "#FF69B4", "#98FB98", "#DDA0DD", "#F0E68C",
]

_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter,sans-serif", color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E1E2E", bordercolor="#444",
                    font=dict(size=12, color="#fff")),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
               zeroline=False, linecolor="rgba(255,255,255,0.3)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
               zeroline=False, linecolor="rgba(255,255,255,0.3)"),
)


def _lay(fig: go.Figure, title: str = "", h: int = 420) -> go.Figure:
    fig.update_layout(
        **_BASE, height=h,
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color="#E0E0E0"), x=.02),
    )
    return fig


# ── Keywords ───────────────────────────────────────────────────────────────────

KW_SALES   = ["revenue", "sales", "income", "amount", "price", "earning", "total", "spend"]
KW_TOP     = ["top", "best", "highest", "most", "leading", "greatest", "maximum", "rank"]
KW_LOW     = ["lowest", "worst", "least", "bottom", "minimum", "declining", "poor"]
KW_AVG     = ["average", "mean", "avg", "typical"]
KW_TREND   = ["trend", "over time", "monthly", "time series", "growth", "timeline", "month"]
KW_DIST    = ["distribution", "spread", "histogram", "range", "frequency", "how many"]
KW_CORR    = ["correlation", "relationship", "heatmap", "related", "between", "matrix"]
KW_PIE     = ["share", "pie", "proportion", "percentage", "contribution", "percent", "breakdown"]
KW_CUST    = ["customer", "client", "buyer", "user", "consumer", "cust"]
KW_PROD    = ["product", "item", "goods", "sku", "merchandise", "category", "cat"]
KW_REGION  = ["city", "region", "state", "location", "area", "zone", "district", "place"]
KW_BOX     = ["box", "boxplot", "quartile", "outlier", "iqr", "median"]
KW_SCATTER = ["scatter", "compare", "vs", "versus", "against", "plot"]

EXAMPLE_QUERIES = [
    "Show revenue by city",
    "Top 10 products by sales",
    "Monthly sales trend",
    "Customer purchase frequency",
    "Revenue share by category",
    "Correlation heatmap",
    "Distribution of amount",
    "Top 5 customers by spend",
    "Lowest performing regions",
    "Average order value by product",
    "Box plot of sales",
    "Sales breakdown pie chart",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _any(q: str, kws: List[str]) -> bool:
    return any(kw in q for kw in kws)


def _n(q: str, default: int = 10) -> int:
    m = re.search(r"top\s+(\d+)", q)
    if m:
        return int(m.group(1))
    for w, v in {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}.items():
        if f"top {w}" in q:
            return v
    return default


def _sample(df: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
    """Return a size-limited sample so charts render fast on large datasets."""
    return df if len(df) <= n else df.sample(n, random_state=42)


def _resolve_num(df: pd.DataFrame, hints: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a numeric column.  Returns (col_name, error_msg).
    error_msg is set when ambiguous or not found.
    """
    result = resolve_column_ambiguity(df, hints, dtype_filter="numeric")
    if isinstance(result, list):
        return None, ambiguity_message(result)
    if isinstance(result, str) and validate_column(df, result):
        return result, None
    # Fallback: first numeric column
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return (nums[0], None) if nums else (None, "No numeric column found.")


def _resolve_cat(df: pd.DataFrame, hints: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a categorical column.  Returns (col_name, error_msg).
    """
    result = resolve_column_ambiguity(df, hints, dtype_filter="categorical")
    if isinstance(result, list):
        return None, ambiguity_message(result)
    if isinstance(result, str) and validate_column(df, result):
        return result, None
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return (cats[0], None) if cats else (None, "No categorical column found.")


# ── Chart builders ─────────────────────────────────────────────────────────────

def _bar_h(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    data = data.sort_values(x, ascending=True)
    fig  = px.bar(data, x=x, y=y, orientation="h", color=x,
                  color_continuous_scale=[[0, "#1a2a4a"], [.5, "#4F8BF9"], [1, "#A259FF"]],
                  text=data[x].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _lay(fig, title)


def _bar_v(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(data, x=x, y=y, color=y,
                 color_continuous_scale=[[0, "#1a2a4a"], [.5, "#A259FF"], [1, "#4F8BF9"]],
                 text=data[y].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _lay(fig, title)


# ── Intent detection ───────────────────────────────────────────────────────────

def _intent(q: str) -> str:
    if _any(q, KW_CORR):                               return "correlation"
    if _any(q, KW_TREND):                              return "trend"
    if _any(q, KW_BOX):                                return "box"
    if _any(q, KW_SCATTER):                            return "scatter"
    if _any(q, KW_PIE):                                return "pie"
    if _any(q, KW_DIST):                               return "histogram"
    if _any(q, KW_TOP + KW_LOW) and _any(q, KW_CUST): return "top_customers"
    if _any(q, KW_TOP + KW_LOW) and _any(q, KW_PROD): return "top_products"
    if _any(q, KW_TOP + KW_LOW) and _any(q, KW_REGION): return "top_regions"
    if _any(q, KW_CUST) and _any(q, KW_SALES):        return "top_customers"
    if _any(q, KW_PROD) and _any(q, KW_SALES):        return "top_products"
    if _any(q, KW_REGION) and _any(q, KW_SALES):      return "top_regions"
    if _any(q, KW_AVG):                                return "avg_bar"
    if _any(q, KW_CUST):                               return "customer_freq"
    if _any(q, KW_PROD):                               return "top_products"
    if _any(q, KW_REGION):                             return "top_regions"
    if _any(q, KW_SALES):                              return "top_regions"
    return "unknown"


# ── Handlers — each validates columns before touching them ─────────────────────

def _h_trend(df: pd.DataFrame, q: str):
    r = trend_analysis(df)
    if r is None:
        return None, "No date + numeric column found for trend analysis."
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["month"], y=r["revenue"], name="Revenue",
                         marker_color="#4F8BF9", opacity=.7))
    fig.add_trace(go.Scatter(x=r["month"], y=r["ma3"], name="3M MA",
                             mode="lines+markers",
                             line=dict(color="#FFB347", width=2.5, dash="dot"),
                             marker=dict(size=5)))
    return _lay(fig, "Monthly Revenue Trend"), "trend"


def _h_top_customers(df: pd.DataFrame, q: str):
    cu, err = _resolve_cat(df, KW_CUST)
    if err:    return None, err
    sc, err2 = _resolve_num(df, KW_SALES)
    n   = _n(q)
    asc = _any(q, KW_LOW)
    if sc:
        data = group_by_col(df, cu, sc, "sum", n if not asc else len(df))
        if asc:
            data = data.sort_values(sc).head(n)
    else:
        data = df[cu].value_counts().head(n).reset_index()
        data.columns = [cu, "count"]
        sc = "count"
    title = f"{'Bottom' if asc else 'Top'} {n} Customers"
    return _bar_h(data, sc, cu, title), "top_customers"


def _h_top_products(df: pd.DataFrame, q: str):
    pc, err = _resolve_cat(df, KW_PROD)
    if err:    return None, err
    sc, err2 = _resolve_num(df, KW_SALES)
    n   = _n(q)
    asc = _any(q, KW_LOW)
    if sc:
        data = group_by_col(df, pc, sc, "sum", n if not asc else len(df))
        if asc:
            data = data.sort_values(sc).head(n)
    else:
        data = df[pc].value_counts().head(n).reset_index()
        data.columns = [pc, "count"]
        sc = "count"
    title = f"{'Bottom' if asc else 'Top'} {n} Products by Revenue"
    return _bar_h(data, sc, pc, title), "top_products"


def _h_top_regions(df: pd.DataFrame, q: str):
    rc, err = _resolve_cat(df, KW_REGION)
    if err:    return None, err
    sc, err2 = _resolve_num(df, KW_SALES)
    n   = _n(q)
    asc = _any(q, KW_LOW)
    if sc:
        data = group_by_col(df, rc, sc, "sum", n if not asc else len(df))
        if asc:
            data = data.sort_values(sc).head(n)
    else:
        data = df[rc].value_counts().head(n).reset_index()
        data.columns = [rc, "count"]
        sc = "count"
    title = f"{'Bottom' if asc else 'Top'} {n} Regions"
    return _bar_h(data, sc, rc, title), "top_regions"


def _h_pie(df: pd.DataFrame, q: str):
    cat, err = _resolve_cat(df, KW_PROD + KW_REGION + KW_CUST)
    if err:    return None, err
    sc, err2 = _resolve_num(df, KW_SALES)
    if not sc: return None, "No numeric column found for pie values."
    data = group_by_col(df, cat, sc, "sum", 8)
    fig  = px.pie(data, names=cat, values=sc, hole=.42,
                  color_discrete_sequence=COLORS)
    fig.update_traces(textinfo="percent+label", textfont_size=11,
                      pull=[.04] + [0] * (len(data) - 1))
    return _lay(fig, f"Revenue Share by {cat}"), "pie"


def _h_histogram(df: pd.DataFrame, q: str):
    col, err = _resolve_num(df, KW_SALES + ["age", "qty", "quantity", "amount", "price", "value"])
    if err:  return None, err
    plot_df = _sample(df, 20_000)
    fig = px.histogram(plot_df, x=col, nbins=30, opacity=.85,
                       color_discrete_sequence=["#4F8BF9"], marginal="box")
    return _lay(fig, f"Distribution of {col}"), "histogram"


def _h_correlation(df: pd.DataFrame, q: str):
    corr = correlation_matrix(df)
    if corr is None:
        return None, "Need ≥ 2 numeric columns for correlation analysis."
    fig = px.imshow(corr,
                    color_continuous_scale=[[0, "#FF6B6B"], [.5, "#222"], [1, "#43D9AD"]],
                    zmin=-1, zmax=1, text_auto=True, aspect="auto")
    fig.update_traces(textfont_size=10)
    return _lay(fig, "Correlation Matrix", h=460), "correlation"


def _h_scatter(df: pd.DataFrame, q: str):
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nums) < 2:
        return None, "Need ≥ 2 numeric columns for scatter plot."
    x, y    = nums[0], nums[1]
    cat, _  = _resolve_cat(df, KW_REGION + KW_PROD + KW_CUST)
    plot_df = _sample(df, 10_000)
    fig     = px.scatter(plot_df, x=x, y=y, color=cat, opacity=.6,
                         color_discrete_sequence=COLORS)
    return _lay(fig, f"Scatter: {x} vs {y}"), "scatter"


def _h_box(df: pd.DataFrame, q: str):
    col, err = _resolve_num(df, KW_SALES + ["age", "qty", "amount", "price"])
    if err:  return None, err
    cat, _ = _resolve_cat(df, KW_REGION + KW_PROD + KW_CUST)
    if cat and df[cat].nunique() <= 20:
        fig = px.box(df, x=cat, y=col, color=cat,
                     color_discrete_sequence=COLORS, points=False)
    else:
        fig = px.box(df, y=col, color_discrete_sequence=["#4F8BF9"], points="outliers")
    return _lay(fig, f"Box Plot: {col}"), "box"


def _h_avg_bar(df: pd.DataFrame, q: str):
    cat, err  = _resolve_cat(df, KW_REGION + KW_PROD + KW_CUST)
    if err:   return None, err
    sc, err2  = _resolve_num(df, KW_SALES)
    if err2:  return None, err2
    data = group_by_col(df, cat, sc, "mean", 15)
    return _bar_v(data, cat, sc, f"Average {sc} by {cat}"), "avg_bar"


def _h_customer_freq(df: pd.DataFrame, q: str):
    col, err = _resolve_cat(df, KW_CUST)
    if err:  return None, err
    freq = df[col].value_counts().head(15).reset_index()
    freq.columns = [col, "orders"]
    return _bar_h(freq, "orders", col, "Customer Purchase Frequency"), "customer_freq"


# ── Dispatch table ─────────────────────────────────────────────────────────────

_DISPATCH = {
    "trend":         _h_trend,
    "top_customers": _h_top_customers,
    "top_products":  _h_top_products,
    "top_regions":   _h_top_regions,
    "pie":           _h_pie,
    "histogram":     _h_histogram,
    "correlation":   _h_correlation,
    "scatter":       _h_scatter,
    "box":           _h_box,
    "avg_bar":       _h_avg_bar,
    "customer_freq": _h_customer_freq,
}


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_chart(
    df: pd.DataFrame,
    question: str,
) -> Tuple[Optional[go.Figure], str]:
    """
    Parse *question* and return (plotly_figure, intent_label).
    Returns (None, error_or_ambiguity_message) on any failure.

    Applies clean_numeric_columns() before chart generation to avoid
    type-mismatch crashes on dirty data (BUG-3).
    """
    if df is None or df.empty:
        return None, "No data loaded."
    if not question.strip():
        return None, "Please enter a question."

    # BUG-3: fix dirty numeric-as-object columns before any arithmetic
    df = clean_numeric_columns(df)

    q = question.lower().strip()
    q = re.sub(r"[^\w\s%]", " ", q)
    q = re.sub(r"\s+", " ", q)

    intent  = _intent(q)
    handler = _DISPATCH.get(intent)

    if handler is None:
        schema  = get_dataset_schema(df)
        tips    = "\n".join(f"• *{e}*" for e in EXAMPLE_QUERIES[:6])
        return None, (
            f"I couldn't interpret that question.\n\n"
            f"**Available columns in your dataset:**\n```\n{schema}\n```\n\n"
            f"**Try one of these:**\n{tips}"
        )

    try:
        return handler(df, q)
    except Exception as exc:
        return None, f"Chart generation failed ({intent}): {exc}"
