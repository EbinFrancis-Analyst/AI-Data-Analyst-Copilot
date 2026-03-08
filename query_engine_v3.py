"""
query_engine_v3.py
Natural Language Chart Generator (v3 — performance-optimised)
Parses plain-English questions and returns Plotly figures.
No external API required — pure rule-based intent detection.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics_engine import group_by_col, trend_analysis, correlation_matrix

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS (shared with dashboard)
# ─────────────────────────────────────────────────────────────────────────────

COLOR_SEQ = ["#4F8BF9","#A259FF","#43D9AD","#FFB347","#FF6B6B",
             "#87CEEB","#FF69B4","#98FB98","#DDA0DD","#F0E68C"]
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#E0E0E0", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="#1E1E2E", bordercolor="#444",
                    font=dict(size=12, color="#fff")),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
               zeroline=False, linecolor="rgba(255,255,255,0.3)"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
               zeroline=False, linecolor="rgba(255,255,255,0.3)"),
)

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD DICTIONARIES
# ─────────────────────────────────────────────────────────────────────────────

KW_SALES   = ["revenue","sales","income","amount","price","earning","total"]
KW_TOP     = ["top","best","highest","most","leading","greatest","maximum"]
KW_LOW     = ["lowest","worst","least","bottom","minimum","declining"]
KW_AVG     = ["average","mean","avg","typical"]
KW_TREND   = ["trend","over time","monthly","time series","growth","timeline"]
KW_DIST    = ["distribution","spread","histogram","range","breakdown"]
KW_CORR    = ["correlation","relationship","heatmap","related","between"]
KW_PIE     = ["share","pie","proportion","percentage","contribution","percent"]
KW_CUST    = ["customer","client","buyer","user","consumer"]
KW_PROD    = ["product","item","goods","sku","merchandise"]
KW_REGION  = ["city","region","state","location","area","zone","district"]
KW_SCATTER = ["scatter","compare","vs","versus","against"]
KW_BOX     = ["box","boxplot","quartile","outlier","iqr"]


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(df: pd.DataFrame, kws: List[str]) -> Optional[str]:
    """Find the best matching column for a list of keywords."""
    cl = {c.lower(): c for c in df.columns}
    for kw in kws:
        if kw in cl:
            return cl[kw]
    for kw in kws:
        for name_l, name in cl.items():
            if kw in name_l:
                return name
    return None


def _num_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _resolve(df, hints)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        return col
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[0] if nums else None


def _cat_col(df: pd.DataFrame, hints: List[str]) -> Optional[str]:
    col = _resolve(df, hints)
    if col and not pd.api.types.is_numeric_dtype(df[col]):
        return col
    cats = df.select_dtypes(include=["object","category"]).columns.tolist()
    return cats[0] if cats else None


def _date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if any(k in col.lower() for k in ["date","time","month","year","day"]):
            try:
                pd.to_datetime(df[col].dropna().head(10), errors="raise")
                return col
            except Exception:
                pass
    return None


def _any(q: str, kws: List[str]) -> bool:
    return any(kw in q for kw in kws)


def _extract_n(q: str, default: int = 10) -> int:
    m = re.search(r"top\s+(\d+)", q)
    if m:
        return int(m.group(1))
    words = {"one":1,"two":2,"three":3,"four":4,"five":5,
             "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    for w, v in words.items():
        if f"top {w}" in q:
            return v
    return default


def _layout(fig: go.Figure, title: str = "", h: int = 420) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT, height=h,
                      title=dict(text=f"<b>{title}</b>",
                                 font=dict(size=14, color="#E0E0E0"), x=0.02))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_intent(q: str) -> str:
    """Map a normalised question to a chart-intent string."""
    if _any(q, KW_CORR):                                  return "correlation_heatmap"
    if _any(q, KW_TREND):                                  return "trend_line"
    if _any(q, KW_SCATTER):                                return "scatter"
    if _any(q, KW_BOX):                                    return "box_plot"
    if _any(q, KW_PIE):                                    return "pie_share"
    if _any(q, KW_DIST):                                   return "histogram"

    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_CUST):       return "top_customers"
    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_PROD):       return "top_products"
    if _any(q, KW_TOP+KW_LOW) and _any(q, KW_REGION):     return "top_regions"

    if _any(q, KW_CUST) and _any(q, KW_SALES):            return "customer_revenue"
    if _any(q, KW_PROD) and _any(q, KW_SALES):            return "product_revenue"
    if _any(q, KW_REGION) and _any(q, KW_SALES):          return "region_revenue"

    if _any(q, KW_CUST):                                   return "customer_freq"
    if _any(q, KW_PROD):                                   return "product_freq"
    if _any(q, KW_REGION):                                 return "region_revenue"

    if _any(q, KW_AVG):                                    return "avg_bar"
    if _any(q, KW_SALES):                                  return "region_revenue"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_h(data: pd.DataFrame, x: str, y: str, title: str,
           color_col: Optional[str] = None) -> go.Figure:
    data = data.sort_values(x, ascending=True)
    fig  = px.bar(data, x=x, y=y, orientation="h",
                  color=color_col or x,
                  color_continuous_scale=[[0,"#1a2a4a"],[0.5,"#4F8BF9"],[1,"#A259FF"]],
                  text=data[x].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _layout(fig, title)


def _bar_v(data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(data, x=x, y=y, color=y,
                 color_continuous_scale=[[0,"#1a2a4a"],[0.5,"#A259FF"],[1,"#4F8BF9"]],
                 text=data[y].apply(lambda v: f"{v:,.0f}"))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_coloraxes(showscale=False)
    return _layout(fig, title)


# ─────────────────────────────────────────────────────────────────────────────
# INTENT → FIGURE HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

def _chart_trend(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    result = trend_analysis(df)
    if result is None:
        return None, "No date + numeric column found for trend."
    fig = go.Figure()
    fig.add_trace(go.Bar(x=result["month"], y=result["revenue"],
                         name="Revenue", marker_color="#4F8BF9", opacity=0.7))
    fig.add_trace(go.Scatter(x=result["month"], y=result["ma3"],
                             name="3M MA", mode="lines+markers",
                             line=dict(color="#FFB347", width=2.5, dash="dot")))
    return _layout(fig, "Monthly Revenue Trend"), "trend_line"


def _chart_top_customers(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    n    = _extract_n(q, 10)
    col  = _cat_col(df, KW_CUST)
    vcol = _num_col(df, KW_SALES)
    if not col:
        return None, "No customer column found."
    data = group_by_col(df, col, vcol, "sum", n) if vcol else \
           df[col].value_counts().head(n).reset_index().rename(columns={col:"count"})
    y_col = vcol if vcol else "count"
    return _bar_h(data, y_col, col, f"Top {n} Customers"), "top_customers"


def _chart_top_products(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    n    = _extract_n(q, 10)
    col  = _cat_col(df, KW_PROD)
    vcol = _num_col(df, KW_SALES)
    if not col:
        return None, "No product column found."
    data = group_by_col(df, col, vcol, "sum", n) if vcol else \
           df[col].value_counts().head(n).reset_index()
    y_col = vcol if vcol else "count"
    return _bar_h(data, y_col, col, f"Top {n} Products by Revenue"), "top_products"


def _chart_top_regions(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    n    = _extract_n(q, 10)
    col  = _cat_col(df, KW_REGION)
    vcol = _num_col(df, KW_SALES)
    if not col:
        return None, "No region column found."
    ascending = _any(q, KW_LOW)
    data = group_by_col(df, col, vcol, "sum", n if not ascending else len(df)) if vcol else \
           df[col].value_counts().head(n).reset_index()
    if ascending and vcol:
        data = data.sort_values(vcol).head(n)
    y_col = vcol if vcol else "count"
    title = f"{'Bottom' if ascending else 'Top'} {n} Regions by Revenue"
    return _bar_h(data, y_col, col, title), "top_regions"


def _chart_pie(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    cat  = _cat_col(df, KW_PROD + KW_REGION + KW_CUST)
    vcol = _num_col(df, KW_SALES)
    if not cat or not vcol:
        return None, "Need a category and numeric column."
    data = group_by_col(df, cat, vcol, "sum", 8)
    fig  = px.pie(data, names=cat, values=vcol, hole=0.42,
                  color_discrete_sequence=COLOR_SEQ)
    fig.update_traces(textinfo="percent+label", textfont_size=11,
                      pull=[0.04]+[0]*(len(data)-1))
    return _layout(fig, f"Revenue Share by {cat}"), "pie_share"


def _chart_histogram(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    col = _num_col(df, KW_SALES + ["age","qty","quantity","amount","price"])
    if not col:
        return None, "No numeric column found."
    fig = px.histogram(df, x=col, nbins=30, opacity=0.85,
                       color_discrete_sequence=["#4F8BF9"],
                       marginal="box")
    return _layout(fig, f"Distribution of {col}"), "histogram"


def _chart_correlation(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    corr = correlation_matrix(df)
    if corr is None:
        return None, "Need at least 2 numeric columns."
    fig  = px.imshow(corr, color_continuous_scale=[[0,"#FF6B6B"],[0.5,"#222"],[1,"#43D9AD"]],
                     zmin=-1, zmax=1, text_auto=True, aspect="auto")
    fig.update_traces(textfont_size=10)
    return _layout(fig, "Correlation Matrix", h=460), "correlation_heatmap"


def _chart_scatter(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(nums) < 2:
        return None, "Need at least 2 numeric columns for scatter."
    x, y = nums[0], nums[1]
    cat  = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    fig  = px.scatter(df.sample(min(5000, len(df)), random_state=42),
                      x=x, y=y, color=cat,
                      opacity=0.6, trendline="ols",
                      color_discrete_sequence=COLOR_SEQ,
                      trendline_color_override="#FFB347")
    return _layout(fig, f"Scatter: {x} vs {y}"), "scatter"


def _chart_box(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    col = _num_col(df, KW_SALES + ["age","qty","amount","price"])
    cat = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    if not col:
        return None, "No numeric column found."
    if cat and df[cat].nunique() <= 20:
        fig = px.box(df, x=cat, y=col, color=cat,
                     color_discrete_sequence=COLOR_SEQ, points=False)
    else:
        fig = px.box(df, y=col, color_discrete_sequence=["#4F8BF9"], points="outliers")
    return _layout(fig, f"Box Plot: {col}"), "box_plot"


def _chart_avg_bar(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    cat  = _cat_col(df, KW_REGION + KW_PROD + KW_CUST)
    vcol = _num_col(df, KW_SALES)
    if not cat or not vcol:
        return None, "Need a category and numeric column."
    data = group_by_col(df, cat, vcol, "mean", 15)
    return _bar_v(data, cat, vcol, f"Average {vcol} by {cat}"), "avg_bar"


def _chart_customer_freq(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    col = _cat_col(df, KW_CUST)
    if not col:
        return None, "No customer column found."
    freq = df[col].value_counts().head(15).reset_index()
    freq.columns = [col, "orders"]
    return _bar_h(freq, "orders", col, "Customer Purchase Frequency"), "customer_freq"


def _chart_product_freq(df: pd.DataFrame, q: str) -> Tuple[go.Figure, str]:
    col = _cat_col(df, KW_PROD)
    if not col:
        return None, "No product column found."
    freq = df[col].value_counts().head(15).reset_index()
    freq.columns = [col, "count"]
    return _bar_h(freq, "count", col, "Product Frequency"), "product_freq"


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCH = {
    "trend_line":           _chart_trend,
    "top_customers":        _chart_top_customers,
    "top_products":         _chart_top_products,
    "top_regions":          _chart_top_regions,
    "pie_share":            _chart_pie,
    "histogram":            _chart_histogram,
    "correlation_heatmap":  _chart_correlation,
    "scatter":              _chart_scatter,
    "box_plot":             _chart_box,
    "avg_bar":              _chart_avg_bar,
    "customer_revenue":     _chart_top_customers,
    "product_revenue":      _chart_top_products,
    "region_revenue":       _chart_top_regions,
    "customer_freq":        _chart_customer_freq,
    "product_freq":         _chart_product_freq,
}

EXAMPLE_QUERIES = [
    "Show revenue by city",
    "Top 10 products by sales",
    "Monthly sales trend",
    "Customer distribution",
    "Revenue share by category",
    "Correlation heatmap",
    "Age distribution histogram",
    "Scatter: amount vs quantity",
    "Top 5 customers by revenue",
    "Lowest performing regions",
    "Average order value by product",
    "Box plot of sales",
]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, question: str) -> Tuple[Optional[go.Figure], str]:
    """
    Parse *question* and return (plotly_figure, description_string).
    Returns (None, error_message) if intent cannot be resolved.

    Args:
        df:       DataFrame to visualise
        question: Plain-English question string

    Returns:
        (Figure, label) | (None, error_message)
    """
    if df is None or df.empty:
        return None, "No data loaded."
    if not question.strip():
        return None, "Please enter a question."

    q = question.lower().strip()
    q = re.sub(r"[^\w\s%]", " ", q)
    q = re.sub(r"\s+", " ", q)

    intent  = _detect_intent(q)
    handler = _DISPATCH.get(intent)

    if handler is None:
        return None, (
            "I couldn't interpret that question. Try:\n"
            + "\n".join(f"• *{e}*" for e in EXAMPLE_QUERIES[:6])
        )

    try:
        return handler(df, q)
    except Exception as exc:
        return None, f"Chart generation failed ({intent}): {exc}"
